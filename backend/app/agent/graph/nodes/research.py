"""Legal Researcher 节点：调用 MCP 检索工具并生成可验证 EvidencePacket。"""

import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from backend.app.agent.schemas import (
    AgentError,
    EvidenceItem,
    EvidencePacket,
    EvidenceSelectionResult,
    UnresolvedIssue,
)
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.logging import audit, summary

from ..evidence import _authoritative_evidence, _payload, _tool_documents
from ..prompts import EVIDENCE_SELECTOR_PROMPT, RESEARCH_PROMPT


class ResearchNode:
    async def legal_researcher(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        """运行唯一允许调用 MCP 的 LangChain Agent，并生成可验证 EvidencePacket。

        输入：CaseAnalysis 中的争议点/研究任务，以及补检索时的 ReviewResult。模型
        自主决定检索次数、查询和何时停止；``response_format=ToolStrategy`` 强制它
        最终必须通过结构化工具汇报 EvidencePacket，而不是自由文本。工具返回的候选
        必须再次映射为权威 chunk，模型不能自行提供法条正文或证据元数据。
        """

        # 步骤 1：发布“研究中”状态。此事件只描述阶段，不包含查询参数或法规正文。
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "legal_researcher", "status": "researching", "message": "正在检索法律依据"}})

        # 步骤 2：读取 Analyst 产物；正常路由下它一定存在，但类型仍允许兜底。
        analysis = state["case_analysis"]

        # 步骤 3：处理“从未发现到 MCP 工具”的降级。Runtime 仍可编译无工具 Graph，
        # 这里明确返回 tool_unavailable，而不是把能力故障误写成正常 no_match。
        if self.research_agent is None:
            issues = [UnresolvedIssue(issue_id=f"issue-{index + 1}", description=item) for index, item in enumerate(analysis.legal_issues if analysis else [])]
            return {"evidence_packet": EvidencePacket(retrieval_status="tool_unavailable", research_tasks=analysis.research_tasks if analysis else [], unresolved_issues=issues, research_summary="法律检索工具当前不可用。")}

        # 步骤 4：取得当前 invocation 的身份、计数和 stream writer 上下文，并开始
        # 节点耗时审计。context 属于本轮请求，不保存到共享 research_agent 实例。
        context = runtime.context
        started = time.perf_counter()
        audit("agent.node.started", status="started", agent="legal_researcher", **context.audit_fields)

        # 步骤 5：把 Graph State 转成模型可消费的普通字典。若 Reviewer 要求补检索，
        # 只追加明确的 supplemental_instruction，不允许模型自由读取整个 Checkpoint。
        research_payload = _payload(state)
        if state["review_result"] and state["review_result"].revision_instruction:
            research_payload["supplemental_instruction"] = state["review_result"].revision_instruction
        try:
            # 步骤 6：初始化本次 LangChain 子 Agent 流的聚合变量。structured_response
            # 保存 ToolStrategy 校验通过的 EvidencePacket；candidates 只收集真实检索
            # 工具（self.tool_names）返回的 ToolMessage，不包含 ToolStrategy 自动生成
            # 的“结构化输出”ToolMessage；seen_tool_messages 防止 updates 重复；两个
            # 布尔值用于区分空结果与工具故障。
            structured_response: EvidencePacket | None = None
            candidates: list[dict[str, Any]] = []
            seen_tool_messages: set[str] = set()
            successful_tool_result = False
            failed_tool_result = False
            # 步骤 7：启动 LangChain Agent 内部模型—工具循环。传入 SystemMessage
            # 约束研究角色，HumanMessage 携带结构化任务；context 供 Middleware 读取
            # 身份/计数。updates 携带模型/工具 Message 与 structured_response，custom
            # 携带 Middleware 写出的安全状态；完整工具参数、法规正文和推理不会直接
            # 转发给客户端。
            async for part in self.research_agent.astream(
                {"messages": [SystemMessage(content=RESEARCH_PROMPT), HumanMessage(content=json.dumps(research_payload, ensure_ascii=False, default=str))]},
                context=context,
                stream_mode=["updates", "custom"],
                version="v2",
            ):
                # 步骤 8A：custom event 已由 ToolAuditMiddleware 做过安全裁剪，可继续
                # 写入外层 Graph stream；它通常表示工具开始、成功、失败或超时。
                if part.get("type") == "custom" and isinstance(part.get("data"), dict):
                    runtime.stream_writer(part["data"])

                # 步骤 8B：updates 是 LangChain 子 Agent 的内部状态增量。只读取其中
                # Message 和 structured_response，不把完整 update 或内部 Agent State
                # 暴露给外层调用者。
                elif part.get("type") == "updates" and isinstance(part.get("data"), dict):
                    for update in part["data"].values():
                        if not isinstance(update, dict):
                            continue
                        if update.get("structured_response") is not None:
                            structured_response = update["structured_response"]
                        for message in update.get("messages", []):
                            # 只有真实检索工具（search_laws/get_law_article）的
                            # ToolMessage 才是候选来源；ToolStrategy 为完成结构化
                            # 输出而自动追加的 ToolMessage 用 name 区分，跳过即可。
                            if not isinstance(message, ToolMessage) or message.name not in self.tool_names:
                                continue
                            # state update 可能重复包含旧 ToolMessage，按 tool_call_id
                            # 去重，避免同一次 MCP 结果被重复收集和审计。
                            message_id = str(message.tool_call_id)
                            if message_id in seen_tool_messages:
                                continue
                            seen_tool_messages.add(message_id)
                            # 错误 ToolMessage 不解析为法规候选，但会参与最终 tool_error
                            # 判断；成功 ToolMessage 才递归提取 document/chunk 元数据。
                            if message.status == "error":
                                failed_tool_result = True
                            else:
                                successful_tool_result = True
                                candidates.extend(_tool_documents(message))

            # 步骤 9：没有 structured_response 说明 LangChain 子 Agent 未能通过
            # ToolStrategy 收敛（例如撞到 ModelCallLimitMiddleware 的硬上限），转入
            # 统一 exception 分支生成 tool_error EvidencePacket。
            if structured_response is None:
                raise RuntimeError("法律研究 Agent 未返回结构化结果")

            # 步骤 10：structured_response 已经过 Pydantic 校验，直接作为 raw_packet；
            # 仍需用真实 ToolMessage 候选回填元数据，绝不直接信任模型生成的正文。
            raw_packet = structured_response

            # 步骤 11：兼容模型只返回 accepted_chunk_ids 的情况。先用候选构造最小
            # EvidenceItem 选择声明，后续仍由 _authoritative_evidence 回填权威字段。
            if not raw_packet.evidence_items and raw_packet.accepted_chunk_ids:
                selected_ids = set(raw_packet.accepted_chunk_ids)
                raw_packet = raw_packet.model_copy(update={
                    "evidence_items": [
                        EvidenceItem(
                            document_id=str(item.get("document_id", "")),
                            chunk_id=str(item.get("chunk_id", "")),
                            supports_issue_ids=[task.issue_id for task in analysis.research_tasks],
                        )
                        for item in candidates
                        if str(item.get("chunk_id", "")) in selected_ids
                    ]
                })
            # 步骤 12：执行第一次权威映射。伪造 ID、歧义 document_id 和不在本轮
            # ToolMessage 中的证据都会被删除。
            accepted = _authoritative_evidence(raw_packet, candidates)
            if candidates and not accepted and not raw_packet.rejected_candidates:
                # 步骤 13（按需）：已有候选但模型既未接受也未拒绝时，追加一次无工具
                # 受限选择。Selector 只能选择已有 chunk，不会重新搜索或产生新证据。
                selection = await self._invoke_json(
                    runtime,
                    "evidence_selector",
                    EVIDENCE_SELECTOR_PROMPT,
                    {
                        "legal_issues": analysis.legal_issues if analysis else [],
                        "candidates": [
                            {
                                "chunk_id": item.get("chunk_id"),
                                "law_name": item.get("law_name"),
                                "article_number": item.get("article_number"),
                                "content": item.get("content"),
                            }
                            for item in candidates
                        ],
                    },
                    EvidenceSelectionResult,
                )
                # 步骤 13A：即使 Selector 通过 Schema，也要用候选 ID 集合再次裁剪，
                # 防止模型输出本轮不存在的 chunk_id。
                valid_ids = {str(item.get("chunk_id", "")) for item in candidates}
                accepted_ids = {
                    item for item in selection.accepted_chunk_ids if item in valid_ids
                }
                valid_rejections = [
                    item for item in selection.rejected_candidates if item.chunk_id in valid_ids
                ]
                selected_packet = raw_packet.model_copy(update={
                    "evidence_items": [
                        EvidenceItem(
                            document_id=str(item.get("document_id", "")),
                            chunk_id=str(item.get("chunk_id", "")),
                            supports_issue_ids=[task.issue_id for task in analysis.research_tasks],
                        )
                        for item in candidates
                        if str(item.get("chunk_id", "")) in accepted_ids
                    ],
                    "accepted_chunk_ids": list(accepted_ids),
                    "rejected_candidates": valid_rejections,
                })
                # 步骤 13B：把 Selector 结果再次映射为带真实正文/条号的证据对象。
                accepted = _authoritative_evidence(selected_packet, candidates)
                raw_packet = selected_packet

            # 步骤 14：状态由真实工具结果和权威证据共同决定：有采纳证据为 matched；仅有
            # 失败 ToolMessage 为 tool_error；其余 evidence_items=[] 是正常 no_match。
            if accepted:
                retrieval_status = "matched"
            elif failed_tool_result and not successful_tool_result:
                retrieval_status = "tool_error"
            else:
                retrieval_status = "no_match"
            # 步骤 15：保证 no_match/tool_error 至少携带一个可解释的未解决问题，
            # 供 Counsel 说明证据限制；这不是异常，也不触发第二次相同查询。
            # 没有候选时收尾调用被跳过，raw_packet.unresolved_issues 恒为空，
            # 因此这里用 Analyst 已识别的争议点兜底，而不是笼统的单条占位说明。
            unresolved = raw_packet.unresolved_issues
            if not unresolved and retrieval_status in {"no_match", "tool_error"}:
                unresolved = [
                    UnresolvedIssue(issue_id=f"issue-{index + 1}", description=item)
                    for index, item in enumerate(analysis.legal_issues if analysis else [])
                ] or [UnresolvedIssue(issue_id="issue-1", description="当前法规库未检索到可直接引用的依据")]
            # 步骤 16：归一化最终 EvidencePacket。candidate_status 描述是否召回候选，
            # evidence_status 描述候选是否被接受，retrieval_status 描述业务终态。
            # EvidencePacket 是 Counsel/Reviewer 唯一允许引用的法规事实源。空结果
            # 可以正常继续回答，不会仅因候选未被采纳而触发异常或重复检索。
            packet = raw_packet.model_copy(update={
                "retrieval_status": retrieval_status,
                "candidate_status": "matched" if candidates else "no_match",
                "evidence_status": (
                    "accepted"
                    if accepted
                    else "error"
                    if failed_tool_result and not successful_tool_result
                    else "rejected"
                ),
                "evidence_items": accepted,
                "accepted_chunk_ids": [item.chunk_id for item in accepted],
                "unresolved_issues": unresolved,
                "research_summary": raw_packet.research_summary or (
                    "检索正常完成，但未找到可引用法条。" if retrieval_status == "no_match" else raw_packet.research_summary
                ),
            })
            # 步骤 17：按 chunk/document ID 去重后记录节点指标，不写完整法规正文。
            unique_candidates = {str(item.get("chunk_id") or item.get("document_id")) for item in candidates}
            audit(
                "agent.node.completed",
                status="success",
                agent="legal_researcher",
                duration_ms=int((time.perf_counter() - started) * 1000),
                retrieval_status=retrieval_status,
                candidate_count=len(unique_candidates),
                accepted_evidence_count=len(accepted),
                rejected_candidate_count=max(0, len(unique_candidates) - len(accepted)),
                evidence_count=len(accepted),
                no_match_reason=("候选法条未被研究 Agent 认定为可引用依据" if retrieval_status == "no_match" else ""),
                remaining_tool_calls=max(0, self.settings.agent_max_tool_calls - context.metrics.tool_call_count),
                remaining_model_calls=max(0, self.settings.agent_max_model_calls - context.metrics.model_call_count),
                **context.audit_fields,
            )
            # 步骤 18：返回 State 增量。若本次由 Reviewer 回流，retry_count 增加；
            # 调用计数和轨迹同步入 State，确保 Checkpoint 恢复后不归零。
            return {
                "evidence_packet": packet,
                "retry_count": state["retry_count"] + (1 if state["review_result"] else 0),
                "model_call_count": context.metrics.model_call_count,
                "tool_call_count": context.metrics.tool_call_count,
                "tool_trajectory": list(context.metrics.tool_trajectory),
            }
        except Exception as exc:  # noqa: BLE001 - research failures become controlled evidence state
            # 失败步骤：研究节点内部异常被收敛成 tool_error，使后续 Counsel 能明确披露未完成
            # 法规核验；这样工具故障不会让整条咨询任务永久卡住。
            audit("agent.node.failed", level=logging.ERROR, status="failed", agent="legal_researcher", error_type=type(exc).__name__, error=summary(str(exc)), **context.audit_fields)
            issues = [UnresolvedIssue(issue_id=f"issue-{index + 1}", description=item) for index, item in enumerate(analysis.legal_issues if analysis else [])]
            packet = EvidencePacket(retrieval_status="tool_error", research_tasks=analysis.research_tasks if analysis else [], unresolved_issues=issues, research_summary="法律检索执行失败，未能完成法规核验。")
            return {
                "evidence_packet": packet,
                "errors": [*state["errors"], AgentError(agent="legal_researcher", message=str(exc))],
                "retry_count": state["retry_count"] + (1 if state["review_result"] else 0),
                "model_call_count": runtime.context.metrics.model_call_count,
                "tool_call_count": runtime.context.metrics.tool_call_count,
                "tool_trajectory": list(runtime.context.metrics.tool_trajectory),
            }
