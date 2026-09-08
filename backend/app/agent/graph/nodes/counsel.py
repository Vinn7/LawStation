"""Legal Counsel 节点：基于案情与 EvidencePacket 生成待复核法律意见草稿。"""

from typing import Any

from langgraph.runtime import Runtime

from backend.app.agent.schemas import AgentError, CounselDraft
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState

from ..evidence import _no_match_disclosure_present, _no_match_safe_answer, _payload
from ..prompts import COUNSEL_PROMPT


class CounselNode:
    async def legal_counsel(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        """基于案情、记忆快照和 EvidencePacket 生成待复核法律意见草稿。

        输入：CaseAnalysis、EvidencePacket 和 Reviewer 修改要求。输出：CounselDraft
        及 revision_count。该节点无 MCP Tool 权限，草稿也不会在此处直接持久化或
        输出给用户。
        """

        # 步骤 1：读取 Research 证据状态，并把 no_match 单独建模。no_match 表示工具
        # 正常完成但无可引用证据，不等同于 tool_error/tool_unavailable。
        evidence = state["evidence_packet"]
        no_match = bool(evidence and evidence.retrieval_status == "no_match")

        # 步骤 2：根据证据状态发布用户可理解的阶段信息；此事件不会携带草稿正文。
        message = "未检索到可引用法条，将基于案情形成一般性分析" if no_match else "正在形成法律意见"
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "legal_counsel", "status": "drafting", "message": message}})

        # 步骤 3：构造有限模型输入，并显式告诉 Counsel 当前是“基于证据回答”还是
        # “无法条的一般性分析”，避免模型仅根据空数组自行猜测模式。
        payload = _payload(state)
        payload["answer_mode"] = "general_analysis_without_citations" if no_match else "evidence_based"

        # 步骤 4：若 Reviewer 要求改稿，只注入简短 revision_instruction；不把
        # Reviewer 隐藏推理或完整 Checkpoint 暴露给 Counsel。
        if state["review_result"] and state["review_result"].revision_instruction:
            payload["revision_instruction"] = state["review_result"].revision_instruction

        try:
            # 步骤 5：_invoke_json 是无工具的完整消息调用。输入含分析、证据、有限记忆和
            # Fact Override；输出 CounselDraft 此时尚未写入业务 messages 表。
            draft = await self._invoke_json(
                runtime,
                "legal_counsel",
                COUNSEL_PROMPT,
                payload,
                CounselDraft,
            )
            # 步骤 6：no_match 模式强制补充披露并把 confidence 压到 low。即使模型
            # 遗漏了说明，也由代码追加；具体法名/条号仍会在 Reviewer/Finalize 检查。
            if no_match:
                disclosure = "本轮法规检索正常完成，但当前法规库中未检索到可引用法条。以上属于一般性分析，不构成已经过法规核验的确定性法律结论。"
                answer = draft.answer if _no_match_disclosure_present(draft.answer) else f"{draft.answer.rstrip()}\n\n## 检索说明\n\n{disclosure}"
                draft = draft.model_copy(update={"answer": answer, "confidence": "low"})
            # 步骤 7：返回草稿 State 增量。只有由 Reviewer 回流才增加 revision_count，
            # 该计数会随 Checkpoint 保存，确保服务重启后不能再次免费改稿。
            return {
                "counsel_draft": draft,
                "revision_count": state["revision_count"] + (1 if state["review_result"] else 0),
                "model_call_count": runtime.context.metrics.model_call_count,
            }
        except Exception as exc:  # noqa: BLE001 - counsel failures use a safe user-facing fallback
            # 失败步骤 1：no_match 使用确定性安全模板；其他状态仅输出研究摘要和
            # 明确限制，不把异常内容或未经复核的半成品草稿返回用户。
            fallback_answer = _no_match_safe_answer(state) if no_match else (evidence.research_summary if evidence else "暂时无法形成完整法律意见。") + "\n\n当前回答生成失败，建议稍后重试或咨询专业律师。"
            fallback = CounselDraft(answer=fallback_answer, confidence="low", limitations=["回答生成或法规核验未完整完成"])
            # 失败步骤 2：把安全草稿和 AgentError 写回 State，让 Graph 仍进入
            # review_gate；错误用于审计，不直接作为最终正文。
            return {
                "counsel_draft": fallback,
                "errors": [*state["errors"], AgentError(agent="legal_counsel", message=str(exc))],
                "model_call_count": runtime.context.metrics.model_call_count,
            }
