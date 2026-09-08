"""Finalize 节点：纯代码收敛最终回答与可追踪 Citation。"""

from typing import Any

from langgraph.runtime import Runtime

from backend.app.agent.schemas import Citation
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState

from ..evidence import _no_match_safe_answer, _no_match_violations


class FinalizeNode:
    async def finalize(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        """纯代码最终出口：收敛回答、执行安全边界并生成可追踪 Citation。

        本节点不调用模型或 MCP。它检查 no_match、复核结果和引用归属，必要时替换
        为安全模板；只有草稿 claims 实际引用且存在于 EvidencePacket 的 chunk 才
        会进入 citations，随后由 AgentRuntime/AgentRunManager 输出并持久化。
        """

        # 步骤 1：确定正文来源。闲聊/澄清由 Analyst 直接写 final_answer；法律咨询
        # 通常使用 CounselDraft.answer。此时只在内存 State 中，尚未保存业务消息。
        answer = state["final_answer"] or (state["counsel_draft"].answer if state["counsel_draft"] else "")

        # 步骤 2：读取 Research 和 Review 产物，后续所有安全判断都使用结构化对象，
        # 不根据自然语言猜测工具是否成功或 Reviewer 是否批准。
        packet = state["evidence_packet"]
        review = state["review_result"]

        # 步骤 3A：no_match 必须同时满足“Reviewer 已批准”和确定性无幻觉检查。
        # 任一条件失败都用固定安全模板覆盖草稿，而不是把越界内容继续输出。
        if packet and packet.retrieval_status == "no_match":
            violations = _no_match_violations(answer)
            if not review or not review.approved or violations:
                answer = _no_match_safe_answer(state)

        # 步骤 3B：非 no_match 且 Reviewer 明确发现无依据论断、遗漏、引用错误或
        # 矛盾时，不输出原草稿；只列出已经核验的证据摘要和重新咨询建议。
        elif review and not review.approved and (
            review.unsupported_claims
            or review.missing_issue_ids
            or review.citation_errors
            or review.contradictions
        ):
            evidence_lines = [
                f"- 《{item.law_name}》{item.article_number}：{item.content[:180]}"
                for item in (packet.evidence_items if packet else [])
            ]
            answer = (
                "本轮自动复核未通过，因此未输出可能缺乏依据的确定性法律结论。\n\n"
                "### 已核验到的材料\n"
                + ("\n".join(evidence_lines) if evidence_lines else "- 暂未取得足够的法规依据。")
                + "\n\n建议补充案情后重新咨询，或由专业律师结合完整材料审查。"
            )

        # 步骤 4：初始化 Citation 聚合器。seen_documents 实际存放 chunk/document
        # 证据键，用于避免多个 claim 引用同一 chunk 时重复展示。
        citations: list[Citation] = []
        seen_documents: set[str] = set()

        # 步骤 5：检索候选不等于用户可见引用。这里只收集 Counsel claims 明确使用
        # 的 chunk_id，以及兼容旧 Fixture 的 document_id。
        referenced_ids = {
            evidence_id
            for claim in (state["counsel_draft"].claims if state["counsel_draft"] else [])
            for evidence_id in [*claim.evidence_chunk_ids, *claim.evidence_document_ids]
        }
        # 步骤 6：统计同一原始法条在 EvidencePacket 中有几个 chunk。只有计数为 1
        # 时，旧 document_id 才能无歧义映射到具体片段。
        document_counts: dict[str, int] = {}
        for item in packet.evidence_items if packet else []:
            document_counts[item.document_id] = document_counts.get(item.document_id, 0) + 1
        # 步骤 7：只遍历 matched EvidencePacket。Citation 的法名、条号和摘录来自
        # 服务端验证后的 EvidenceItem，不能从回答
        # 文本反向猜测，因此可以追溯到本轮 MCP 返回的具体 chunk。
        for item in packet.evidence_items if packet and packet.retrieval_status == "matched" else []:
            # 步骤 7A：判断当前 chunk 是否被草稿实际引用。旧 document_id 只有单
            # chunk 时才兼容，避免长法条错误引用第一个片段。
            legacy_match = item.document_id in referenced_ids and document_counts[item.document_id] == 1
            if item.chunk_id not in referenced_ids and not legacy_match:
                continue

            # 步骤 7B：去重证据键，同一个法条片段最多生成一条 Citation。
            evidence_key = item.chunk_id or item.document_id
            if evidence_key in seen_documents:
                continue
            seen_documents.add(evidence_key)
            # 步骤 7C：构造用户可见引用，摘录限制为 240 字；完整检索候选和内部
            # 分数不会进入最终 Citation/SSE。
            citations.append(Citation(document_id=item.document_id, chunk_id=item.chunk_id, law_name=item.law_name, article_number=item.article_number, quoted_excerpt=item.content[:240], data_version=item.data_version))

        # 步骤 8：发布最终阶段状态。正文和 citations 仍由节点返回值交给 Runtime，
        # stream_writer 在这里仅表示“图分析完成”。
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "completed", "message": "分析已完成"}})

        # 步骤 9：返回最终 State 增量。LangGraph 随后到达 END 并写最终 Checkpoint；
        # AgentRuntime 才把 final_answer 切成安全 token，AgentRunManager 再幂等落库。
        return {
            "final_answer": answer,
            "citations": citations,
            "model_call_count": runtime.context.metrics.model_call_count,
            "tool_call_count": runtime.context.metrics.tool_call_count,
            "tool_trajectory": list(runtime.context.metrics.tool_trajectory),
        }