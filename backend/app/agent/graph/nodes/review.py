"""Review 节点：LLM 复核与确定性门禁（Fast Path）。"""

from typing import Any

from langgraph.runtime import Runtime

from backend.app.agent.schemas import AgentError, ReviewResult
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.logging import audit

from ..evidence import _citation_errors, _fact_boundary_errors, _no_match_violations, _payload
from ..prompts import REVIEW_PROMPT


class ReviewNode:
    async def reviewer(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        """执行 Case Analyst 的 LLM 复核阶段，而不是另一个工具型服务。

        ReviewResult 可要求 finalize、research_again 或 revise_draft；路由函数会用
        retry_count/revision_count 强制限制回流次数。
        """

        # 步骤 1：发布复核状态。Reviewer 仍属于 Case Analyst 的复核阶段，因此前端
        # agent 名称沿用 case_analyst，而状态明确为 reviewing。
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "reviewing", "message": "正在核验回答"}})
        try:
            # 步骤 2：执行一次无工具结构化模型调用。输入包含草稿、证据包、分析和
            # 修订计数，输出必须满足 ReviewResult Schema。
            review = await self._invoke_json(
                runtime,
                "case_analyst_reviewer",
                REVIEW_PROMPT,
                _payload(state),
                ReviewResult,
            )
        except Exception as exc:  # noqa: BLE001 - review failures finalize with explicit limitations
            # 步骤 2-失败：Reviewer 自身失败时不无限重试模型，而是生成“未批准但
            # 结束”的 ReviewResult。Finalize 会依据限制输出安全结果。
            review = ReviewResult(approved=False, revision_instruction="自动复核未完成，最终回答应保留风险提示。", next_action="finalize")
            return {
                "review_result": review,
                "errors": [*state["errors"], AgentError(agent="case_analyst_reviewer", message=str(exc))],
                "model_call_count": runtime.context.metrics.model_call_count,
            }
        # 步骤 3：LLM 复核之后仍执行代码级引用校验。Reviewer 的 approved 只是模型
        # 意见，不能覆盖“引用必须属于本轮 EvidencePacket”的硬约束。
        deterministic_errors = _citation_errors(state["counsel_draft"], state["evidence_packet"])
        packet = state["evidence_packet"]

        # 步骤 5：no_match 额外检查是否出现具体法名/条号及是否披露未找到依据。
        if packet and packet.retrieval_status == "no_match":
            deterministic_errors.extend(_no_match_violations(state["counsel_draft"].answer if state["counsel_draft"] else ""))

        # 步骤 6：检查回答是否继续使用已被当前用户消息替换的旧事实。
        deterministic_errors.extend(
            _fact_boundary_errors(
                state,
                state["counsel_draft"].answer if state["counsel_draft"] else "",
            )
        )

        # 步骤 7：任何确定性错误都覆盖模型的批准结论，并要求仅修改草稿；这些问题
        # 不需要重新检索，因为证据集合没有发生变化。
        if deterministic_errors:
            review.approved = False
            review.citation_errors = list(dict.fromkeys([*review.citation_errors, *deterministic_errors]))
            review.next_action = "revise_draft"
            instruction = "删除或改写所有不在 EvidencePacket 中的法条和证据引用。"
            review.revision_instruction = " ".join(
                item for item in (review.revision_instruction, instruction) if item
            )
        # 步骤 8：no_match 没有硬错误时，根据模型指出的无依据论断/矛盾决定改稿；
        # 两者都为空才批准结束，绝不因为 evidence_items=[] 返回 Research。
        elif packet and packet.retrieval_status == "no_match":
            if review.unsupported_claims or review.contradictions:
                review.approved = False
                review.next_action = "revise_draft"
            else:
                review.approved = True
                review.next_action = "finalize"
        # 步骤 9：返回 ReviewResult 和最新模型计数。after_review 随后把 next_action
        # 映射为 finalize、Research 或 Counsel。
        return {
            "review_result": review,
            "model_call_count": runtime.context.metrics.model_call_count,
        }

    async def review_gate(
        self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]
    ) -> dict[str, Any]:
        """用确定性规则决定是否需要额外执行一次 LLM Reviewer。

        只有低风险、no_match、低置信度、无证据引用且通过披露检查的回答可以
        Fast Path。其余情况继续走 Reviewer，以节省简单问题延迟而不降低安全边界。
        """

        # 步骤 1：读取 Gate 所需的三个结构化产物；Gate 本身不调用模型或工具。
        analysis = state["case_analysis"]
        packet = state["evidence_packet"]
        draft = state["counsel_draft"]

        # 步骤 2：skip_reason 命名表示“不能跳过 Reviewer 的原因”。非空就进入 LLM；
        # 空字符串表示所有 Fast Path 条件都通过。
        skip_reason = ""

        # 步骤 3：评测可强制 always-llm，确保 Baseline 与自动 Fast Path 可公平对比。
        if self.settings.agent_review_mode == "always-llm":
            skip_reason = "评测配置要求始终执行模型复核"

        # 步骤 4：Reviewer 要求改稿后必须再次复核，不能让第二版草稿绕过安全检查。
        elif state["review_result"] is not None:
            skip_reason = "修订后的草稿必须再次复核"

        # 步骤 5：中高风险问题始终交给 LLM Reviewer；只有 low 才有资格 Fast Path。
        elif not analysis or analysis.risk_level != "low":
            skip_reason = "中高风险问题必须进行模型复核"

        # 步骤 6：Fast Path 当前只服务正常 no_match。matched 需要审查证据一致性，
        # tool_error/unavailable 需要审查降级表述，因此都不能跳过。
        elif not packet or packet.retrieval_status != "no_match":
            skip_reason = "存在法规依据或检索异常，必须进行模型复核"

        # 步骤 7：无法条回答必须是低置信度，否则说明 Counsel 越过证据边界。
        elif not draft or draft.confidence != "low":
            skip_reason = "无法条回答的置信度边界未满足"

        # 步骤 8：确定性检查披露和法条幻觉；失败时交给 Reviewer 指导改稿。
        elif _no_match_violations(draft.answer):
            skip_reason = "无法条回答未通过确定性边界校验"

        # 步骤 9：no_match 的 claims 不得携带任何证据 ID。
        elif any(claim.evidence_chunk_ids or claim.evidence_document_ids for claim in draft.claims):
            skip_reason = "无法条回答包含证据引用"

        # 步骤 10A：存在任一不能跳过的原因时，不写 ReviewResult；after_review_gate
        # 看到空结果会进入 reviewer 节点。review_mode 同步到评测和 Checkpoint 指标。
        if skip_reason:
            runtime.context.metrics.review_mode = "llm"
            audit(
                "agent.review.selected",
                status="selected",
                review_mode="llm",
                review_skip_reason=skip_reason,
                **runtime.context.audit_fields,
            )
            return {}

        # 步骤 10B：全部确定性门禁通过时，生成等价的已批准 ReviewResult，直接去
        # Finalize，节省一次 LLM 调用但保留相同最终安全出口。
        audit(
            "agent.review.skipped",
            status="success",
            review_mode="deterministic",
            review_skip_reason="低风险无法条回答已通过确定性边界校验",
            **runtime.context.audit_fields,
        )
        runtime.context.metrics.review_mode = "deterministic"
        return {
            "review_result": ReviewResult(approved=True, next_action="finalize")
        }