"""Case Analyst 节点：案情分类、事实整理与研究规划。"""

import asyncio
from typing import Any

from langgraph.runtime import Runtime
from pydantic import ValidationError

from backend.app.agent.schemas import AgentError, CaseAnalysis, ResearchTask
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState

from ..evidence import _message_text, _payload, _validate_fact_overrides
from ..prompts import ANALYST_PROMPT


class CaseAnalystNode:
    async def case_analyst(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        """分类请求、整理事实并提出研究任务。

        输入：本轮 messages、memory_context 以及尚为空的中间状态。
        输出：case_analysis 和有效事实覆盖；闲聊或需要澄清时还会直接写
        final_answer。返回字典由 LangGraph 合并回 State。
        """

        # 步骤 1：先发布节点状态。stream_writer 产生 LangGraph custom event，随后
        # AgentRuntime 会补充 request/user/conversation 快照并交给 AgentRunEvent；
        # 它不会把 Prompt、内部推理或完整 Graph State 暴露给订阅者。
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "analyzing", "message": "正在分析案情"}})

        # 步骤 2A（仅离线评测）：evaluation_case_analysis 是测试提供的固定分析结果，
        # 用于把"路由/后续节点测试"与真实 Analyst 模型波动分离。生产请求默认为
        # None；即使走固定输入，仍必须执行事实所有权校验。
        if runtime.context.evaluation_case_analysis is not None:
            # 步骤 2A-1：先用 CaseAnalysis 校验测试对象，保证字段与真实模型输出一致。
            analysis = CaseAnalysis.model_validate(runtime.context.evaluation_case_analysis)
            # 步骤 2A-2：模型/Fixture 提供的 replaced_memory_id 不能直接信任。
            # to_thread 把同步 SQLAlchemy 查询放到工作线程，避免阻塞事件循环。
            validated_overrides = await asyncio.to_thread(
                _validate_fact_overrides, analysis.current_fact_overrides, runtime.context
            )
            # 步骤 2A-3：只保留通过 tenant/user/conversation 所有权校验的修正事实。
            analysis = analysis.model_copy(update={"current_fact_overrides": validated_overrides})
            # 步骤 2A-4：构造节点增量。LangGraph 会把这些键合并进现有 State；
            # model_call_count 同步写回是为了 Checkpoint 恢复时不重置调用额度。
            update: dict[str, Any] = {
                "case_analysis": analysis,
                "current_fact_overrides": [
                    item.model_dump() for item in analysis.current_fact_overrides
                ],
                "model_call_count": runtime.context.metrics.model_call_count,
            }
            # 步骤 2A-5：无需研究的两种请求直接准备最终正文。after_analysis 会把
            # Graph 路由到 finalize，而不是进入 MCP Research。
            if analysis.next_action == "direct_answer":
                update["final_answer"] = analysis.direct_answer
            elif analysis.next_action == "ask_clarification":
                questions = analysis.clarification_questions or analysis.missing_facts
                update["final_answer"] = "为了更准确地分析，请补充以下信息：\n\n" + "\n".join(
                    f"- {item}" for item in questions
                )
            return update

        # 步骤 2B（生产路径）：调用 Analyst 模型并得到结构化 CaseAnalysis。
        try:
            # 步骤 2B-1：_invoke_json 使用 ChatOpenAI.ainvoke，返回完整 AIMessage，
            # 再经 JSON 提取和 CaseAnalysis Pydantic 校验；该调用不绑定任何工具。
            analysis = await self._invoke_json(
                runtime, "case_analyst", ANALYST_PROMPT, _payload(state), CaseAnalysis
            )
        except (ValueError, ValidationError, RuntimeError) as exc:
            # 步骤 2B-失败：模型输出无法解析、Schema 不合法或调用额度耗尽时，采用
            # 保守研究路线，避免把法律问题误当闲聊直接回答。错误写入 State 供审计，
            # 但不会把异常堆栈或内部细节直接返回用户。
            question = _message_text(state["messages"][-1]) if state["messages"] else ""
            analysis = CaseAnalysis(request_type="legal_consultation", case_summary=question, legal_issues=[question], research_tasks=[ResearchTask(issue_id="issue-1", query=question, purpose="核验法律依据")], next_action="research")
            return {
                "case_analysis": analysis,
                "errors": [*state["errors"], AgentError(agent="case_analyst", message=str(exc))],
                "model_call_count": runtime.context.metrics.model_call_count,
            }

        # 步骤 3：对生产模型给出的 Fact Override 执行与评测路径相同的所有权校验。
        validated_overrides = await asyncio.to_thread(
            _validate_fact_overrides, analysis.current_fact_overrides, runtime.context
        )
        analysis = analysis.model_copy(update={"current_fact_overrides": validated_overrides})

        # 步骤 4：构造统一的 State 增量，供下一节点及 Checkpoint 使用。
        update: dict[str, Any] = {
            "case_analysis": analysis,
            "current_fact_overrides": [
                item.model_dump() for item in analysis.current_fact_overrides
            ],
            "model_call_count": runtime.context.metrics.model_call_count,
        }

        # 步骤 5：闲聊/澄清提前写 final_answer；法律咨询不写正文，等待 Research。
        if analysis.next_action == "direct_answer":
            update["final_answer"] = analysis.direct_answer or "您好，请告诉我需要咨询的法律问题。"
        elif analysis.next_action == "ask_clarification":
            questions = analysis.clarification_questions or analysis.missing_facts
            update["final_answer"] = "为了更准确地分析，请补充以下信息：\n\n" + "\n".join(f"- {item}" for item in questions)

        # 步骤 6：返回的 dict 不是 HTTP 响应；LangGraph 会先合并 State，然后调用
        # after_analysis 选择 legal_researcher 或 finalize。
        return update
