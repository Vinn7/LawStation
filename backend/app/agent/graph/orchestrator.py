"""LawStation 法律咨询的 LangGraph 编排入口：图拓扑组装与跨节点共享调用。

本模块只负责“图怎么连”和节点间共享的基础设施（无工具结构化调用），六个节点
的业务逻辑分别在 nodes/ 目录下按阶段拆分，通过多继承组合进本文件的
LegalConsultationGraph。

``legal_researcher``（nodes/research.py）节点内部使用 LangChain
``create_agent``。该 Agent 才会让 DeepSeek 自主产生 tool_calls、经 MCP 执行
工具、接收 ToolMessage，并继续调用模型形成研究结论；模型自主决定检索次数和
参数，不由代码替它决定。``response_format=ToolStrategy(EvidencePacket)`` 强制
它在结束检索后必须通过结构化工具（而不是自由文本）汇报结果：LangChain 在这种
模式下对每一轮模型调用都设置 ``tool_choice="required"``，模型在这个子 Agent
里物理上不能返回纯文本。

主路径为 ``START -> case_analyst -> [finalize | legal_researcher] ->
legal_counsel -> review_gate -> [finalize | reviewer]``。Reviewer 最多把状态送回
Research 或 Counsel 各一次，最后由 ``finalize`` 执行确定性证据边界校验。

``LegalConsultationState`` 是节点间传递且可被 LangGraph Checkpoint 持久化的数据；
``AgentInvocationContext`` 则保存当前请求的身份、调用计数和审计对象，不属于跨节点
业务状态，也不得成为跨轮会话记忆。
"""

import json
import logging
import time
from typing import Any, TypeVar

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel

from backend.app.agent.middleware import InvocationModelLimitMiddleware, ToolAuditMiddleware
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.schemas import EvidencePacket
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.config import Settings
from backend.app.core.logging import audit, summary

from .evidence import _extract_json, _message_text
from .nodes.case_analyst import CaseAnalystNode
from .nodes.counsel import CounselNode
from .nodes.finalize import FinalizeNode
from .nodes.research import ResearchNode
from .nodes.review import ReviewNode

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LegalConsultationGraph(
    CaseAnalystNode, ResearchNode, CounselNode, ReviewNode, FinalizeNode,
):
    """编译并持有可复用的三 Agent LangGraph。

    类实例可以由多个请求共享；每次 ``compiled.astream`` 接收的 State 与
    ``AgentInvocationContext`` 仍相互隔离。只有 Research 节点需要 LangChain
    create_agent，因为只有它允许模型自主选择 MCP 工具。
    """

    def __init__(
        self,
        model: Any,
        tools: list[BaseTool],
        registry: MCPToolRegistry,
        settings: Settings,
        checkpointer: Any = None,
    ) -> None:
        # response_format=json_object 是 DeepSeek 原生 JSON Output 模式：从接口层
        # 强制模型只能返回一个 JSON 对象，取代“靠 Prompt 文字拜托模型别唠叨”的
        # 弱约束（memory_tasks.py 的记忆抽取模型已验证过这个模式）。这里只用于
        # _invoke_json 的无工具调用（case_analyst/legal_counsel/reviewer/
        # evidence_selector）；create_agent 内部另外用原始未绑定的 model，因为
        # 它需要自己 bind_tools + tool_choice，不能与固定 response_format 叠加。
        self.model = model.bind(response_format={"type": "json_object"})
        self.tools = tools
        # 真实 MCP 工具名集合，供 legal_researcher 从 update 消息里区分“真实检索
        # 结果”和 ToolStrategy 自动生成的结构化输出 ToolMessage。
        self.tool_names = {tool.name for tool in tools}
        # LangChain create_agent 返回一个可 astream 的模型—工具循环：tools 是 MCP
        # Adapter 包装的 BaseTool，context_schema 让中间件取得本轮身份和计数。
        # response_format=ToolStrategy(EvidencePacket) 让 LangChain 对每一轮模型
        # 调用都设置 tool_choice="required"：模型必须调用某个工具（真实检索工具，
        # 或代表“完成”的结构化输出工具），不能返回自由文本，从接口层杜绝了模型在
        # 撞到工具调用上限后转而输出大段自然语言解释的情况。handle_errors=True 让
        # Schema 校验失败时自动生成一条错误 ToolMessage 要求模型重试，而不是直接
        # 抛异常降级。构造阶段不会请求模型；没有工具时保留 None，由节点走
        # tool_unavailable。
        self.research_agent = create_agent(
            model=model,
            tools=tools,
            response_format=ToolStrategy(EvidencePacket, handle_errors=True),
            context_schema=AgentInvocationContext,
            middleware=[
                InvocationModelLimitMiddleware(settings),
                # 工具调用上限只由 ToolAuditMiddleware 强制执行：达到上限时返回一条
                # 正常 ToolMessage（而不是叠加 LangChain 内置 ToolCallLimitMiddleware
                # 在模型/工具循环外层强制打断），让模型在其熟悉的“处理工具结果”
                # 路径里继续，避免两套限流机制给出不一致信号。
                ModelCallLimitMiddleware(run_limit=settings.agent_max_model_calls, exit_behavior="end"),
                ToolAuditMiddleware(registry, settings),
            ],
            name="legal-research-agent",
        ) if tools else None
        self.registry = registry
        self.settings = settings
        self.checkpointer = checkpointer
        self.compiled = self._compile()

    def _compile(self):
        """声明节点拓扑并编译，不在此阶段执行任何模型或工具。"""

        # LegalConsultationState 会在节点返回 dict 时被 LangGraph 合并并写入
        # Checkpoint；context_schema 只提供运行期依赖，不成为可持久化 Graph State。
        graph = StateGraph(LegalConsultationState, context_schema=AgentInvocationContext)
        # Analyst：模型结构化分析，写 case_analysis 和事实覆盖。
        graph.add_node("case_analyst", self.case_analyst)
        # Research：内部 LangChain Agent 可调用 MCP，写 evidence_packet。
        graph.add_node("legal_researcher", self.legal_researcher)
        # Counsel：无工具模型调用，基于证据包写 counsel_draft。
        graph.add_node("legal_counsel", self.legal_counsel)
        # Gate：纯代码快速检查，决定是否需要额外 LLM Reviewer。
        graph.add_node("review_gate", self.review_gate)
        # Reviewer：结构化复核，可能要求一次补检索或一次改稿。
        graph.add_node("reviewer", self.reviewer)
        # Finalize：纯代码安全出口，生成 final_answer 与 citations。
        graph.add_node("finalize", self.finalize)
        graph.add_edge(START, "case_analyst")
        # conditional_edges 先调用 after_analysis，再用返回的字符串查右侧映射。
        graph.add_conditional_edges("case_analyst", self.after_analysis, {"finish": "finalize", "research": "legal_researcher"})
        # 普通 edge 无条件执行后继节点。
        graph.add_edge("legal_researcher", "legal_counsel")
        graph.add_edge("legal_counsel", "review_gate")
        graph.add_conditional_edges(
            "review_gate", self.after_review_gate, {"review": "reviewer", "finish": "finalize"}
        )
        graph.add_conditional_edges(
            "reviewer",
            self.after_review,
            {"finish": "finalize", "research": "legal_researcher", "revise": "legal_counsel"},
        )
        graph.add_edge("finalize", END)
        # compile 生成可复用 CompiledStateGraph。Checkpointer 在 super-step 边界保存
        # State，使重启后能从最近完成节点继续；它不会保存每个 token 或请求级对象。
        return graph.compile(
            name="lawstation-three-agent-graph",
            checkpointer=self.checkpointer,
        )

    async def _invoke_json(self, runtime: Runtime[AgentInvocationContext], agent_name: str, prompt: str, payload: dict[str, Any], schema: type[SchemaT]) -> SchemaT:
        """执行一次不带工具的 LangChain 模型调用并返回 Pydantic 对象。

        顺序为：检查/增加模型额度 -> 组装 System/Human Message -> ChatOpenAI
        ``ainvoke`` -> 提取 AIMessage 文本 -> JSON 解析 -> Schema 校验。``ainvoke``
        返回完整消息而非 SSE token；Provider/解析错误向上抛给节点决定降级策略。
        """
        context = runtime.context
        if context.metrics.model_call_count >= self.settings.agent_max_model_calls:
            raise RuntimeError("本轮模型调用次数已达到上限")
        context.metrics.model_call_count += 1
        started = time.perf_counter()
        audit("agent.node.started", status="started", agent=agent_name, **context.audit_fields)
        try:
            # self.model 已在 __init__ 绑定 response_format=json_object，且这里没有
            # bind_tools，因此模型只能返回一个 JSON 对象，不能在该调用中执行 MCP。
            response = await self.model.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
            ])
            result = schema.model_validate(_extract_json(_message_text(response)))
            audit("agent.node.completed", status="success", agent=agent_name, duration_ms=int((time.perf_counter() - started) * 1000), **context.audit_fields)
            return result
        except Exception as exc:  # Central model boundary audits and re-raises provider errors.
            audit("agent.node.failed", level=logging.ERROR, status="failed", agent=agent_name, error_type=type(exc).__name__, error=summary(str(exc)), duration_ms=int((time.perf_counter() - started) * 1000), **context.audit_fields)
            raise

    def after_analysis(self, state: LegalConsultationState) -> str:
        """LangGraph 条件路由：只有 next_action=research 才进入 MCP 研究节点。"""
        # 步骤 1：读取 Analyst 已写入的结构化结果，不再次调用模型。
        analysis = state["case_analysis"]
        # 步骤 2：返回值必须匹配 _compile() 中 conditional_edges 的映射 key。
        return "research" if analysis and analysis.next_action == "research" else "finish"

    @staticmethod
    def after_review_gate(state: LegalConsultationState) -> str:
        """LangGraph 路由：Gate 已批准则 finalize，否则进入 LLM Reviewer。"""
        # 步骤 1：Gate Fast Path 会写 approved=True；需要 LLM 时返回空增量，State
        # 中 review_result 仍为空（首次草稿）或未批准（修订场景）。
        review = state["review_result"]
        # 步骤 2：返回值对应 _compile() 中 finish->finalize、review->reviewer。
        return "finish" if review and review.approved else "review"

    def after_review(self, state: LegalConsultationState) -> str:
        """把 ReviewResult 映射为结束、补检索或改稿，并强制循环上限。"""

        # 步骤 1：读取 Reviewer 决策和当前证据状态；路由函数不调用外部服务。
        review = state["review_result"]
        packet = state["evidence_packet"]

        # 步骤 2：无复核结果、已经批准或明确 finalize 时结束，避免无意义回流。
        if not review or review.approved or review.next_action == "finalize":
            return "finish"

        # 步骤 3：空结果或工具故障无法通过相同查询可靠补足；即使模型建议
        # research_again，也只允许一次必要草稿修订，不再次消耗 MCP 调用。
        if packet and packet.retrieval_status in {"no_match", "tool_unavailable", "tool_error"}:
            return "revise" if review.next_action == "revise_draft" and state["revision_count"] < 1 else "finish"

        # 步骤 4：matched 模式只有明确 research_again 且尚未补检索过，才返回
        # research；retry_count>=1 后强制结束。
        if review.next_action == "research_again" and state["retry_count"] < 1:
            return "research"

        # 步骤 5：表达/论证问题最多返回 Counsel 修改一次。
        if review.next_action == "revise_draft" and state["revision_count"] < 1:
            return "revise"

        # 步骤 6：未知动作或额度耗尽统一结束，防止 Graph 无限循环。
        return "finish"
