"""请求级 Agent 门面：把业务消息和身份转换为 AgentRuntime 调用。"""

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage

from backend.app.agent.runtime import AgentRuntime
from backend.app.agent.state import AgentInvocationContext, AgentInvocationIdentity
from backend.app.core.context import RequestUserContext


class AgentService:
    """共享 Runtime 之上的请求级门面；它本身不是第四个 Agent。"""

    def __init__(
        self,
        runtime: AgentRuntime,
        ctx: RequestUserContext,
        conversation_id: str,
        trace_config: dict | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
        resume: bool = False,
    ) -> None:
        self.runtime = runtime
        self.ctx = ctx
        self.conversation_id = conversation_id
        self.tool_call_count = 0
        self.model_call_count = 0
        self.final_answer = ""
        self.langsmith_trace_id: str | None = trace_id
        self.active_skills: list[dict[str, str]] = []
        self.trace_config = trace_config
        self.run_id = run_id
        self.resume = resume

    async def run(self, memory_context: str, history, question: str) -> AsyncIterator[dict]:
        """组装本轮上下文、执行 Graph，并收集最终计数与安全输出。"""

        messages = []
        # SQLAlchemy Message 是业务持久化模型；Graph/ChatOpenAI 只理解 LangChain
        # Message，因此在请求边界转换角色，同时不把 ORM Session 带入异步调用。
        for message in history:
            if message.role == "user":
                messages.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                messages.append(AIMessage(content=message.content))
        # 当前问题最后追加，确保它在与摘要/历史冲突时拥有最高时序优先级。
        messages.append(HumanMessage(content=question))
        # Invocation Context 不写入 Graph Checkpoint，保存本轮不可变所有权身份和
        # 可变调用计数，供节点、中间件和审计共同使用。
        invocation = AgentInvocationContext(
            identity=AgentInvocationIdentity(
                request_id=self.ctx.request_id,
                tenant_id=self.ctx.tenant_id,
                user_id=self.ctx.user_id,
                conversation_id=self.conversation_id,
            ),
            trace_config=self.trace_config,
        )
        try:
            invocation.langsmith_trace_id = self.langsmith_trace_id
            runtime_kwargs = {}
            if self.trace_config is not None:
                runtime_kwargs["trace_config"] = self.trace_config
            if self.run_id:
                # thread_id 按 Run 而非 Conversation 生成：Checkpoint 只负责一次长
                # 任务恢复，不会形成与 MemoryService 竞争的第二套跨轮记忆。
                runtime_kwargs["thread_id"] = f"agent-run:{self.run_id}"
                runtime_kwargs["resume"] = self.resume
            stream = self.runtime.stream(
                invocation, messages, memory_context, **runtime_kwargs
            )
            async for item in stream:
                if item["event"] == "agent_final":
                    # agent_final 是服务端内部终值，由 AgentRunManager 幂等持久化；
                    # 其余安全事件才继续交给任务事件日志/SSE。
                    self.final_answer = item["data"]
                else:
                    yield item
        except Exception as exc:
            if (
                invocation.langsmith_trace_id is None
                and hasattr(self.runtime, "observability")
            ):
                invocation.langsmith_trace_id = await self.runtime.observability.force_outcome_trace(
                    request_id=self.ctx.request_id,
                    tenant_id=self.ctx.tenant_id,
                    user_id=self.ctx.user_id,
                    conversation_id=self.conversation_id,
                    question=question,
                    outcome={"status": "failed"},
                    error=type(exc).__name__,
                )
            self.langsmith_trace_id = invocation.langsmith_trace_id
            raise
        # Runtime 完成后再回填计数、Trace 和 Skill 元数据，避免把请求级可变对象
        # 存到共享 AgentRuntime 实例上。
        self.tool_call_count = invocation.metrics.tool_call_count
        self.model_call_count = invocation.metrics.model_call_count
        self.langsmith_trace_id = invocation.langsmith_trace_id
        self.active_skills = list(
            invocation.evaluation_output.get("active_skills", [])
        )
