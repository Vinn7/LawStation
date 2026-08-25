from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage

from backend.app.agent.runtime import AgentRuntime
from backend.app.agent.state import AgentInvocationContext, AgentInvocationIdentity
from backend.app.core.context import RequestUserContext


class AgentService:
    """Request-scoped facade over the shared LangChain Agent runtime."""

    def __init__(
        self,
        runtime: AgentRuntime,
        ctx: RequestUserContext,
        conversation_id: str,
        trace_config: dict | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.ctx = ctx
        self.conversation_id = conversation_id
        self.tool_call_count = 0
        self.model_call_count = 0
        self.final_answer = ""
        self.langsmith_trace_id: str | None = trace_id
        self.trace_config = trace_config

    async def run(self, memory_context: str, history, question: str) -> AsyncIterator[dict]:
        messages = []
        for message in history:
            if message.role == "user":
                messages.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                messages.append(AIMessage(content=message.content))
        messages.append(HumanMessage(content=question))
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
            stream = (
                self.runtime.stream(
                    invocation,
                    messages,
                    memory_context,
                    trace_config=self.trace_config,
                )
                if self.trace_config is not None
                else self.runtime.stream(invocation, messages, memory_context)
            )
            async for item in stream:
                if item["event"] == "agent_final":
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
        self.tool_call_count = invocation.metrics.tool_call_count
        self.model_call_count = invocation.metrics.model_call_count
        self.langsmith_trace_id = invocation.langsmith_trace_id
