from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage

from backend.app.agent.runtime import AgentInvocationContext, AgentRuntime
from backend.app.core.context import RequestUserContext


class AgentService:
    """Request-scoped facade over the shared LangChain Agent runtime."""

    def __init__(
        self,
        runtime: AgentRuntime,
        ctx: RequestUserContext,
        conversation_id: str,
    ) -> None:
        self.runtime = runtime
        self.ctx = ctx
        self.conversation_id = conversation_id
        self.tool_call_count = 0
        self.final_answer = ""

    async def run(self, memory_context: str, history, question: str) -> AsyncIterator[dict]:
        messages = []
        for message in history:
            if message.role == "user":
                messages.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                messages.append(AIMessage(content=message.content))
        messages.append(HumanMessage(content=question))
        invocation = AgentInvocationContext(
            user=self.ctx,
            conversation_id=self.conversation_id,
        )
        async for item in self.runtime.stream(invocation, messages, memory_context):
            if item["event"] == "agent_final":
                self.final_answer = item["data"]
            else:
                yield item
        self.tool_call_count = invocation.tool_call_count
