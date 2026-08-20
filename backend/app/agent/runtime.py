import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import BaseMessage

from backend.app.agent.graph import LegalConsultationGraph
from backend.app.agent.provider import LLMProvider
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.config import Settings, get_settings


class AgentRuntime:
    def __init__(
        self,
        registry: MCPToolRegistry,
        provider: LLMProvider,
        settings: Settings | None = None,
    ) -> None:
        self.registry = registry
        self.provider = provider
        self.settings = settings or get_settings()
        self._compile_lock = asyncio.Lock()
        self._graph: LegalConsultationGraph | None = None
        self._graph_key: tuple[int, tuple[str, ...]] | None = None

    async def ensure_ready(self, context: AgentInvocationContext) -> LegalConsultationGraph:
        model = self.provider.get_chat_model()
        tools = await self.registry.get_tools(context.audit_fields)
        registry_status = self.registry.status()
        key = (registry_status.version, tuple(tool.name for tool in tools))
        if self._graph is not None and self._graph_key == key:
            return self._graph
        async with self._compile_lock:
            if self._graph is None or self._graph_key != key:
                self._graph = LegalConsultationGraph(
                    model=model,
                    tools=tools,
                    registry=self.registry,
                    settings=self.settings,
                )
                self._graph_key = key
        return self._graph

    async def stream(
        self,
        context: AgentInvocationContext,
        messages: list[BaseMessage],
        memory_context: str,
    ) -> AsyncIterator[dict[str, Any]]:
        graph = await self.ensure_ready(context)
        state: LegalConsultationState = {
            "messages": list(messages),
            "memory_context": memory_context,
            "case_analysis": None,
            "evidence_packet": None,
            "counsel_draft": None,
            "review_result": None,
            "retry_count": 0,
            "revision_count": 0,
            "final_answer": "",
            "citations": [],
            "errors": [],
        }
        final_state: dict[str, Any] = dict(state)
        async for part in graph.compiled.astream(
            state,
            context=context,
            stream_mode=["updates", "custom"],
            version="v2",
        ):
            part_type = part.get("type")
            data = part.get("data")
            if part_type == "custom" and isinstance(data, dict) and "event" in data:
                event_data = data.get("data")
                if isinstance(event_data, dict):
                    event_data = {
                        **event_data,
                        "request_id": context.identity.request_id,
                        "user_id": context.identity.user_id,
                        "conversation_id": context.identity.conversation_id,
                    }
                yield {"event": data["event"], "data": event_data}
            elif part_type == "updates" and isinstance(data, dict):
                for update in data.values():
                    if isinstance(update, dict):
                        final_state.update(update)
        citations = final_state.get("citations") or []
        if citations:
            yield {
                "event": "citations",
                "data": [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in citations
                ],
            }
        final_answer = str(final_state.get("final_answer") or "")
        # Draft tokens are withheld until review completes. Stream the approved
        # answer in small chunks without exposing internal drafts or reasoning.
        for start in range(0, len(final_answer), 24):
            yield {"event": "token", "data": final_answer[start : start + 24]}
            await asyncio.sleep(0)
        yield {"event": "agent_final", "data": final_answer}

    async def close(self) -> None:
        await self.registry.close()
