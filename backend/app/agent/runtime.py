import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import BaseMessage

from backend.app.agent.graph import LegalConsultationGraph
from backend.app.agent.provider import LLMProvider
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.config import Settings, get_settings
from backend.app.observability import LangSmithObservability


class AgentRuntime:
    def __init__(
        self,
        registry: MCPToolRegistry,
        provider: LLMProvider,
        settings: Settings | None = None,
        observability: LangSmithObservability | None = None,
        checkpointer: Any = None,
    ) -> None:
        self.registry = registry
        self.provider = provider
        self.settings = settings or get_settings()
        self.observability = observability or LangSmithObservability(self.settings)
        self.checkpointer = checkpointer
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
                    checkpointer=self.checkpointer,
                )
                self._graph_key = key
        return self._graph

    async def stream(
        self,
        context: AgentInvocationContext,
        messages: list[BaseMessage],
        memory_context: str,
        trace_config: dict[str, Any] | None = None,
        thread_id: str | None = None,
        resume: bool = False,
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
            "current_fact_overrides": [],
            "model_call_count": 0,
            "tool_call_count": 0,
            "tool_trajectory": [],
        }
        final_state: dict[str, Any] = dict(state)
        config = dict(trace_config or context.trace_config or {})
        configurable = dict(config.get("configurable") or {})
        configurable.update({
            "thread_id": thread_id or configurable.get("thread_id") or context.identity.request_id,
            "checkpoint_ns": "lawstation-consultation-v1",
        })
        config["configurable"] = configurable
        graph_input: LegalConsultationState | None = state
        if resume and self.checkpointer is not None:
            snapshot = await graph.compiled.aget_state(config)
            if snapshot and snapshot.values:
                final_state.update(snapshot.values)
                context.metrics.model_call_count = int(snapshot.values.get("model_call_count", 0))
                context.metrics.tool_call_count = int(snapshot.values.get("tool_call_count", 0))
                context.metrics.tool_trajectory = list(
                    snapshot.values.get("tool_trajectory", [])
                )
                graph_input = None
        async for part in graph.compiled.astream(
            graph_input,
            context=context,
            config=config,
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
        context.evaluation_output = {
            "final_answer": str(final_state.get("final_answer") or ""),
            "case_analysis": (
                final_state["case_analysis"].model_dump()
                if final_state.get("case_analysis") else None
            ),
            "evidence_packet": (
                final_state["evidence_packet"].model_dump()
                if final_state.get("evidence_packet") else None
            ),
            "review_result": (
                final_state["review_result"].model_dump()
                if final_state.get("review_result") else None
            ),
            "citations": [
                item.model_dump() if hasattr(item, "model_dump") else item for item in citations
            ],
            "tool_call_count": context.metrics.tool_call_count,
            "model_call_count": context.metrics.model_call_count,
            "review_mode": context.metrics.review_mode,
            "tool_trajectory": list(context.metrics.tool_trajectory),
            "retry_count": int(final_state.get("retry_count") or 0),
            "revision_count": int(final_state.get("revision_count") or 0),
            "errors": [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in final_state.get("errors", [])
            ],
        }
        analysis = context.evaluation_output.get("case_analysis") or {}
        evidence = context.evaluation_output.get("evidence_packet") or {}
        important_outcome = (
            analysis.get("risk_level") == "high"
            or evidence.get("retrieval_status") in {"tool_error", "tool_unavailable"}
        )
        if not context.langsmith_trace_id and important_outcome:
            question = ""
            if messages:
                content = messages[-1].content
                question = content if isinstance(content, str) else str(content)
            context.langsmith_trace_id = await self.observability.force_outcome_trace(
                request_id=context.identity.request_id,
                tenant_id=context.identity.tenant_id,
                user_id=context.identity.user_id,
                conversation_id=context.identity.conversation_id,
                question=question,
                outcome=context.evaluation_output,
            )
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

    async def checkpoint_info(self, thread_id: str) -> dict[str, Any]:
        if self.checkpointer is None:
            return {}
        graph = self._graph
        if graph is None:
            return {}
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "lawstation-consultation-v1",
            }
        }
        snapshot = await graph.compiled.aget_state(config)
        configurable = (snapshot.config or {}).get("configurable", {}) if snapshot else {}
        return {
            "checkpoint_id": str(configurable.get("checkpoint_id") or ""),
            "next": list(snapshot.next) if snapshot else [],
        }

    async def delete_checkpoint_thread(self, thread_id: str) -> None:
        if self.checkpointer is not None:
            await self.checkpointer.adelete_thread(thread_id)
