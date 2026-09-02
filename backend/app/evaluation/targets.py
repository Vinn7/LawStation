import asyncio
import time
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.runtime import Runtime

from backend.app.agent.graph import LegalConsultationGraph
from backend.app.agent.provider import LLMProvider
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.runtime import AgentRuntime
from backend.app.agent.schemas import CaseAnalysis, CounselDraft, EvidencePacket
from backend.app.agent.state import AgentInvocationContext, AgentInvocationIdentity
from backend.app.core.config import Settings, get_settings
from backend.app.observability import LangSmithObservability
from mcp_servers.law_rag.engine import LawSearchEngine


class FixtureToolRegistry:
    def __init__(self, documents: list[dict[str, Any]], fail: bool = False) -> None:
        self.documents = documents
        self.fail = fail
        self._tools = [
            StructuredTool.from_function(
                coroutine=self.search_laws,
                name="search_laws",
                description="检索法律法规。",
            ),
            StructuredTool.from_function(
                coroutine=self.get_law_article,
                name="get_law_article",
                description="按法律名称和条号读取法条。",
            ),
        ]

    async def search_laws(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("fixture tool error")
        return {"documents": self.documents[:top_k], "query": query}

    async def get_law_article(self, law_name: str, article_number: str) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("fixture tool error")
        matches = [
            item for item in self.documents
            if item.get("law_name") == law_name and item.get("article_number") == article_number
        ]
        return {"documents": matches}

    async def get_tools(self, audit_context=None):
        return self._tools

    def status(self):
        return SimpleNamespace(version=1)

    def invalidate(self, reason: str, audit_context=None) -> None:
        return None

    async def close(self) -> None:
        return None


def _history(inputs: dict[str, Any]):
    messages = []
    for item in inputs.get("history", []):
        content = str(item.get("content", ""))
        messages.append(AIMessage(content=content) if item.get("role") == "assistant" else HumanMessage(content=content))
    messages.append(HumanMessage(content=str(inputs.get("question", ""))))
    return messages


async def _invoke(
    inputs: dict[str, Any],
    registry: Any,
    settings: Settings,
    review_mode: str = "auto",
    use_fixture_analysis: bool = False,
) -> dict[str, Any]:
    settings = settings.model_copy(update={"agent_review_mode": review_mode})
    disabled = settings.model_copy(update={"langsmith_enabled": False})
    runtime = AgentRuntime(
        registry,
        LLMProvider(settings),
        settings,
        observability=LangSmithObservability(disabled),
    )
    context = AgentInvocationContext(
        identity=AgentInvocationIdentity(
            request_id=str(uuid4()),
            tenant_id="eval-tenant",
            user_id="eval-user",
            conversation_id=str(uuid4()),
        ),
        persist_tool_audit=False,
        evaluation_case_analysis=(
            inputs.get("fixture_case_analysis") if use_fixture_analysis else None
        ),
    )
    try:
        async for _ in runtime.stream(
            context,
            _history(inputs),
            str(inputs.get("memory_context", "")),
        ):
            pass
        return context.evaluation_output
    finally:
        await runtime.close()


async def component_target(inputs: dict[str, Any]) -> dict[str, Any]:
    registry = FixtureToolRegistry(
        list(inputs.get("fixture_documents", [])),
        fail=bool(inputs.get("fixture_tool_error")),
    )
    return await _invoke(inputs, registry, get_settings(), use_fixture_analysis=True)


async def live_target(inputs: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    return await _invoke(inputs, MCPToolRegistry(settings), settings)


def agent_target(mode: str, review_mode: str, settings: Settings | None = None):
    """Build an isolated evaluation target without changing production defaults."""
    base_settings = settings or get_settings()

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        if mode == "component":
            registry: Any = FixtureToolRegistry(
                list(inputs.get("fixture_documents", [])),
                fail=bool(inputs.get("fixture_tool_error")),
            )
        elif mode == "live":
            registry = MCPToolRegistry(base_settings)
        else:
            raise ValueError(f"不支持的 Agent 评测模式：{mode}")
        return await _invoke(
            inputs,
            registry,
            base_settings,
            review_mode,
            use_fixture_analysis=(mode == "component"),
        )

    return target


def reviewer_target(settings: Settings | None = None):
    """Evaluate the production review/gate/finalize path from a frozen draft."""

    base_settings = (settings or get_settings()).model_copy(
        update={"agent_review_mode": "auto"}
    )

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        registry = FixtureToolRegistry([])
        graph = LegalConsultationGraph(
            LLMProvider(base_settings).get_chat_model(), [], registry, base_settings
        )
        context = AgentInvocationContext(
            identity=AgentInvocationIdentity(
                request_id=str(uuid4()),
                tenant_id="eval-tenant",
                user_id="eval-user",
                conversation_id=str(uuid4()),
            ),
            persist_tool_audit=False,
        )
        runtime = Runtime(context=context, stream_writer=lambda _event: None)
        analysis = CaseAnalysis.model_validate(inputs["fixture_case_analysis"])
        packet = EvidencePacket.model_validate(inputs["fixture_evidence_packet"])
        original_draft = CounselDraft.model_validate(inputs["fixture_counsel_draft"])
        state: dict[str, Any] = {
            "messages": _history(inputs),
            "memory_context": str(inputs.get("memory_context", "")),
            "case_analysis": analysis,
            "evidence_packet": packet,
            "counsel_draft": original_draft,
            "review_result": None,
            "retry_count": 0,
            "revision_count": 0,
            "final_answer": "",
            "citations": [],
            "errors": [],
            "current_fact_overrides": list(inputs.get("current_fact_overrides", [])),
            "model_call_count": 0,
            "tool_call_count": 0,
            "tool_trajectory": [],
        }
        try:
            state.update(await graph.review_gate(state, runtime))
            if graph.after_review_gate(state) == "review":
                state.update(await graph.reviewer(state, runtime))
            initial_review = state.get("review_result")
            revision_performed = graph.after_review(state) == "revise"
            revised_draft = None
            if revision_performed:
                state.update(await graph.legal_counsel(state, runtime))
                revised_draft = state.get("counsel_draft")
                state.update(await graph.reviewer(state, runtime))
            state.update(await graph.finalize(state, runtime))
            return {
                "initial_review_result": (
                    initial_review.model_dump() if initial_review else None
                ),
                "final_review_result": (
                    state["review_result"].model_dump()
                    if state.get("review_result") else None
                ),
                "original_draft": original_draft.model_dump(),
                "revised_draft": revised_draft.model_dump() if revised_draft else None,
                "revision_performed": revision_performed,
                "final_answer": state["final_answer"],
                "citations": [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in state["citations"]
                ],
                "evidence_packet": packet.model_dump(),
                "review_mode": context.metrics.review_mode,
                "model_call_count": context.metrics.model_call_count,
                "tool_call_count": context.metrics.tool_call_count,
                "errors": [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in state["errors"]
                ],
            }
        finally:
            await registry.close()

    return target


def counsel_quality_target(settings: Settings | None = None):
    """Start at Counsel with a frozen analysis/evidence packet, then review/finalize."""

    base_settings = settings or get_settings()

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        registry = FixtureToolRegistry([])
        graph = LegalConsultationGraph(
            LLMProvider(base_settings).get_chat_model(), [], registry, base_settings
        )
        context = AgentInvocationContext(
            identity=AgentInvocationIdentity(
                request_id=str(uuid4()), tenant_id="eval-tenant", user_id="eval-user",
                conversation_id=str(uuid4()),
            ),
            persist_tool_audit=False,
        )
        runtime = Runtime(context=context, stream_writer=lambda _event: None)
        analysis = CaseAnalysis.model_validate(inputs["fixture_case_analysis"])
        packet = EvidencePacket.model_validate(inputs["fixture_evidence_packet"])
        state: dict[str, Any] = {
            "messages": _history(inputs),
            "memory_context": str(inputs.get("memory_context", "")),
            "case_analysis": analysis,
            "evidence_packet": packet,
            "counsel_draft": None,
            "review_result": None,
            "retry_count": 0,
            "revision_count": 0,
            "final_answer": "",
            "citations": [],
            "errors": [],
            "current_fact_overrides": list(inputs.get("current_fact_overrides", [])),
            "model_call_count": 0,
            "tool_call_count": 0,
            "tool_trajectory": [],
        }
        try:
            state.update(await graph.legal_counsel(state, runtime))
            state.update(await graph.review_gate(state, runtime))
            if graph.after_review_gate(state) == "review":
                state.update(await graph.reviewer(state, runtime))
            if graph.after_review(state) == "revise":
                state.update(await graph.legal_counsel(state, runtime))
                state.update(await graph.reviewer(state, runtime))
            state.update(await graph.finalize(state, runtime))
            draft = state.get("counsel_draft")
            return {
                "final_answer": state["final_answer"],
                "case_analysis": analysis.model_dump(),
                "evidence_packet": packet.model_dump(),
                "review_result": (
                    state["review_result"].model_dump()
                    if state.get("review_result") else None
                ),
                "counsel_claims": (
                    [item.model_dump() for item in draft.claims] if draft else []
                ),
                "counsel_confidence": draft.confidence if draft else None,
                "citations": [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in state["citations"]
                ],
                "review_mode": context.metrics.review_mode,
                "model_call_count": context.metrics.model_call_count,
                "tool_call_count": context.metrics.tool_call_count,
                "errors": [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in state["errors"]
                ],
            }
        finally:
            await registry.close()

    return target


class RetrievalTarget:
    """Application-scoped direct retrieval target used for RAG ablations."""

    def __init__(self, engine: LawSearchEngine, retrieval_mode: str) -> None:
        self.engine = engine
        self.retrieval_mode = retrieval_mode

    @classmethod
    async def create(
        cls, settings: Settings, retrieval_mode: str
    ) -> "RetrievalTarget":
        engine = await asyncio.to_thread(LawSearchEngine, settings)
        await engine.initialize_index(wait=True)
        if retrieval_mode == "hybrid" and not engine.status().get("dense_enabled"):
            await engine.close()
            raise RuntimeError("Hybrid 评测要求有效 Dense 索引，当前检索引擎处于降级状态")
        if settings.rag_rerank_enabled and engine.status().get("reranker_status") != "ready":
            await engine.close()
            raise RuntimeError("Reranker 评测要求精排模型 ready，禁止静默降级为 RRF")
        return cls(engine, retrieval_mode)

    async def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        results = await self.engine.search(
            str(inputs.get("question", "")),
            top_k=int(inputs.get("top_k", 5)),
            filters=inputs.get("filters"),
            retrieval_mode=self.retrieval_mode,
        )
        engine_status = self.engine.status()
        reranker_model = str(engine_status.get("reranker_model") or "")
        reranker_model_digest = str(
            engine_status.get("reranker_model_digest") or ""
        )
        result_ranking_version = str(
            results[0].get("ranking_version") if results else ""
        )
        ranking_version = result_ranking_version or (
            f"{reranker_model}@{reranker_model_digest}"
            if reranker_model and reranker_model_digest
            else "rrf-v1"
        )
        return {
            "retrieval_status": "matched" if results else "no_match",
            "retrieval_results": results,
            "retrieval_mode": self.retrieval_mode,
            "retrieval_duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "dense_enabled": bool(engine_status.get("dense_enabled")),
            "rerank_applied": bool(results and results[0].get("rerank_applied")),
            "rerank_duration_ms": (
                float(results[0].get("rerank_duration_ms", 0)) if results else 0.0
            ),
            "rerank_prompt_tokens": (
                int(results[0].get("rerank_prompt_tokens", 0)) if results else 0
            ),
            "rerank_candidate_count": (
                int(results[0].get("rerank_candidate_count", 0)) if results else 0
            ),
            "reranker_status": engine_status.get("reranker_status"),
            "reranker_provider": engine_status.get("reranker_provider"),
            "reranker_model": reranker_model,
            "reranker_model_digest": reranker_model_digest,
            "ranking_version": ranking_version,
        }

    async def close(self) -> None:
        await self.engine.close()
