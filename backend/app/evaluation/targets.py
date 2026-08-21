from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from backend.app.agent.provider import LLMProvider
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.runtime import AgentRuntime
from backend.app.agent.state import AgentInvocationContext, AgentInvocationIdentity
from backend.app.core.config import Settings, get_settings
from backend.app.observability import LangSmithObservability


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


async def _invoke(inputs: dict[str, Any], registry: Any, settings: Settings) -> dict[str, Any]:
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
    return await _invoke(inputs, registry, get_settings())


async def live_target(inputs: dict[str, Any]) -> dict[str, Any]:
    return await _invoke(inputs, MCPToolRegistry(), get_settings())
