import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from backend.app.agent.provider import AgentConfigurationError, LLMProvider
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.runtime import AgentInvocationContext, AgentRuntime
from backend.app.agent.service import AgentService
from backend.app.core.config import Settings
from backend.app.core.context import RequestUserContext


def settings(**changes):
    values = {
        "deepseek_api_key": "test-key",
        "mcp_tool_timeout_seconds": 1,
        "mcp_tool_discovery_retry_seconds": 30,
    }
    values.update(changes)
    return Settings(_env_file=None, **values)


def fake_tool():
    async def search_laws(query: str) -> str:
        """Search laws for a query."""
        return query

    return StructuredTool.from_function(coroutine=search_laws)


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@pytest.mark.asyncio
async def test_registry_discovers_tools_only_once_for_concurrent_requests():
    client = SimpleNamespace(get_tools=AsyncMock(return_value=[fake_tool()]))
    registry = MCPToolRegistry(settings(), client=client)

    results = await asyncio.gather(*(registry.get_tools() for _ in range(20)))

    assert client.get_tools.await_count == 1
    assert all([tool.name for tool in result] == ["search_laws"] for result in results)
    assert registry.status().status == "ready"
    assert registry.status().version == 1


@pytest.mark.asyncio
async def test_registry_failure_obeys_retry_cooldown():
    client = SimpleNamespace(get_tools=AsyncMock(side_effect=ConnectionError("offline")))
    registry = MCPToolRegistry(settings(), client=client)

    assert await registry.get_tools() == []
    assert await registry.get_tools() == []

    assert client.get_tools.await_count == 1
    assert registry.status().status == "failed"


@pytest.mark.asyncio
async def test_runtime_streams_tokens_and_final_answer_without_tools():
    registry = SimpleNamespace(
        get_tools=AsyncMock(return_value=[]),
        status=lambda: SimpleNamespace(version=0),
        close=AsyncMock(),
    )
    provider = SimpleNamespace(
        get_chat_model=lambda: GenericFakeChatModel(messages=iter(["测试回答"]))
    )
    runtime = AgentRuntime(registry, provider, settings())
    context = AgentInvocationContext(
        RequestUserContext("tenant", "user", "request"), "conversation"
    )

    events = [
        event
        async for event in runtime.stream(
            context,
            [HumanMessage(content="问题")],
            "用户记忆",
        )
    ]

    assert events == [
        {"event": "token", "data": "测试回答"},
        {"event": "agent_final", "data": "测试回答"},
    ]


@pytest.mark.asyncio
async def test_create_agent_executes_cached_mcp_tool_and_streams_safe_events(monkeypatch):
    monkeypatch.setattr("backend.app.agent.middleware._persist_tool_audit", lambda *args: None)
    tool = fake_tool()
    registry = SimpleNamespace(
        get_tools=AsyncMock(return_value=[tool]),
        status=lambda: SimpleNamespace(version=1),
        close=AsyncMock(),
        invalidate=lambda *args: None,
    )
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_laws",
                        "args": {"query": "劳动合同"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="最终回答"),
        ]
    )
    runtime = AgentRuntime(
        registry,
        SimpleNamespace(get_chat_model=lambda: model),
        settings(),
    )
    context = AgentInvocationContext(
        RequestUserContext("tenant", "user", "request"), "conversation"
    )

    events = [
        event
        async for event in runtime.stream(
            context,
            [HumanMessage(content="问题")],
            "",
        )
    ]

    assert {"event": "tool_call_start", "data": {"name": "search_laws", "status": "started"}} in events
    assert {"event": "tool_call_result", "data": {"name": "search_laws", "status": "success"}} in events
    assert events[-1] == {"event": "agent_final", "data": "最终回答"}
    assert context.tool_call_count == 1


@pytest.mark.asyncio
async def test_tool_call_limit_counts_actual_executions(monkeypatch):
    monkeypatch.setattr("backend.app.agent.middleware._persist_tool_audit", lambda *args: None)
    tool = fake_tool()
    registry = SimpleNamespace(
        get_tools=AsyncMock(return_value=[tool]),
        status=lambda: SimpleNamespace(version=1),
        close=AsyncMock(),
        invalidate=lambda *args: None,
    )
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_laws",
                        "args": {"query": "问题一"},
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_laws",
                        "args": {"query": "问题二"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="基于已获得的结果回答"),
        ]
    )
    runtime = AgentRuntime(
        registry,
        SimpleNamespace(get_chat_model=lambda: model),
        settings(agent_max_tool_calls=1),
    )
    context = AgentInvocationContext(
        RequestUserContext("tenant", "user", "request"), "conversation"
    )

    events = [
        event
        async for event in runtime.stream(
            context,
            [HumanMessage(content="问题")],
            "",
        )
    ]

    starts = [event for event in events if event["event"] == "tool_call_start"]
    assert len(starts) == 1
    assert context.tool_call_count == 1
    assert events[-1] == {"event": "agent_final", "data": "基于已获得的结果回答"}


@pytest.mark.asyncio
async def test_agent_service_keeps_request_state_out_of_shared_runtime():
    class FakeRuntime:
        async def stream(self, context, messages, memory_context):
            assert context.user.user_id == "user-a"
            assert context.conversation_id == "conversation-a"
            assert memory_context == "memory-a"
            assert messages[-1].content == "question-a"
            context.tool_call_count = 2
            yield {"event": "token", "data": "streamed"}
            yield {"event": "agent_final", "data": "final"}

    service = AgentService(
        FakeRuntime(),
        RequestUserContext("tenant", "user-a", "request-a"),
        "conversation-a",
    )

    events = [event async for event in service.run("memory-a", [], "question-a")]

    assert events == [{"event": "token", "data": "streamed"}]
    assert service.final_answer == "final"
    assert service.tool_call_count == 2


def test_llm_provider_rejects_missing_api_key():
    provider = LLMProvider(settings(deepseek_api_key=""))

    with pytest.raises(AgentConfigurationError):
        provider.get_chat_model()
