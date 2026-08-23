import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from backend.app.agent.concurrency import (
    AgentConcurrencyManager,
    ConcurrencyIdentity,
    ConversationBusyError,
)
from backend.app.agent.graph import (
    ANALYST_PROMPT,
    COUNSEL_PROMPT,
    REVIEW_PROMPT,
    _authoritative_evidence,
    _citation_errors,
    _no_match_violations,
)
from backend.app.agent.middleware import result_metadata
from backend.app.agent.provider import AgentConfigurationError, LLMProvider
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.runtime import AgentRuntime
from backend.app.agent.schemas import CaseAnalysis, CounselDraft, EvidenceItem, EvidencePacket
from backend.app.agent.service import AgentService
from backend.app.agent.state import AgentInvocationContext, AgentInvocationIdentity
from backend.app.core.config import Settings
from backend.app.core.context import RequestUserContext


def settings(**changes):
    values = {
        "deepseek_api_key": "test-key",
        "mcp_tool_timeout_seconds": 1,
        "mcp_tool_discovery_retry_seconds": 30,
        "agent_max_model_calls": 8,
    }
    values.update(changes)
    return Settings(_env_file=None, **values)


def test_agent_prompts_prioritize_latest_user_facts_over_memory():
    assert "必须采用当前用户最新明确陈述的事实" in ANALYST_PROMPT
    assert "以最新消息" in COUNSEL_PROMPT
    assert "与用户最新消息" in REVIEW_PROMPT


def test_memory_model_is_non_streaming_non_thinking_and_separate(monkeypatch):
    created = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr("backend.app.agent.provider.ChatOpenAI", FakeChatOpenAI)
    provider = LLMProvider(settings(deepseek_model="deepseek-v4-flash"))

    agent_model = provider.get_chat_model()
    memory_model = provider.get_memory_model()

    assert agent_model is not memory_model
    assert created[0]["streaming"] is True
    assert "extra_body" not in created[0]
    assert created[1]["streaming"] is False
    assert created[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert created[1]["max_tokens"] == 4096


def fake_tool():
    async def search_laws(query: str) -> str:
        """Search laws for a query."""
        return json.dumps([
            {
                "document_id": "law-1",
                "chunk_id": "chunk-1",
                "law_name": "劳动合同法",
                "article_number": "第八十二条",
                "content": query,
            }
        ], ensure_ascii=False)

    return StructuredTool.from_function(coroutine=search_laws)


def empty_tool():
    async def search_laws(query: str) -> str:
        """Search laws for a query and return no candidates."""
        return "[]"

    return StructuredTool.from_function(coroutine=search_laws)


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def invocation(user="user", conversation="conversation", request="request"):
    return AgentInvocationContext(
        AgentInvocationIdentity(request, "tenant", user, conversation)
    )


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
async def test_three_agent_graph_routes_casual_chat_without_research():
    registry = SimpleNamespace(
        get_tools=AsyncMock(return_value=[]),
        status=lambda: SimpleNamespace(version=0),
        close=AsyncMock(),
    )
    analysis = {
        "request_type": "casual_chat",
        "case_summary": "问候",
        "next_action": "direct_answer",
        "direct_answer": "您好，请问有什么法律问题？",
    }
    provider = SimpleNamespace(
        get_chat_model=lambda: GenericFakeChatModel(messages=iter([json.dumps(analysis, ensure_ascii=False)]))
    )
    runtime = AgentRuntime(registry, provider, settings())
    context = invocation()

    events = [
        event
        async for event in runtime.stream(context, [HumanMessage(content="你好")], "用户记忆")
    ]

    assert any(event["event"] == "agent_status" and event["data"]["status"] == "analyzing" for event in events)
    assert not any(event["event"] == "tool_call_start" for event in events)
    assert events[-1] == {"event": "agent_final", "data": "您好，请问有什么法律问题？"}
    assert context.metrics.model_call_count == 1


@pytest.mark.asyncio
async def test_three_agent_graph_researches_drafts_reviews_and_cites(monkeypatch):
    monkeypatch.setattr("backend.app.agent.middleware._persist_tool_audit", lambda *args: None)
    analysis = {
        "request_type": "legal_consultation",
        "case_summary": "未签劳动合同",
        "legal_issues": ["未签合同责任"],
        "research_tasks": [{"issue_id": "issue-1", "query": "未签劳动合同 二倍工资", "purpose": "核验责任"}],
        "next_action": "research",
    }
    evidence = {
        "research_tasks": analysis["research_tasks"],
        "evidence_items": [{
            "document_id": "law-1",
            "chunk_id": "chunk-1",
            "law_name": "劳动合同法",
            "article_number": "第八十二条",
            "content": "用人单位未依法订立书面劳动合同，应当依法承担责任。",
            "supports_issue_ids": ["issue-1"],
            "retrieval_sources": ["bm25"],
            "verification_status": "retrieved",
            "data_version": "v1",
        }],
        "unresolved_issues": [],
        "conflicts": [],
        "research_summary": "已找到相关依据",
    }
    draft = {
        "answer": "依据《劳动合同法》第八十二条，可以依法主张权利。",
        "claims": [{"claim": "可以依法主张权利", "evidence_chunk_ids": ["chunk-1"]}],
        "confidence": "high",
        "limitations": [],
        "follow_up_questions": [],
    }
    review = {"approved": True, "next_action": "finalize"}
    model = ToolCallingFakeModel(responses=[
        AIMessage(content=json.dumps(analysis, ensure_ascii=False)),
        AIMessage(content="", tool_calls=[{
            "name": "search_laws",
            "args": {"query": "未签劳动合同 二倍工资"},
            "id": "call-1",
            "type": "tool_call",
        }]),
        AIMessage(content=json.dumps(evidence, ensure_ascii=False)),
        AIMessage(content=json.dumps(draft, ensure_ascii=False)),
        AIMessage(content=json.dumps(review, ensure_ascii=False)),
    ])
    tool = fake_tool()
    registry = SimpleNamespace(
        get_tools=AsyncMock(return_value=[tool]),
        status=lambda: SimpleNamespace(version=1),
        close=AsyncMock(),
        invalidate=lambda *args: None,
    )
    runtime = AgentRuntime(registry, SimpleNamespace(get_chat_model=lambda: model), settings())
    context = invocation()

    events = [event async for event in runtime.stream(context, [HumanMessage(content="问题")], "")]

    stages = [event["data"]["status"] for event in events if event["event"] == "agent_status"]
    assert stages == ["analyzing", "researching", "drafting", "reviewing", "completed"]
    assert any(event["event"] == "tool_call_start" for event in events)
    citations = next(event["data"] for event in events if event["event"] == "citations")
    assert citations[0]["document_id"] == "law-1"
    assert citations[0]["chunk_id"] == "chunk-1"
    assert events[-1]["data"] == draft["answer"]
    assert context.metrics.tool_call_count == 1


@pytest.mark.asyncio
async def test_no_match_is_successful_and_does_not_return_to_research(monkeypatch):
    monkeypatch.setattr("backend.app.agent.middleware._persist_tool_audit", lambda *args: None)
    analysis = {
        "request_type": "legal_consultation",
        "case_summary": "借款到期未还",
        "legal_issues": ["如何处理逾期借款"],
        "research_tasks": [{"issue_id": 1, "query": "借款到期未还", "purpose": "查找依据"}],
        "next_action": "research",
    }
    research = {
        "retrieval_status": "matched",
        "research_tasks": analysis["research_tasks"],
        "evidence_items": [],
        "unresolved_issues": [{"issue_id": 1, "description": "当前法规库没有可引用结果"}],
        "research_summary": "检索正常完成，但未找到可引用法条。",
    }
    answer = (
        "## 初步判断\n\n可以先整理借款约定、付款记录和催收记录，再结合完整事实选择处理方式。"
        "\n\n## 检索说明\n\n本轮法规检索正常完成，但当前法规库中未检索到可引用法条。"
        "以上属于一般性分析，不构成已经过法规核验的确定性法律结论。"
    )
    draft = {
        "answer": answer,
        "claims": [],
        "confidence": "low",
        "limitations": ["当前没有可引用法条"],
        "follow_up_questions": [],
    }
    # The reviewer asks to research again, but no_match must terminate research
    # and approve a boundary-compliant general answer instead.
    review = {"approved": False, "next_action": "research_again"}
    model = ToolCallingFakeModel(responses=[
        AIMessage(content=json.dumps(analysis, ensure_ascii=False)),
        AIMessage(content="", tool_calls=[{
            "name": "search_laws",
            "args": {"query": "借款到期未还"},
            "id": "call-empty",
            "type": "tool_call",
        }]),
        AIMessage(content=json.dumps(research, ensure_ascii=False)),
        AIMessage(content=json.dumps(draft, ensure_ascii=False)),
        AIMessage(content=json.dumps(review, ensure_ascii=False)),
    ])
    registry = SimpleNamespace(
        get_tools=AsyncMock(return_value=[empty_tool()]),
        status=lambda: SimpleNamespace(version=1),
        close=AsyncMock(),
        invalidate=lambda *args: None,
    )
    runtime = AgentRuntime(registry, SimpleNamespace(get_chat_model=lambda: model), settings())
    context = invocation()

    events = [event async for event in runtime.stream(context, [HumanMessage(content="问题")], "")]

    stages = [event["data"]["status"] for event in events if event["event"] == "agent_status"]
    assert stages == ["analyzing", "researching", "drafting", "reviewing", "completed"]
    assert context.metrics.tool_call_count == 1
    assert context.metrics.model_call_count == 5
    assert not any(event["event"] == "citations" for event in events)
    assert events[-1] == {"event": "agent_final", "data": answer}


def test_no_match_boundary_detects_unverified_law_references():
    safe = "当前法规库中未检索到可引用法条，以下仅为一般性分析。"
    unsafe = "依据《民法典》第一百八十八条处理。当前法规库中未检索到可引用法条。"

    assert _no_match_violations(safe) == []
    assert _no_match_violations(unsafe) == ["无法条模式包含未经检索核验的法律名称或条号"]


def test_authoritative_evidence_uses_exact_chunk_and_rejects_ambiguous_document_id():
    candidates = [
        {"document_id": "law-1", "chunk_id": "chunk-1", "content": "前半段"},
        {"document_id": "law-1", "chunk_id": "chunk-2", "content": "后半段"},
    ]
    selected = EvidencePacket(evidence_items=[EvidenceItem(
        document_id="law-1", chunk_id="chunk-2", supports_issue_ids=["issue-1"]
    )])
    legacy_ambiguous = EvidencePacket(evidence_items=[EvidenceItem(document_id="law-1")])

    accepted = _authoritative_evidence(selected, candidates)

    assert len(accepted) == 1
    assert accepted[0].chunk_id == "chunk-2"
    assert accepted[0].content == "后半段"
    assert _authoritative_evidence(legacy_ambiguous, candidates) == []


@pytest.mark.asyncio
async def test_low_risk_no_match_uses_deterministic_review_fast_path(monkeypatch):
    monkeypatch.setattr("backend.app.agent.middleware._persist_tool_audit", lambda *args: None)
    analysis = {
        "request_type": "legal_consultation",
        "case_summary": "一般咨询",
        "legal_issues": ["一般处理建议"],
        "research_tasks": [{"issue_id": "issue-1", "query": "一般咨询", "purpose": "查找依据"}],
        "risk_level": "low",
        "next_action": "research",
    }
    research = {
        "retrieval_status": "no_match",
        "research_tasks": analysis["research_tasks"],
        "evidence_items": [],
        "unresolved_issues": [{"issue_id": "issue-1", "description": "没有可引用结果"}],
        "research_summary": "检索完成但没有匹配。",
    }
    answer = (
        "可以先保存资料并补充事实。\n\n本轮法规检索正常完成，但当前法规库中"
        "未检索到可引用法条。以上属于一般性分析，不构成已经过法规核验的确定性法律结论。"
    )
    draft = {"answer": answer, "claims": [], "confidence": "low"}
    model = ToolCallingFakeModel(responses=[
        AIMessage(content=json.dumps(analysis, ensure_ascii=False)),
        AIMessage(content="", tool_calls=[{
            "name": "search_laws", "args": {"query": "一般咨询"},
            "id": "call-empty-fast", "type": "tool_call",
        }]),
        AIMessage(content=json.dumps(research, ensure_ascii=False)),
        AIMessage(content=json.dumps(draft, ensure_ascii=False)),
    ])
    registry = SimpleNamespace(
        get_tools=AsyncMock(return_value=[empty_tool()]),
        status=lambda: SimpleNamespace(version=1),
        close=AsyncMock(),
        invalidate=lambda *args: None,
    )
    runtime = AgentRuntime(registry, SimpleNamespace(get_chat_model=lambda: model), settings())
    context = invocation()

    events = [event async for event in runtime.stream(context, [HumanMessage(content="问题")], "")]

    stages = [event["data"]["status"] for event in events if event["event"] == "agent_status"]
    assert stages == ["analyzing", "researching", "drafting", "completed"]
    assert context.metrics.model_call_count == 4
    assert events[-1] == {"event": "agent_final", "data": answer}


def test_tool_audit_parses_documents_nested_in_mcp_text_blocks():
    nested = json.dumps([{
        "type": "text",
        "text": json.dumps([{
            "document_id": "law-1",
            "law_name": "示例法",
            "article_number": "第一条",
        }], ensure_ascii=False),
    }], ensure_ascii=False)

    metadata = result_metadata(nested)

    assert metadata["result_count"] == 1
    assert metadata["documents"] == [{
        "document_id": "law-1",
        "law_name": "示例法",
        "article_number": "第一条",
    }]


@pytest.mark.asyncio
async def test_agent_service_keeps_request_state_out_of_shared_runtime():
    class FakeRuntime:
        async def stream(self, context, messages, memory_context):
            assert context.identity.user_id == "user-a"
            assert context.identity.conversation_id == "conversation-a"
            assert memory_context == "memory-a"
            assert messages[-1].content == "question-a"
            context.metrics.tool_call_count = 2
            context.metrics.model_call_count = 4
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
    assert service.model_call_count == 4


@pytest.mark.asyncio
async def test_concurrency_rejects_duplicate_conversation_and_limits_per_user():
    manager = AgentConcurrencyManager(settings(agent_global_concurrency=2, agent_per_user_concurrency=1))
    first = ConcurrencyIdentity("r1", "tenant", "user-a", "conversation-a")
    duplicate = ConcurrencyIdentity("r2", "tenant", "user-a", "conversation-a")
    second = ConcurrencyIdentity("r3", "tenant", "user-a", "conversation-b")
    other_user = ConcurrencyIdentity("r4", "tenant", "user-b", "conversation-c")

    await manager.reserve(first)
    with pytest.raises(ConversationBusyError):
        await manager.reserve(duplicate)
    await manager.acquire(first)
    assert await manager.would_queue(second)
    assert not await manager.would_queue(other_user)
    await manager.release(first)
    await manager.release_reservation(first)


def test_llm_provider_rejects_missing_api_key():
    provider = LLMProvider(settings(deepseek_api_key=""))

    with pytest.raises(AgentConfigurationError):
        provider.get_chat_model()


def test_deterministic_review_rejects_law_article_outside_evidence_packet():
    packet = EvidencePacket(evidence_items=[EvidenceItem(
        document_id="law-1",
        law_name="中华人民共和国劳动合同法",
        article_number="第八十二条",
        content="已核验内容",
    )])
    draft = CounselDraft(
        answer="依据《劳动合同法》第八十二条可以主张权利，但《民法典》第五百条另有规定。",
        claims=[],
    )

    errors = _citation_errors(draft, packet)

    assert not any("劳动合同法" in error for error in errors)
    assert any("《民法典》第五百条" in error for error in errors)


def test_case_analysis_normalizes_nullable_optional_model_fields():
    analysis = CaseAnalysis.model_validate({
        "request_type": "legal_consultation",
        "case_summary": None,
        "jurisdiction": None,
        "legal_domain": None,
        "key_facts": None,
        "missing_facts": None,
        "legal_issues": None,
        "research_tasks": None,
        "next_action": "research",
        "direct_answer": None,
        "clarification_questions": None,
    })

    assert analysis.case_summary == ""
    assert analysis.jurisdiction == "中国大陆"
    assert analysis.legal_domain == "其他"
    assert analysis.direct_answer == ""
    assert analysis.key_facts == []
    assert analysis.research_tasks == []
