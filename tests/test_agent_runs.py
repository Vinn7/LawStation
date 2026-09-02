from types import SimpleNamespace
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.agent.checkpoint import checkpoint_saver
from backend.app.core.context import RequestUserContext
from backend.app.db.models import AgentRun, Base, Conversation, Tenant, User
from backend.app.services import agent_runs as run_module
from backend.app.services.agent_runs import AgentRunConflict, AgentRunManager


def _database(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(run_module, "SessionLocal", sessions)
    with sessions() as db:
        db.add(Tenant(id="tenant", name="tenant"))
        db.add_all([
            User(id="user-a", tenant_id="tenant", name="A"),
            User(id="user-b", tenant_id="tenant", name="B"),
            Conversation(id="conversation-a", tenant_id="tenant", user_id="user-a"),
            Conversation(id="conversation-b", tenant_id="tenant", user_id="user-b"),
        ])
        db.commit()
    return sessions


def _manager():
    settings = SimpleNamespace(agent_run_recovery_max_attempts=2)
    return AgentRunManager(None, None, None, None, settings=settings)


class _ScenarioRuntime:
    async def scenario_outcome(self, thread_id):
        assert thread_id.startswith("agent-run:")
        return {
            "checkpoint_available": True,
            "retrieval_status": "matched",
            "citation_count": 1,
        }


def test_agent_run_creation_is_owner_scoped_and_conversation_unique(monkeypatch):
    sessions = _database(monkeypatch)
    manager = _manager()
    owner = RequestUserContext("tenant", "user-a", "request-a")
    other = RequestUserContext("tenant", "user-b", "request-b")

    created = manager.create(owner, "conversation-a", "问题")

    assert manager.owned(owner, created.id).id == created.id
    assert manager.owned(other, created.id) is None
    assert manager.events(other, created.id, 0) == (None, [])
    with pytest.raises(AgentRunConflict):
        manager.create(owner, "conversation-a", "重复问题")
    with sessions() as db:
        assert db.query(AgentRun).count() == 1


@pytest.mark.asyncio
async def test_queued_run_cancel_is_replayable_and_idempotent(monkeypatch):
    _database(monkeypatch)
    manager = _manager()
    owner = RequestUserContext("tenant", "user-a", "request-a")
    created = manager.create(owner, "conversation-a", "问题")

    cancelled = await manager.cancel(owner, created.id)
    again = await manager.cancel(owner, created.id)
    run, events = manager.events(owner, created.id, 0)

    assert cancelled.status == "interrupted"
    assert again.status == "interrupted"
    assert run.status == "interrupted"
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[-1].event_type == "message_end"


@pytest.mark.asyncio
async def test_scenario_outcome_is_owner_scoped_and_checkpoint_safe(monkeypatch):
    _database(monkeypatch)
    manager = _manager()
    manager.runtime = _ScenarioRuntime()
    owner = RequestUserContext("tenant", "user-a", "request-a")
    other = RequestUserContext("tenant", "user-b", "request-b")
    created = manager.create(owner, "conversation-a", "问题")

    outcome = await manager.scenario_outcome(owner, created.id)

    assert outcome == {
        "terminal_status": "queued",
        "observed_events": ["message_start", "agent_status"],
        "retrieval_status": "matched",
        "citation_count": 1,
        "model_call_count": 0,
        "tool_call_count": 0,
        "last_event_sequence": 2,
        "checkpoint_available": True,
    }
    assert await manager.scenario_outcome(other, created.id) is None


@pytest.mark.asyncio
async def test_async_sqlite_saver_persists_state_without_pickle(tmp_path):
    settings = SimpleNamespace(
        langgraph_checkpoint_enabled=True,
        langgraph_checkpoint_path=str(tmp_path / "checkpoints.db"),
        langgraph_strict_msgpack=True,
    )
    async with checkpoint_saver(settings) as saver:
        config = {
            "configurable": {
                "thread_id": "agent-run:test",
                # 低层 Saver API 要求显式 namespace；根图对应空字符串。
                # 业务层调用 compiled Graph 时只传 thread_id，由 LangGraph 补入该值。
                "checkpoint_ns": "",
            }
        }
        checkpoint = {
            "v": 4,
            "ts": "2026-08-26T00:00:00+00:00",
            "id": "00000000-0000-0000-0000-000000000001",
            "channel_values": {"final_answer": "已恢复"},
            "channel_versions": {"final_answer": "1"},
            "versions_seen": {},
            "updated_channels": ["final_answer"],
        }
        await saver.aput(config, checkpoint, {}, {})
        loaded = await saver.aget_tuple(config)

    assert loaded is not None
    assert loaded.checkpoint["channel_values"]["final_answer"] == "已恢复"


@pytest.mark.asyncio
async def test_compiled_root_graph_reads_checkpoint_without_subgraph_namespace(tmp_path):
    class RootState(TypedDict):
        value: int

    settings = SimpleNamespace(
        langgraph_checkpoint_enabled=True,
        langgraph_checkpoint_path=str(tmp_path / "compiled-checkpoints.db"),
        langgraph_strict_msgpack=True,
    )
    async with checkpoint_saver(settings) as saver:
        builder = StateGraph(RootState)
        builder.add_node("increment", lambda state: {"value": state["value"] + 1})
        builder.add_edge(START, "increment")
        builder.add_edge("increment", END)
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "agent-run:compiled-root"}}

        updates = [
            update
            async for update in graph.astream(
                {"value": 0}, config=config, stream_mode="updates"
            )
        ]
        snapshot = await graph.aget_state(config)

    assert updates == [{"increment": {"value": 1}}]
    assert snapshot.values["value"] == 1
    assert snapshot.next == ()
    assert snapshot.config["configurable"]["checkpoint_ns"] == ""
    assert snapshot.config["configurable"]["checkpoint_id"]
