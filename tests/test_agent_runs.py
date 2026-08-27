from types import SimpleNamespace

import pytest
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
                "checkpoint_ns": "lawstation-consultation-v1",
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
