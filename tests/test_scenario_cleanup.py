
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api import routes
from backend.app.core.context import RequestUserContext
from backend.app.db.models import AgentRun, Base, Conversation, Tenant, User


def _database(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(routes, "SessionLocal", sessions)
    with sessions() as db:
        db.add(Tenant(id="tenant", name="tenant"))
        db.add_all([
            User(id="user-a", tenant_id="tenant", name="A"),
            User(id="user-b", tenant_id="tenant", name="B"),
            Conversation(
                id="scenario-a",
                tenant_id="tenant",
                user_id="user-a",
                title="[场景] sample-01 · primary/main",
            ),
            Conversation(
                id="ordinary-a",
                tenant_id="tenant",
                user_id="user-a",
                title="普通会话",
            ),
        ])
        db.commit()
    return sessions


def test_cleanup_is_owner_and_prefix_scoped(monkeypatch):
    sessions = _database(monkeypatch)
    owner = RequestUserContext("tenant", "user-a", "request-a")
    other = RequestUserContext("tenant", "user-b", "request-b")

    with pytest.raises(LookupError):
        routes._delete_scenario_conversation(other, "scenario-a")
    with pytest.raises(PermissionError):
        routes._delete_scenario_conversation(owner, "ordinary-a")

    assert routes._delete_scenario_conversation(owner, "scenario-a") == []
    with sessions() as db:
        assert db.get(Conversation, "scenario-a") is None
        assert db.get(Conversation, "ordinary-a") is not None


def test_cleanup_rejects_active_run(monkeypatch):
    sessions = _database(monkeypatch)
    owner = RequestUserContext("tenant", "user-a", "request-a")
    with sessions() as db:
        db.add(AgentRun(
            id="run-a",
            request_id="request-run-a",
            tenant_id="tenant",
            user_id="user-a",
            conversation_id="scenario-a",
            input_text="问题",
            status="running",
            langgraph_thread_id="agent-run:run-a",
        ))
        db.commit()

    with pytest.raises(RuntimeError, match="任务运行"):
        routes._delete_scenario_conversation(owner, "scenario-a")


def test_disabled_catalog_is_hidden_as_not_found():
    request = type("Request", (), {
        "app": type("App", (), {
            "state": type("State", (), {
                "scenario_catalog": type("Catalog", (), {"enabled": False})()
            })()
        })()
    })()

    with pytest.raises(HTTPException) as raised:
        routes._scenario_catalog(request)

    assert raised.value.status_code == 404
