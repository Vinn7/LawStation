from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from backend.app.api.routes import message_feedback
from backend.app.core.context import RequestUserContext
from backend.app.db.models import Conversation, Message, MessageFeedback, Tenant, User
from backend.app.db.session import Base
from backend.app.schemas import MessageFeedbackRequest


def feedback_db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, expire_on_commit=False)()
    db.add(Tenant(id="t", name="tenant"))
    db.add_all([
        User(id="a", tenant_id="t", name="A"),
        User(id="b", tenant_id="t", name="B"),
    ])
    db.add_all([
        Conversation(id="ca", tenant_id="t", user_id="a"),
        Conversation(id="cb", tenant_id="t", user_id="b"),
    ])
    db.add_all([
        Message(id="ma", tenant_id="t", user_id="a", conversation_id="ca", role="assistant", content="A", langsmith_trace_id="00000000-0000-0000-0000-000000000001"),
        Message(id="mb", tenant_id="t", user_id="b", conversation_id="cb", role="assistant", content="B"),
    ])
    db.commit()
    return db


@pytest.mark.asyncio
async def test_feedback_is_owned_and_persisted_before_export():
    db = feedback_db()
    app = SimpleNamespace(state=SimpleNamespace(langsmith_observability=SimpleNamespace()))
    request = Request({"type": "http", "app": app, "headers": []})
    tasks = BackgroundTasks()
    ctx = RequestUserContext("t", "a", "request-a")

    result = await message_feedback(
        "ma", MessageFeedbackRequest(score=-1, comment="引用不足"), request, tasks, ctx, db
    )
    assert result == {"ok": True, "sync_status": "pending"}
    saved = db.query(MessageFeedback).one()
    assert (saved.tenant_id, saved.user_id, saved.message_id, saved.score) == ("t", "a", "ma", -1)

    with pytest.raises(HTTPException) as exc:
        await message_feedback(
            "mb", MessageFeedbackRequest(score=1), request, BackgroundTasks(), ctx, db
        )
    assert exc.value.status_code == 404
