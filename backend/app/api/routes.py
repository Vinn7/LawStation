import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent.service import AgentService
from backend.app.core.context import RequestUserContext, get_user_context
from backend.app.core.logging import audit, summary
from backend.app.db.models import Conversation, Message, User
from backend.app.db.session import get_db
from backend.app.schemas import ChatRequest, ConversationCreate, MemoryUpdate
from backend.app.services.memory import MemoryService
from backend.app.services.repositories import OwnedRepository
from mcp_servers.law_rag.server import get_index_status

router = APIRouter(prefix="/api")


@router.get("/index/status")
def index_status():
    return get_index_status()


def obj(row):
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


@router.get("/users")
def users(db: Session = Depends(get_db)):
    return [obj(user) for user in db.scalars(select(User).order_by(User.name))]


@router.get("/conversations")
def conversations(ctx: RequestUserContext = Depends(get_user_context), db: Session = Depends(get_db)):
    return [obj(item) for item in OwnedRepository(db, ctx).conversations()]


@router.post("/conversations", status_code=201)
def create_conversation(payload: ConversationCreate, ctx=Depends(get_user_context), db=Depends(get_db)):
    item = Conversation(tenant_id=ctx.tenant_id, user_id=ctx.user_id, title=payload.title)
    db.add(item)
    db.commit()
    return obj(item)


@router.get("/conversations/{conversation_id}/messages")
def messages(conversation_id: str, ctx=Depends(get_user_context), db=Depends(get_db)):
    repo = OwnedRepository(db, ctx)
    if not repo.conversation(conversation_id):
        raise HTTPException(404, "会话不存在或无权访问")
    return [obj(item) for item in repo.messages(conversation_id)]


@router.get("/memories")
def memories(conversation_id: str | None = Query(None), ctx=Depends(get_user_context), db=Depends(get_db)):
    repo = OwnedRepository(db, ctx)
    if conversation_id and not repo.conversation(conversation_id):
        raise HTTPException(404, "会话不存在或无权访问")
    return [obj(item) for item in repo.memories(conversation_id)]


@router.patch("/memories/{memory_id}")
def update_memory(memory_id: str, payload: MemoryUpdate, ctx=Depends(get_user_context), db=Depends(get_db)):
    if not OwnedRepository(db, ctx).update_memory(memory_id, payload.content):
        raise HTTPException(404, "记忆不存在或无权访问")
    db.commit()
    return {"ok": True}


@router.delete("/memories/{memory_id}", status_code=204)
def delete_memory(memory_id: str, ctx=Depends(get_user_context), db=Depends(get_db)):
    if not OwnedRepository(db, ctx).delete_memory(memory_id):
        raise HTTPException(404, "记忆不存在或无权访问")
    db.commit()


@router.delete("/memories", status_code=204)
def clear_memories(conversation_id: str | None = Query(None), ctx=Depends(get_user_context), db=Depends(get_db)):
    repo = OwnedRepository(db, ctx)
    if conversation_id and not repo.conversation(conversation_id):
        raise HTTPException(404, "会话不存在或无权访问")
    repo.clear_memories(conversation_id)
    db.commit()


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/conversations/{conversation_id}/messages/stream")
def stream_message(conversation_id: str, payload: ChatRequest, ctx=Depends(get_user_context), db=Depends(get_db)):
    started = time.perf_counter()
    repo = OwnedRepository(db, ctx)
    conversation = repo.conversation(conversation_id)
    if not conversation:
        raise HTTPException(404, "会话不存在或无权访问")
    audit("conversation.received", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="received", question_chars=len(payload.content), question_summary=summary(payload.content))
    user_message = Message(
        tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id,
        role="user", content=payload.content,
    )
    db.add(user_message)
    db.commit()
    memory_context, history = MemoryService(db, ctx).context(conversation_id)
    history = [m for m in history if m.id != user_message.id]
    audit("conversation.started", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="started", history_count=len(history), memory_context_chars=len(memory_context))

    async def events():
        answer = []
        agent = AgentService(db, ctx, conversation_id)
        yield sse("message_start", {"request_id": ctx.request_id})
        try:
            async for item in agent.run(memory_context, history, payload.content):
                if item["event"] == "token":
                    answer.append(item["data"])
                yield sse(item["event"], item["data"])
            assistant = Message(
                tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id,
                role="assistant", content="".join(answer), status="complete",
            )
            db.add(assistant)
            db.commit()
            compressed = MemoryService(db, ctx).consolidate(conversation_id)
            audit("conversation.completed", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="success", duration_ms=int((time.perf_counter()-started)*1000), answer_chars=len(assistant.content), answer_summary=summary(assistant.content), memory_compressed=compressed, tool_call_count=agent.tool_call_count)
            yield sse("memory_status", {"compressed": compressed})
            yield sse("message_end", {"message_id": assistant.id})
        except asyncio.CancelledError:
            if answer:
                db.add(Message(tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, role="assistant", content="".join(answer), status="interrupted"))
                db.commit()
            audit("conversation.interrupted", level=logging.WARNING, request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="interrupted", duration_ms=int((time.perf_counter()-started)*1000), answer_chars=len("".join(answer)))
            raise
        except Exception as exc:
            if answer:
                db.add(Message(
                    tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id,
                    role="assistant", content="".join(answer), status="interrupted",
                ))
                db.commit()
            audit("conversation.failed", level=logging.ERROR, request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="failed", duration_ms=int((time.perf_counter()-started)*1000), error_type=type(exc).__name__, error=summary(str(exc)))
            yield sse("error", {"message": str(exc)})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
