import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agent.concurrency import (
    AgentConcurrencyManager,
    AgentQueueTimeoutError,
    ConcurrencyIdentity,
    ConversationBusyError,
)
from backend.app.agent.service import AgentService
from backend.app.core.context import RequestUserContext, get_user_context
from backend.app.core.logging import audit, summary
from backend.app.db.models import Conversation, Message, User
from backend.app.db.session import SessionLocal, get_db
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


def _prepare_chat(ctx: RequestUserContext, conversation_id: str, content: str):
    with SessionLocal() as db:
        repo = OwnedRepository(db, ctx)
        if not repo.conversation(conversation_id):
            raise HTTPException(404, "会话不存在或无权访问")
        user_message = Message(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            conversation_id=conversation_id,
            role="user",
            content=content,
        )
        db.add(user_message)
        db.commit()
        memory_context, history = MemoryService(db, ctx).context(conversation_id)
        history = [message for message in history if message.id != user_message.id]
        return memory_context, history


def _save_assistant(
    ctx: RequestUserContext,
    conversation_id: str,
    content: str,
    status: str,
) -> Message:
    with SessionLocal() as db:
        assistant = Message(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            status=status,
        )
        db.add(assistant)
        db.commit()
        return assistant


def _consolidate_memory(ctx: RequestUserContext, conversation_id: str) -> bool:
    with SessionLocal() as db:
        return MemoryService(db, ctx).consolidate(conversation_id)


@router.post("/conversations/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: str,
    payload: ChatRequest,
    request: Request,
    ctx=Depends(get_user_context),
):
    started = time.perf_counter()
    concurrency: AgentConcurrencyManager = request.app.state.agent_concurrency
    identity = ConcurrencyIdentity(
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=conversation_id,
    )
    try:
        await concurrency.reserve(identity)
    except ConversationBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    audit("conversation.received", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="received", question_chars=len(payload.content), question_summary=summary(payload.content))
    try:
        memory_context, history = await asyncio.to_thread(
            _prepare_chat, ctx, conversation_id, payload.content
        )
    except Exception:
        await concurrency.release_reservation(identity)
        raise
    audit("conversation.started", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="started", history_count=len(history), memory_context_chars=len(memory_context))

    async def events():
        answer = []
        agent = AgentService(request.app.state.agent_runtime, ctx, conversation_id)
        try:
            yield sse("message_start", {"request_id": ctx.request_id})
            if await concurrency.would_queue(identity):
                yield sse("agent_status", {
                    "request_id": ctx.request_id,
                    "user_id": ctx.user_id,
                    "conversation_id": conversation_id,
                    "agent": "coordinator",
                    "status": "queued",
                    "message": "咨询任务正在排队",
                })
            async with concurrency.slot(identity):
                async for item in agent.run(memory_context, history, payload.content):
                    if item["event"] == "token":
                        answer.append(item["data"])
                    yield sse(item["event"], item["data"])
            persisted_answer = agent.final_answer or "".join(answer)
            assistant = await asyncio.to_thread(
                _save_assistant, ctx, conversation_id, persisted_answer, "complete"
            )
            compressed = await asyncio.to_thread(_consolidate_memory, ctx, conversation_id)
            audit("conversation.completed", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="success", duration_ms=int((time.perf_counter()-started)*1000), answer_chars=len(assistant.content), answer_summary=summary(assistant.content), memory_compressed=compressed, tool_call_count=agent.tool_call_count, model_call_count=agent.model_call_count)
            yield sse("memory_status", {"compressed": compressed})
            yield sse("message_end", {"message_id": assistant.id})
        except AgentQueueTimeoutError as exc:
            audit("conversation.failed", level=logging.WARNING, request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="queue_timeout", duration_ms=int((time.perf_counter()-started)*1000))
            yield sse("error", {"message": str(exc)})
        except asyncio.CancelledError:
            if answer:
                await asyncio.to_thread(
                    _save_assistant, ctx, conversation_id, "".join(answer), "interrupted"
                )
            audit("conversation.interrupted", level=logging.WARNING, request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="interrupted", duration_ms=int((time.perf_counter()-started)*1000), answer_chars=len("".join(answer)))
            raise
        except Exception as exc:
            if answer:
                await asyncio.to_thread(
                    _save_assistant, ctx, conversation_id, "".join(answer), "interrupted"
                )
            audit("conversation.failed", level=logging.ERROR, request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="failed", duration_ms=int((time.perf_counter()-started)*1000), error_type=type(exc).__name__, error=summary(str(exc)))
            yield sse("error", {"message": str(exc)})
        finally:
            await concurrency.release_reservation(identity)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
