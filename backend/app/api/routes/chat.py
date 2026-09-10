import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.app.agent.concurrency import (
    AgentConcurrencyManager,
    AgentQueueTimeoutError,
    ConcurrencyIdentity,
    ConversationBusyError,
)
from backend.app.agent.service import AgentService
from backend.app.core.config import get_settings
from backend.app.core.context import RequestUserContext, get_user_context
from backend.app.core.logging import audit, summary
from backend.app.db.models import Message
from backend.app.db.session import SessionLocal
from backend.app.schemas import ChatRequest
from backend.app.services.memory import MemoryService
from backend.app.services.repositories import OwnedRepository

from .helpers import sse, with_sse_heartbeat

router = APIRouter(prefix="/api")


def _validate_conversation(ctx: RequestUserContext, conversation_id: str) -> None:
    with SessionLocal() as db:
        if not OwnedRepository(db, ctx).conversation(conversation_id):
            raise HTTPException(404, "会话不存在或无权访问")


def _prepare_chat(
    ctx: RequestUserContext,
    conversation_id: str,
    content: str,
    langsmith_trace_id: str | None = None,
):
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
            langsmith_trace_id=langsmith_trace_id,
        )
        db.add(user_message)
        db.commit()
        memory_context, history = MemoryService(db, ctx).context(
            conversation_id, content, user_message.id
        )
        return user_message.id, memory_context, history


def _save_assistant(
    ctx: RequestUserContext,
    conversation_id: str,
    content: str,
    status: str,
    langsmith_trace_id: str | None = None,
) -> Message:
    with SessionLocal() as db:
        assistant = Message(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            status=status,
            langsmith_trace_id=langsmith_trace_id,
        )
        db.add(assistant)
        db.commit()
        return assistant


@router.post("/conversations/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: str,
    payload: ChatRequest,
    request: Request,
    ctx=Depends(get_user_context),
):
    started = time.perf_counter()
    observability = request.app.state.langsmith_observability
    consultation_trace = observability.start_consultation(
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=conversation_id,
        question=payload.content,
        model_name=get_settings().deepseek_model,
    )
    concurrency: AgentConcurrencyManager = request.app.state.agent_concurrency
    identity = ConcurrencyIdentity(
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=conversation_id,
    )
    try:
        with consultation_trace.activate():
            async with consultation_trace.span(
                "chat.reserve_conversation",
                inputs={"conversation_id": conversation_id},
            ):
                await concurrency.reserve(identity)
    except ConversationBusyError as exc:
        await consultation_trace.finish(
            outputs={"status": "conversation_busy"},
            error="ConversationBusyError",
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        with consultation_trace.activate():
            async with consultation_trace.span(
                "chat.validate_conversation",
                inputs={"conversation_id": conversation_id},
            ):
                await asyncio.to_thread(_validate_conversation, ctx, conversation_id)
    except Exception as exc:
        await concurrency.release_reservation(identity)
        await consultation_trace.finish(
            outputs={"status": "validation_failed"},
            error=type(exc).__name__,
        )
        raise
    audit("conversation.received", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="received", question_chars=len(payload.content), question_summary=summary(payload.content))

    async def events():
        answer = []
        citations = []
        agent = AgentService(
            request.app.state.agent_runtime,
            ctx,
            conversation_id,
            trace_config=consultation_trace.config,
            trace_id=consultation_trace.trace_id,
        )
        queue_started = time.perf_counter()
        queue_duration_ms = 0
        slot_acquired = False
        first_status_duration_ms: int | None = None
        first_text_token_duration_ms: int | None = None
        try:
            with consultation_trace.activate():
                yield sse("message_start", {"request_id": ctx.request_id})
                if await concurrency.would_queue(identity):
                    first_status_duration_ms = int((time.perf_counter() - started) * 1000)
                    yield sse("agent_status", {
                        "request_id": ctx.request_id,
                        "user_id": ctx.user_id,
                        "conversation_id": conversation_id,
                        "agent": "coordinator",
                        "status": "queued",
                        "message": "咨询任务正在排队",
                    })
                async with consultation_trace.span("chat.queue"):
                    await concurrency.acquire(identity)
                    slot_acquired = True
                queue_duration_ms = int((time.perf_counter() - queue_started) * 1000)
                async with consultation_trace.span(
                    "chat.load_memory_snapshot",
                    inputs={"question_chars": len(payload.content)},
                ):
                    user_message_id, memory_context, history = await asyncio.to_thread(
                        _prepare_chat,
                        ctx,
                        conversation_id,
                        payload.content,
                        consultation_trace.trace_id,
                    )
                audit("conversation.started", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="started", history_count=len(history), memory_context_chars=len(memory_context))
                async for item in agent.run(memory_context, history, payload.content):
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    if item["event"] == "agent_status" and first_status_duration_ms is None:
                        first_status_duration_ms = elapsed_ms
                    if item["event"] == "token":
                        if first_text_token_duration_ms is None:
                            first_text_token_duration_ms = elapsed_ms
                        answer.append(item["data"])
                    elif item["event"] == "citations":
                        citations = item["data"]
                    yield sse(item["event"], item["data"])
                persisted_answer = agent.final_answer or "".join(answer)
                async with consultation_trace.span(
                    "chat.persist_answer",
                    inputs={"answer_chars": len(persisted_answer)},
                ):
                    assistant = await asyncio.to_thread(
                        _save_assistant, ctx, conversation_id, persisted_answer, "complete",
                        consultation_trace.trace_id,
                    )
                async with consultation_trace.span("chat.enqueue_memory"):
                    job = await asyncio.to_thread(
                        request.app.state.memory_tasks.enqueue,
                        ctx,
                        conversation_id,
                        user_message_id,
                    )
            total_duration_ms = int((time.perf_counter() - started) * 1000)
            audit("conversation.completed", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="success", duration_ms=total_duration_ms, total_duration_ms=total_duration_ms, queue_duration_ms=queue_duration_ms, first_status_duration_ms=first_status_duration_ms, first_text_token_duration_ms=first_text_token_duration_ms, answer_chars=len(assistant.content), answer_summary=summary(assistant.content), memory_job_id=job.id, tool_call_count=agent.tool_call_count, model_call_count=agent.model_call_count)
            yield sse("memory_status", {"status": "pending", "job_id": job.id})
            await consultation_trace.finish(
                outputs={
                    "status": "success",
                    "final_answer": assistant.content,
                    "citations": citations,
                    "model_call_count": agent.model_call_count,
                    "tool_call_count": agent.tool_call_count,
                    "queue_duration_ms": queue_duration_ms,
                    "total_duration_ms": total_duration_ms,
                },
            )
            yield sse("message_end", {"message_id": assistant.id, "trace_available": bool(agent.langsmith_trace_id)})
        except AgentQueueTimeoutError as exc:
            total_duration_ms = int((time.perf_counter() - started) * 1000)
            audit("conversation.failed", level=logging.WARNING, request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="queue_timeout", duration_ms=total_duration_ms, total_duration_ms=total_duration_ms, queue_duration_ms=total_duration_ms, first_status_duration_ms=first_status_duration_ms, first_text_token_duration_ms=None)
            await consultation_trace.finish(
                outputs={"status": "queue_timeout"},
                error="AgentQueueTimeoutError",
            )
            yield sse("error", {"message": str(exc)})
        except asyncio.CancelledError:
            if answer:
                await asyncio.to_thread(
                    _save_assistant, ctx, conversation_id, "".join(answer), "interrupted",
                    agent.langsmith_trace_id,
                )
            total_duration_ms = int((time.perf_counter() - started) * 1000)
            audit("conversation.interrupted", level=logging.WARNING, request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="interrupted", duration_ms=total_duration_ms, total_duration_ms=total_duration_ms, queue_duration_ms=queue_duration_ms, first_status_duration_ms=first_status_duration_ms, first_text_token_duration_ms=first_text_token_duration_ms, answer_chars=len("".join(answer)))
            await consultation_trace.finish(
                outputs={
                    "status": "interrupted",
                    "partial_answer": "".join(answer),
                    "answer_chars": len("".join(answer)),
                },
                error="ClientDisconnected",
            )
            raise
        except Exception as exc:  # noqa: BLE001 - legacy SSE boundary converts failures to terminal events
            if answer:
                await asyncio.to_thread(
                    _save_assistant, ctx, conversation_id, "".join(answer), "interrupted",
                    agent.langsmith_trace_id,
                )
            total_duration_ms = int((time.perf_counter() - started) * 1000)
            audit("conversation.failed", level=logging.ERROR, request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="failed", duration_ms=total_duration_ms, total_duration_ms=total_duration_ms, queue_duration_ms=queue_duration_ms, first_status_duration_ms=first_status_duration_ms, first_text_token_duration_ms=first_text_token_duration_ms, error_type=type(exc).__name__, error=summary(str(exc)))
            await consultation_trace.finish(
                outputs={
                    "status": "failed",
                    "partial_answer": "".join(answer),
                },
                error=type(exc).__name__,
            )
            yield sse("error", {"message": str(exc)})
        finally:
            if slot_acquired:
                await concurrency.release(identity)
            await concurrency.release_reservation(identity)
            await consultation_trace.finish(outputs={"status": "closed"})

    settings = get_settings()
    return StreamingResponse(
        with_sse_heartbeat(events(), settings.sse_heartbeat_seconds),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
