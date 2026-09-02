import asyncio
import json
import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

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
from backend.app.db.models import (
    AgentRun,
    Conversation,
    MemoryJob,
    Message,
    MessageFeedback,
    RetrievalTrace,
    ToolCallRecord,
    User,
)
from backend.app.db.session import SessionLocal, get_db
from backend.app.schemas import (
    ChatRequest,
    ConversationCreate,
    MemoryUpdate,
    MemoryVersionRequest,
    MessageFeedbackRequest,
)
from backend.app.services.agent_runs import (
    TERMINAL_STATUSES,
    AgentRunConflict,
    run_payload,
)
from backend.app.services.memory import MemoryService
from backend.app.services.repositories import OwnedRepository
from mcp_servers.law_rag.server import get_index_status

router = APIRouter(prefix="/api")
SCENARIO_CONVERSATION_PREFIX = "[场景] "


@router.get("/index/status")
def index_status():
    return get_index_status()


def _scenario_catalog(request: Request):
    catalog = getattr(request.app.state, "scenario_catalog", None)
    if catalog is None or not catalog.enabled:
        raise HTTPException(404, "场景观察模式未启用")
    return catalog


@router.get("/test-scenarios/datasets")
def scenario_datasets(request: Request, _ctx=Depends(get_user_context)):
    return _scenario_catalog(request).datasets()


@router.get("/test-scenarios/datasets/{dataset_id}/scenarios")
def scenario_summaries(dataset_id: str, request: Request, _ctx=Depends(get_user_context)):
    result = _scenario_catalog(request).scenario_summaries(dataset_id)
    if result is None:
        raise HTTPException(404, "测试数据集不存在")
    return result


@router.get("/test-scenarios/datasets/{dataset_id}/scenarios/{scenario_id}")
def scenario_detail(
    dataset_id: str, scenario_id: str, request: Request, _ctx=Depends(get_user_context)
):
    result = _scenario_catalog(request).scenario(dataset_id, scenario_id)
    if result is None:
        raise HTTPException(404, "测试场景不存在")
    return result


@router.get("/test-scenarios/agent-runs/{run_id}/outcome")
async def scenario_run_outcome(run_id: str, request: Request, ctx=Depends(get_user_context)):
    _scenario_catalog(request)
    result = await request.app.state.agent_runs.scenario_outcome(ctx, run_id)
    if result is None:
        raise HTTPException(404, "任务不存在或无权访问")
    return result


def _delete_scenario_conversation(ctx, conversation_id: str) -> list[str]:
    with SessionLocal() as db:
        conversation = db.scalar(select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == ctx.tenant_id,
            Conversation.user_id == ctx.user_id,
        ))
        if conversation is None:
            raise LookupError("会话不存在或无权访问")
        if not conversation.title.startswith(SCENARIO_CONVERSATION_PREFIX):
            raise PermissionError("只能清理场景观察模式创建的会话")
        active = db.scalar(select(AgentRun.id).where(
            AgentRun.tenant_id == ctx.tenant_id,
            AgentRun.user_id == ctx.user_id,
            AgentRun.conversation_id == conversation_id,
            AgentRun.status.in_(("queued", "running")),
        ).limit(1))
        if active:
            raise RuntimeError("场景会话仍有任务运行，请先停止或等待完成")
        thread_ids = list(db.scalars(select(AgentRun.langgraph_thread_id).where(
            AgentRun.tenant_id == ctx.tenant_id,
            AgentRun.user_id == ctx.user_id,
            AgentRun.conversation_id == conversation_id,
        )))
        owner = (
            MemoryJob.tenant_id == ctx.tenant_id,
            MemoryJob.user_id == ctx.user_id,
            MemoryJob.conversation_id == conversation_id,
        )
        db.execute(delete(MemoryJob).where(*owner))
        db.execute(delete(ToolCallRecord).where(
            ToolCallRecord.tenant_id == ctx.tenant_id,
            ToolCallRecord.user_id == ctx.user_id,
            ToolCallRecord.conversation_id == conversation_id,
        ))
        db.execute(delete(RetrievalTrace).where(
            RetrievalTrace.tenant_id == ctx.tenant_id,
            RetrievalTrace.user_id == ctx.user_id,
            RetrievalTrace.conversation_id == conversation_id,
        ))
        db.delete(conversation)
        db.commit()
        return thread_ids


@router.delete("/test-scenarios/conversations/{conversation_id}", status_code=204)
async def delete_scenario_conversation(
    conversation_id: str, request: Request, ctx=Depends(get_user_context)
):
    _scenario_catalog(request)
    try:
        thread_ids = await asyncio.to_thread(
            _delete_scenario_conversation, ctx, conversation_id
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    for thread_id in thread_ids:
        await request.app.state.agent_runtime.delete_checkpoint_thread(thread_id)


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
    items = repo.messages(conversation_id)
    feedback_by_message = {
        item.message_id: item.score
        for item in db.scalars(select(MessageFeedback).where(
            MessageFeedback.tenant_id == ctx.tenant_id,
            MessageFeedback.user_id == ctx.user_id,
            MessageFeedback.message_id.in_([message.id for message in items]),
        ))
    } if items else {}
    return [{**obj(item), "feedback_score": feedback_by_message.get(item.id)} for item in items]


@router.get("/memories")
def memories(
    conversation_id: str | None = Query(None),
    scope: str | None = Query(None, pattern="^(user|conversation)$"),
    status: str | None = Query(None, pattern="^(pending|active|superseded|rejected|expired)$"),
    memory_type: str | None = Query(None),
    ctx=Depends(get_user_context),
    db=Depends(get_db),
):
    repo = OwnedRepository(db, ctx)
    if conversation_id and not repo.conversation(conversation_id):
        raise HTTPException(404, "会话不存在或无权访问")
    return [obj(item) for item in repo.memories(
        conversation_id, scope=scope, status=status, memory_type=memory_type
    )]


@router.patch("/memories/{memory_id}")
def update_memory(memory_id: str, payload: MemoryUpdate, ctx=Depends(get_user_context), db=Depends(get_db)):
    result = OwnedRepository(db, ctx).update_memory(memory_id, payload.content, payload.version)
    if result == "missing":
        raise HTTPException(404, "记忆不存在或无权访问")
    if result == "conflict":
        raise HTTPException(409, "记忆已被其他操作更新，请刷新后重试")
    db.commit()
    return {"ok": True}


def _set_memory_status(memory_id: str, payload: MemoryVersionRequest, target: str, ctx, db):
    repo = OwnedRepository(db, ctx)
    owned_memory = repo.memory(memory_id)
    result = repo.set_memory_status(memory_id, payload.version, target)
    if result == "missing":
        raise HTTPException(404, "记忆不存在或无权访问")
    if result == "conflict":
        raise HTTPException(409, "记忆已被其他操作更新，请刷新后重试")
    db.commit()
    audit(
        "memory.candidate.confirmed" if target == "active" else "memory.candidate.rejected",
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=owned_memory.conversation_id if owned_memory else None,
        status=target,
        memory_id=memory_id,
    )
    return {"ok": True}


@router.post("/memories/{memory_id}/confirm")
def confirm_memory(memory_id: str, payload: MemoryVersionRequest, ctx=Depends(get_user_context), db=Depends(get_db)):
    return _set_memory_status(memory_id, payload, "active", ctx, db)


@router.post("/memories/{memory_id}/reject")
def reject_memory(memory_id: str, payload: MemoryVersionRequest, ctx=Depends(get_user_context), db=Depends(get_db)):
    return _set_memory_status(memory_id, payload, "rejected", ctx, db)


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


@router.get("/memory-jobs/{job_id}")
def memory_job(job_id: str, ctx=Depends(get_user_context), db=Depends(get_db)):
    job = db.scalar(select(MemoryJob).where(
        MemoryJob.id == job_id,
        MemoryJob.tenant_id == ctx.tenant_id,
        MemoryJob.user_id == ctx.user_id,
    ))
    if not job:
        raise HTTPException(404, "记忆任务不存在或无权访问")
    return obj(job)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def persisted_sse(sequence: int, event: str, data) -> str:
    return (
        f"id: {sequence}\nevent: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


@router.post("/conversations/{conversation_id}/runs", status_code=202)
async def create_agent_run(
    conversation_id: str,
    payload: ChatRequest,
    request: Request,
    ctx=Depends(get_user_context),
):
    """接受一个后台 AgentRun；202 只表示已排队，不表示回答已经生成。"""

    identity = ConcurrencyIdentity(
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=conversation_id,
    )
    try:
        await request.app.state.agent_concurrency.reserve(identity)
        run = await asyncio.to_thread(
            request.app.state.agent_runs.create, ctx, conversation_id, payload.content
        )
    except ConversationBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except LookupError as exc:
        await request.app.state.agent_concurrency.release_reservation(identity)
        raise HTTPException(404, str(exc)) from exc
    except AgentRunConflict as exc:
        await request.app.state.agent_concurrency.release_reservation(identity)
        raise HTTPException(409, str(exc)) from exc
    except Exception:
        await request.app.state.agent_concurrency.release_reservation(identity)
        raise
    return run_payload(run)


@router.get("/agent-runs/{run_id}")
async def get_agent_run(run_id: str, request: Request, ctx=Depends(get_user_context)):
    run = await asyncio.to_thread(request.app.state.agent_runs.owned, ctx, run_id)
    if run is None:
        raise HTTPException(404, "任务不存在或无权访问")
    return run_payload(run)


@router.get("/conversations/{conversation_id}/active-run")
async def active_agent_run(
    conversation_id: str, request: Request, ctx=Depends(get_user_context)
):
    run = await asyncio.to_thread(
        request.app.state.agent_runs.active_for_conversation, ctx, conversation_id
    )
    return run_payload(run) if run else None


@router.post("/agent-runs/{run_id}/cancel")
async def cancel_agent_run(run_id: str, request: Request, ctx=Depends(get_user_context)):
    run = await request.app.state.agent_runs.cancel(ctx, run_id)
    if run is None:
        raise HTTPException(404, "任务不存在或无权访问")
    return run_payload(run)


@router.get("/agent-runs/{run_id}/events")
async def agent_run_events(
    run_id: str,
    request: Request,
    after_sequence: int = Query(0, ge=0),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    ctx=Depends(get_user_context),
):
    """从 sequence 游标重放历史事件，并继续等待该 Run 的新事件。"""

    # Query 游标和标准 Last-Event-ID 二者取较大值，既支持显式重连，也兼容浏览器
    # EventSource 语义；非法游标在开始 StreamingResponse 前即被拒绝。
    try:
        cursor = max(after_sequence, int(last_event_id or 0))
    except ValueError as exc:
        raise HTTPException(400, "Last-Event-ID必须是非负整数") from exc
    manager = request.app.state.agent_runs
    owned = await asyncio.to_thread(manager.owned, ctx, run_id)
    if owned is None:
        raise HTTPException(404, "任务不存在或无权访问")

    async def source():
        nonlocal cursor
        while True:
            run, rows = await asyncio.to_thread(manager.events, ctx, run_id, cursor)
            if run is None:
                return
            for row in rows:
                cursor = row.sequence
                yield persisted_sse(
                    row.sequence, row.event_type, json.loads(row.payload_json)
                )
            # 只有终态且所有持久事件均已发送时才关闭流。浏览器主动断开只结束这个
            # generator，不会取消独立运行的 AgentRun Worker。
            if run.status in TERMINAL_STATUSES and cursor >= run.last_event_seq:
                return
            await manager.wait_for_events(get_settings().sse_heartbeat_seconds)
            if not rows:
                # SSE comment 不属于 AgentRunEvent，不占 sequence，也不会生成空消息。
                yield ": heartbeat\n\n"

    return StreamingResponse(
        source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def with_sse_heartbeat(source, interval_seconds: float):
    """Keep an SSE connection alive without turning heartbeats into app events."""
    iterator = source.__aiter__()
    pending: asyncio.Task | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(iterator))
            done, _ = await asyncio.wait(
                {pending}, timeout=max(0.01, interval_seconds)
            )
            if not done:
                yield ": heartbeat\n\n"
                continue
            try:
                item = pending.result()
            except StopAsyncIteration:
                break
            pending = None
            yield item
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


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


async def _sync_message_feedback(app, feedback_id: str) -> None:
    with SessionLocal() as db:
        feedback = db.get(MessageFeedback, feedback_id)
        if not feedback:
            return
        trace_id = feedback.langsmith_trace_id or ""
        score = feedback.score
        comment = feedback.comment
    synced = await app.state.langsmith_observability.create_user_feedback(
        trace_id=trace_id, score=score, comment=comment
    )
    with SessionLocal() as db:
        feedback = db.get(MessageFeedback, feedback_id)
        if feedback:
            feedback.sync_status = "synced" if synced else "unavailable"
            feedback.last_error = "" if synced else "LangSmith 当前不可用或该消息没有 trace"
            db.commit()


@router.post("/messages/{message_id}/feedback")
async def message_feedback(
    message_id: str,
    payload: MessageFeedbackRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    ctx=Depends(get_user_context),
    db=Depends(get_db),
):
    message = db.scalar(select(Message).where(
        Message.id == message_id,
        Message.tenant_id == ctx.tenant_id,
        Message.user_id == ctx.user_id,
        Message.role == "assistant",
    ))
    if not message:
        raise HTTPException(404, "消息不存在或无权访问")
    feedback = db.scalar(select(MessageFeedback).where(
        MessageFeedback.tenant_id == ctx.tenant_id,
        MessageFeedback.user_id == ctx.user_id,
        MessageFeedback.message_id == message_id,
    ))
    if feedback is None:
        feedback = MessageFeedback(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            message_id=message_id,
        )
        db.add(feedback)
    feedback.score = payload.score
    feedback.comment = payload.comment.strip()
    feedback.langsmith_trace_id = message.langsmith_trace_id
    feedback.sync_status = "pending"
    feedback.last_error = ""
    db.commit()
    db.refresh(feedback)
    audit(
        "message.feedback.saved",
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=message.conversation_id,
        message_id=message.id,
        score=payload.score,
        status="saved",
    )
    background_tasks.add_task(_sync_message_feedback, request.app, feedback.id)
    return {"ok": True, "sync_status": feedback.sync_status}


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
