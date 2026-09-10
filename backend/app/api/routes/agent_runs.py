import asyncio
import json

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.app.agent.concurrency import ConcurrencyIdentity, ConversationBusyError
from backend.app.core.config import get_settings
from backend.app.core.context import get_user_context
from backend.app.schemas import ChatRequest
from backend.app.services.agent_runs import TERMINAL_STATUSES, AgentRunConflict, run_payload

from .helpers import persisted_sse

router = APIRouter(prefix="/api")


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
