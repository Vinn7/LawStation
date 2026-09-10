import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import delete, select

from backend.app.core.context import get_user_context
from backend.app.db.models import (
    AgentRun,
    Conversation,
    MemoryJob,
    RetrievalTrace,
    ToolCallRecord,
)
from backend.app.db.session import SessionLocal

router = APIRouter(prefix="/api")
SCENARIO_CONVERSATION_PREFIX = "[场景] "


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
