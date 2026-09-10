from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from backend.app.core.context import get_user_context
from backend.app.core.logging import audit
from backend.app.db.models import MemoryJob
from backend.app.db.session import get_db
from backend.app.schemas import MemoryUpdate, MemoryVersionRequest
from backend.app.services.repositories import OwnedRepository

from .helpers import obj

router = APIRouter(prefix="/api")


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
