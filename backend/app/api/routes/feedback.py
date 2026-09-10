from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy import select

from backend.app.core.context import get_user_context
from backend.app.core.logging import audit
from backend.app.db.models import Message, MessageFeedback
from backend.app.db.session import SessionLocal, get_db
from backend.app.schemas import MessageFeedbackRequest

router = APIRouter(prefix="/api")


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
