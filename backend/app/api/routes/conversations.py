from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.context import RequestUserContext, get_user_context
from backend.app.db.models import Conversation, MessageFeedback, User
from backend.app.db.session import get_db
from backend.app.schemas import ConversationCreate
from backend.app.services.repositories import OwnedRepository

from .helpers import obj

router = APIRouter(prefix="/api")


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
