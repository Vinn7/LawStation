from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from backend.app.core.context import RequestUserContext
from backend.app.db.models import Conversation, Message, UserMemory


class OwnedRepository:
    def __init__(self, db: Session, ctx: RequestUserContext):
        self.db, self.ctx = db, ctx

    def conversation(self, conversation_id: str) -> Conversation | None:
        return self.db.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == self.ctx.tenant_id,
                Conversation.user_id == self.ctx.user_id,
            )
        )

    def conversations(self) -> list[Conversation]:
        return list(
            self.db.scalars(
                select(Conversation)
                .where(
                    Conversation.tenant_id == self.ctx.tenant_id,
                    Conversation.user_id == self.ctx.user_id,
                )
                .order_by(Conversation.updated_at.desc())
            )
        )

    def messages(self, conversation_id: str, limit: int | None = None) -> list[Message]:
        query = (
            select(Message)
            .where(
                Message.tenant_id == self.ctx.tenant_id,
                Message.user_id == self.ctx.user_id,
                Message.conversation_id == conversation_id,
            )
            .order_by(Message.created_at.desc())
        )
        if limit:
            query = query.limit(limit)
        return list(reversed(list(self.db.scalars(query))))

    def memories(self, conversation_id: str | None = None) -> list[UserMemory]:
        query = select(UserMemory).where(
            UserMemory.tenant_id == self.ctx.tenant_id,
            UserMemory.user_id == self.ctx.user_id,
            UserMemory.active.is_(True),
        )
        if conversation_id:
            query = query.where(UserMemory.conversation_id == conversation_id)
        return list(self.db.scalars(query.order_by(UserMemory.updated_at.desc())))

    def update_memory(self, memory_id: str, content: str) -> bool:
        result = self.db.execute(
            update(UserMemory)
            .where(
                UserMemory.id == memory_id,
                UserMemory.tenant_id == self.ctx.tenant_id,
                UserMemory.user_id == self.ctx.user_id,
            )
            .values(content=content)
        )
        return bool(result.rowcount)

    def delete_memory(self, memory_id: str) -> bool:
        result = self.db.execute(
            delete(UserMemory).where(
                UserMemory.id == memory_id,
                UserMemory.tenant_id == self.ctx.tenant_id,
                UserMemory.user_id == self.ctx.user_id,
            )
        )
        return bool(result.rowcount)

    def clear_memories(self, conversation_id: str | None = None) -> int:
        query = delete(UserMemory).where(
            UserMemory.tenant_id == self.ctx.tenant_id,
            UserMemory.user_id == self.ctx.user_id,
        )
        if conversation_id:
            query = query.where(UserMemory.conversation_id == conversation_id)
        result = self.db.execute(query)
        return result.rowcount or 0

