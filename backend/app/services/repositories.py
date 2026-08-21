from datetime import UTC, datetime

from sqlalchemy import delete, or_, select, update
from sqlalchemy.orm import Session

from backend.app.core.context import RequestUserContext
from backend.app.db.models import Conversation, MemoryRevision, Message, UserMemory


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

    def memories(
        self,
        conversation_id: str | None = None,
        *,
        scope: str | None = None,
        status: str | None = "active",
        memory_type: str | None = None,
        limit: int | None = None,
    ) -> list[UserMemory]:
        query = select(UserMemory).where(
            UserMemory.tenant_id == self.ctx.tenant_id,
            UserMemory.user_id == self.ctx.user_id,
        )
        if conversation_id:
            query = query.where(UserMemory.conversation_id == conversation_id)
        if scope:
            query = query.where(UserMemory.scope == scope)
        if status:
            query = query.where(UserMemory.status == status)
        if memory_type:
            query = query.where(UserMemory.memory_type == memory_type)
        query = query.order_by(UserMemory.importance.desc(), UserMemory.updated_at.desc())
        if limit:
            query = query.limit(limit)
        return list(self.db.scalars(query))

    def context_memories(self, conversation_id: str) -> tuple[list[UserMemory], list[UserMemory]]:
        current_time = datetime.now(UTC)
        valid = or_(UserMemory.expires_at.is_(None), UserMemory.expires_at > current_time)
        case_query = (
            select(UserMemory)
            .where(
                UserMemory.tenant_id == self.ctx.tenant_id,
                UserMemory.user_id == self.ctx.user_id,
                UserMemory.conversation_id == conversation_id,
                UserMemory.scope == "conversation",
                UserMemory.status == "active",
                valid,
            )
            .order_by(UserMemory.importance.desc(), UserMemory.updated_at.desc())
        )
        profile_query = (
            select(UserMemory)
            .where(
                UserMemory.tenant_id == self.ctx.tenant_id,
                UserMemory.user_id == self.ctx.user_id,
                UserMemory.scope == "user",
                UserMemory.status == "active",
                UserMemory.memory_type.in_(("profile_preference", "identity_background")),
                valid,
            )
            .order_by(UserMemory.importance.desc(), UserMemory.updated_at.desc())
        )
        return list(self.db.scalars(case_query)), list(self.db.scalars(profile_query))

    def memory(self, memory_id: str) -> UserMemory | None:
        return self.db.scalar(
            select(UserMemory).where(
                UserMemory.id == memory_id,
                UserMemory.tenant_id == self.ctx.tenant_id,
                UserMemory.user_id == self.ctx.user_id,
            )
        )

    def update_memory(self, memory_id: str, content: str, version: int) -> str:
        memory = self.memory(memory_id)
        if memory is None:
            return "missing"
        if memory.version != version:
            return "conflict"
        self.db.add(self._revision(memory, "update", content, memory.status))
        result = self.db.execute(
            update(UserMemory)
            .where(
                UserMemory.id == memory_id,
                UserMemory.tenant_id == self.ctx.tenant_id,
                UserMemory.user_id == self.ctx.user_id,
                UserMemory.version == version,
            )
            .values(content=content, version=version + 1)
        )
        return "updated" if result.rowcount else "conflict"

    def set_memory_status(self, memory_id: str, version: int, target: str) -> str:
        memory = self.memory(memory_id)
        if memory is None:
            return "missing"
        if memory.version != version:
            return "conflict"
        if target == "active":
            conflicts = list(self.db.scalars(select(UserMemory).where(
                UserMemory.tenant_id == self.ctx.tenant_id,
                UserMemory.user_id == self.ctx.user_id,
                UserMemory.id != memory.id,
                UserMemory.scope == memory.scope,
                UserMemory.canonical_key == memory.canonical_key,
                UserMemory.status == "active",
            )))
            if memory.scope == "conversation":
                conflicts = [
                    conflict for conflict in conflicts
                    if conflict.conversation_id == memory.conversation_id
                ]
            for conflict in conflicts:
                conflict.status = "superseded"
                conflict.active = False
                conflict.superseded_by_id = memory.id
                conflict.version += 1
                self.db.add(self._revision(conflict, "supersede", conflict.content, "superseded"))
        self.db.add(self._revision(memory, "confirm" if target == "active" else "reject", memory.content, target))
        memory.status = target
        memory.active = target == "active"
        memory.confirmed_at = datetime.now(UTC) if target == "active" else None
        memory.version += 1
        return "updated"

    def _revision(
        self, memory: UserMemory, action: str, new_content: str, new_status: str
    ) -> MemoryRevision:
        return MemoryRevision(
            memory_id=memory.id,
            tenant_id=self.ctx.tenant_id,
            user_id=self.ctx.user_id,
            conversation_id=memory.conversation_id,
            action=action,
            previous_content=memory.content,
            new_content=new_content,
            previous_status=memory.status,
            new_status=new_status,
        )

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
