from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.context import RequestUserContext
from backend.app.db.models import ConversationSummary, Message, UserMemory
from backend.app.services.repositories import OwnedRepository


class MemoryService:
    def __init__(self, db: Session, ctx: RequestUserContext):
        self.db, self.ctx = db, ctx
        self.repo = OwnedRepository(db, ctx)
        self.settings = get_settings()

    def context(self, conversation_id: str) -> tuple[str, list[Message]]:
        summary = self.db.scalar(
            select(ConversationSummary).where(
                ConversationSummary.tenant_id == self.ctx.tenant_id,
                ConversationSummary.user_id == self.ctx.user_id,
                ConversationSummary.conversation_id == conversation_id,
            )
        )
        memories = self.repo.memories()[:20]
        blocks = []
        if memories:
            blocks.append("用户已确认/沉淀记忆：\n" + "\n".join(f"- {m.content}" for m in memories))
        if summary:
            blocks.append("较早对话摘要：\n" + summary.content)
        return "\n\n".join(blocks), self.repo.messages(
            conversation_id, self.settings.memory_recent_message_count
        )

    def consolidate(self, conversation_id: str) -> bool:
        messages = self.repo.messages(conversation_id)
        approx_tokens = sum(len(m.content) for m in messages) // 2
        if approx_tokens < self.settings.memory_compression_threshold:
            return False
        keep = self.settings.memory_recent_message_count
        old = messages[:-keep] if len(messages) > keep else []
        if not old:
            return False
        digest = "\n".join(f"{m.role}: {m.content[:600]}" for m in old)[-12000:]
        summary = self.db.scalar(
            select(ConversationSummary).where(
                ConversationSummary.tenant_id == self.ctx.tenant_id,
                ConversationSummary.user_id == self.ctx.user_id,
                ConversationSummary.conversation_id == conversation_id,
            )
        )
        if summary:
            summary.content = digest
            summary.covered_until_message_id = old[-1].id
            summary.version += 1
        else:
            self.db.add(
                ConversationSummary(
                    tenant_id=self.ctx.tenant_id,
                    user_id=self.ctx.user_id,
                    conversation_id=conversation_id,
                    content=digest,
                    covered_until_message_id=old[-1].id,
                )
            )
        # MVP: persist substantive user statements as durable, owner-scoped facts.
        for message in old:
            if message.role != "user" or len(message.content.strip()) < 20:
                continue
            exists = self.db.scalar(
                select(UserMemory.id).where(
                    UserMemory.tenant_id == self.ctx.tenant_id,
                    UserMemory.user_id == self.ctx.user_id,
                    UserMemory.source_message_id == message.id,
                )
            )
            if not exists:
                self.db.add(
                    UserMemory(
                        tenant_id=self.ctx.tenant_id,
                        user_id=self.ctx.user_id,
                        conversation_id=conversation_id,
                        content=message.content[:2000],
                        source_message_id=message.id,
                    )
                )
        self.db.commit()
        return True

