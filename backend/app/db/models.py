from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.session import Base


def uid() -> str:
    return str(uuid4())


def now() -> datetime:
    return datetime.now(UTC)


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(100))


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    tenant_id: Mapped[str] = mapped_column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(default=now)


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id", "id"),
        ForeignKeyConstraint(["tenant_id", "user_id"], ["users.tenant_id", "users.id"], ondelete="CASCADE"),
        Index("ix_conversation_owner", "tenant_id", "user_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    tenant_id: Mapped[str] = mapped_column(String(36))
    user_id: Mapped[str] = mapped_column(String(36))
    title: Mapped[str] = mapped_column(String(200), default="新对话")
    created_at: Mapped[datetime] = mapped_column(default=now)
    updated_at: Mapped[datetime] = mapped_column(default=now, onupdate=now)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "user_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.user_id", "conversations.id"],
            ondelete="CASCADE",
        ),
        Index("ix_message_owner_conversation", "tenant_id", "user_id", "conversation_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    tenant_id: Mapped[str] = mapped_column(String(36))
    user_id: Mapped[str] = mapped_column(String(36))
    conversation_id: Mapped[str] = mapped_column(String(36))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="complete")
    created_at: Mapped[datetime] = mapped_column(default=now)


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "user_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.user_id", "conversations.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("tenant_id", "user_id", "conversation_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    tenant_id: Mapped[str] = mapped_column(String(36))
    user_id: Mapped[str] = mapped_column(String(36))
    conversation_id: Mapped[str] = mapped_column(String(36))
    content: Mapped[str] = mapped_column(Text)
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    covered_until_message_id: Mapped[str] = mapped_column(String(36))
    version: Mapped[int] = mapped_column(Integer, default=1)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    generated_at: Mapped[datetime] = mapped_column(default=now)
    updated_at: Mapped[datetime] = mapped_column(default=now, onupdate=now)


class UserMemory(Base):
    __tablename__ = "user_memories"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "user_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.user_id", "conversations.id"],
            ondelete="CASCADE",
        ),
        Index("ix_memory_owner", "tenant_id", "user_id", "created_at"),
        Index("ix_memory_context", "tenant_id", "user_id", "scope", "status", "conversation_id"),
        UniqueConstraint(
            "tenant_id", "user_id", "source_message_id", "canonical_key",
            name="uq_memory_source_canonical",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    tenant_id: Mapped[str] = mapped_column(String(36))
    user_id: Mapped[str] = mapped_column(String(36))
    conversation_id: Mapped[str] = mapped_column(String(36))
    memory_type: Mapped[str] = mapped_column(String(30), default="case_fact")
    scope: Mapped[str] = mapped_column(String(20), default="conversation")
    status: Mapped[str] = mapped_column(String(20), default="pending")
    canonical_key: Mapped[str] = mapped_column(String(160), default="")
    content: Mapped[str] = mapped_column(Text)
    source_message_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    source_excerpt: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    importance: Mapped[int] = mapped_column(Integer, default=50)
    superseded_by_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("user_memories.id", ondelete="SET NULL"), nullable=True
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(default=now)
    updated_at: Mapped[datetime] = mapped_column(default=now, onupdate=now)


class MemoryRevision(Base):
    __tablename__ = "memory_revisions"
    __table_args__ = (Index("ix_memory_revision_owner", "tenant_id", "user_id", "memory_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("user_memories.id", ondelete="CASCADE")
    )
    tenant_id: Mapped[str] = mapped_column(String(36))
    user_id: Mapped[str] = mapped_column(String(36))
    conversation_id: Mapped[str] = mapped_column(String(36))
    action: Mapped[str] = mapped_column(String(30))
    previous_content: Mapped[str] = mapped_column(Text, default="")
    new_content: Mapped[str] = mapped_column(Text, default="")
    previous_status: Mapped[str] = mapped_column(String(20), default="")
    new_status: Mapped[str] = mapped_column(String(20), default="")
    created_at: Mapped[datetime] = mapped_column(default=now)


class MemoryJob(Base):
    __tablename__ = "memory_jobs"
    __table_args__ = (
        Index("ix_memory_job_status", "status", "created_at"),
        UniqueConstraint(
            "tenant_id", "user_id", "source_message_id", name="uq_memory_job_source"
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    tenant_id: Mapped[str] = mapped_column(String(36))
    user_id: Mapped[str] = mapped_column(String(36))
    conversation_id: Mapped[str] = mapped_column(String(36))
    source_message_id: Mapped[str] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    summary_updated: Mapped[bool] = mapped_column(Boolean, default=False)
    last_error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=now)
    updated_at: Mapped[datetime] = mapped_column(default=now, onupdate=now)


class ToolCallRecord(Base):
    __tablename__ = "tool_call_records"
    __table_args__ = (Index("ix_tool_owner", "tenant_id", "user_id", "conversation_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    tenant_id: Mapped[str] = mapped_column(String(36))
    user_id: Mapped[str] = mapped_column(String(36))
    conversation_id: Mapped[str] = mapped_column(String(36))
    tool_name: Mapped[str] = mapped_column(String(100))
    arguments_json: Mapped[str] = mapped_column(Text)
    result_summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20))
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=now)


class RetrievalTrace(Base):
    __tablename__ = "retrieval_traces"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    query: Mapped[str] = mapped_column(Text)
    results_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=now)


class IndexManifest(Base):
    __tablename__ = "index_manifests"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    data_version: Mapped[str] = mapped_column(String(100), unique=True)
    embedding_model: Mapped[str] = mapped_column(String(100))
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    document_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(default=now)
