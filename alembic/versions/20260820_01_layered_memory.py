"""Add layered, confirmable memory records and durable processing jobs."""

import sqlalchemy as sa

from alembic import op

revision = "20260820_01"
down_revision = None
branch_labels = None
depends_on = None


def _columns(inspector, table: str) -> set[str]:
    return {column["name"] for column in inspector.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "conversation_summaries" in tables:
        columns = _columns(inspector, "conversation_summaries")
        with op.batch_alter_table("conversation_summaries") as batch:
            if "summary_json" not in columns:
                batch.add_column(sa.Column("summary_json", sa.Text(), nullable=False, server_default="{}"))
            if "token_count" not in columns:
                batch.add_column(sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"))
            if "generated_at" not in columns:
                batch.add_column(sa.Column("generated_at", sa.DateTime(), nullable=True))

    if "user_memories" in tables:
        columns = _columns(inspector, "user_memories")
        additions = [
            ("scope", sa.Column("scope", sa.String(20), nullable=False, server_default="conversation")),
            ("status", sa.Column("status", sa.String(20), nullable=False, server_default="pending")),
            ("canonical_key", sa.Column("canonical_key", sa.String(160), nullable=False, server_default="")),
            ("source_excerpt", sa.Column("source_excerpt", sa.Text(), nullable=False, server_default="")),
            ("confidence", sa.Column("confidence", sa.Float(), nullable=False, server_default="0")),
            ("importance", sa.Column("importance", sa.Integer(), nullable=False, server_default="50")),
            ("superseded_by_id", sa.Column("superseded_by_id", sa.String(36), nullable=True)),
            ("confirmed_at", sa.Column("confirmed_at", sa.DateTime(), nullable=True)),
            ("expires_at", sa.Column("expires_at", sa.DateTime(), nullable=True)),
            ("version", sa.Column("version", sa.Integer(), nullable=False, server_default="1")),
        ]
        with op.batch_alter_table("user_memories") as batch:
            for name, column in additions:
                if name not in columns:
                    batch.add_column(column)
        op.execute("UPDATE user_memories SET scope='conversation', status='pending', active=0")

    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "memory_revisions" not in tables:
        op.create_table(
            "memory_revisions",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("memory_id", sa.String(36), sa.ForeignKey("user_memories.id", ondelete="CASCADE"), nullable=False),
            sa.Column("tenant_id", sa.String(36), nullable=False),
            sa.Column("user_id", sa.String(36), nullable=False),
            sa.Column("conversation_id", sa.String(36), nullable=False),
            sa.Column("action", sa.String(30), nullable=False),
            sa.Column("previous_content", sa.Text(), nullable=False, server_default=""),
            sa.Column("new_content", sa.Text(), nullable=False, server_default=""),
            sa.Column("previous_status", sa.String(20), nullable=False, server_default=""),
            sa.Column("new_status", sa.String(20), nullable=False, server_default=""),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_memory_revision_owner", "memory_revisions", ["tenant_id", "user_id", "memory_id"])
    if "memory_jobs" not in tables:
        op.create_table(
            "memory_jobs",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("tenant_id", sa.String(36), nullable=False),
            sa.Column("user_id", sa.String(36), nullable=False),
            sa.Column("conversation_id", sa.String(36), nullable=False),
            sa.Column("source_message_id", sa.String(36), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("candidate_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("summary_updated", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("last_error", sa.Text(), nullable=False, server_default=""),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("tenant_id", "user_id", "source_message_id", name="uq_memory_job_source"),
        )
        op.create_index("ix_memory_job_status", "memory_jobs", ["status", "created_at"])


def downgrade() -> None:
    op.drop_table("memory_jobs")
    op.drop_table("memory_revisions")
