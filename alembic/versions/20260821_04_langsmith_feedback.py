"""Add LangSmith trace correlation and durable message feedback."""

import sqlalchemy as sa

from alembic import op

revision = "20260821_04"
down_revision = "20260821_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "messages" in tables:
        columns = {item["name"] for item in inspector.get_columns("messages")}
        if "langsmith_trace_id" not in columns:
            with op.batch_alter_table("messages") as batch:
                batch.add_column(sa.Column("langsmith_trace_id", sa.String(36), nullable=True))
    if "message_feedback" not in tables:
        op.create_table(
            "message_feedback",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("tenant_id", sa.String(36), nullable=False),
            sa.Column("user_id", sa.String(36), nullable=False),
            sa.Column("message_id", sa.String(36), nullable=False),
            sa.Column("score", sa.Integer(), nullable=False),
            sa.Column("comment", sa.Text(), nullable=False, server_default=""),
            sa.Column("langsmith_trace_id", sa.String(36), nullable=True),
            sa.Column("sync_status", sa.String(20), nullable=False, server_default="pending"),
            sa.Column("last_error", sa.Text(), nullable=False, server_default=""),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
            sa.UniqueConstraint(
                "tenant_id", "user_id", "message_id", name="uq_feedback_owner_message"
            ),
        )
        op.create_index(
            "ix_feedback_owner", "message_feedback", ["tenant_id", "user_id", "created_at"]
        )


def downgrade() -> None:
    op.drop_index("ix_feedback_owner", table_name="message_feedback")
    op.drop_table("message_feedback")
    with op.batch_alter_table("messages") as batch:
        batch.drop_column("langsmith_trace_id")
