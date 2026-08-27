"""Add durable agent runs and replayable event log."""

import sqlalchemy as sa

from alembic import op

revision = "20260826_05"
down_revision = "20260821_04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("request_id", sa.String(36), nullable=False, unique=True),
        sa.Column("tenant_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("conversation_id", sa.String(36), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("current_stage", sa.String(30), nullable=False, server_default="queued"),
        sa.Column("user_message_id", sa.String(36), nullable=True),
        sa.Column("assistant_message_id", sa.String(36), nullable=True),
        sa.Column("langgraph_thread_id", sa.String(100), nullable=False, unique=True),
        sa.Column("latest_checkpoint_id", sa.String(100), nullable=False, server_default=""),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("lease_owner", sa.String(100), nullable=False, server_default=""),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_event_seq", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("model_call_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_call_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_type", sa.String(100), nullable=False, server_default=""),
        sa.Column("error_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["tenant_id", "user_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.user_id", "conversations.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["user_message_id"], ["messages.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["assistant_message_id"], ["messages.id"], ondelete="SET NULL"),
    )
    op.create_index(
        "ix_agent_run_owner",
        "agent_runs",
        ["tenant_id", "user_id", "conversation_id", "created_at"],
    )
    op.create_index("ix_agent_run_queue", "agent_runs", ["status", "created_at"])
    op.create_index(
        "uq_agent_run_active_conversation",
        "agent_runs",
        ["tenant_id", "user_id", "conversation_id"],
        unique=True,
        sqlite_where=sa.text("status IN ('queued', 'running')"),
    )
    op.create_table(
        "agent_run_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("tenant_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("conversation_id", sa.String(36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(40), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "user_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.user_id", "conversations.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("run_id", "sequence", name="uq_agent_run_event_sequence"),
    )
    op.create_index(
        "ix_agent_run_event_owner",
        "agent_run_events",
        ["tenant_id", "user_id", "run_id", "sequence"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_run_event_owner", table_name="agent_run_events")
    op.drop_table("agent_run_events")
    op.drop_index("uq_agent_run_active_conversation", table_name="agent_runs")
    op.drop_index("ix_agent_run_queue", table_name="agent_runs")
    op.drop_index("ix_agent_run_owner", table_name="agent_runs")
    op.drop_table("agent_runs")
