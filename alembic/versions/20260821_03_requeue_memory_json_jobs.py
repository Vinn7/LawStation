"""Requeue memory jobs rejected by DeepSeek thinking mode tool choice."""

import sqlalchemy as sa

from alembic import op

revision = "20260821_03"
down_revision = "20260820_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "memory_jobs" not in set(inspector.get_table_names()):
        return
    bind.execute(
        sa.text(
            "UPDATE memory_jobs "
            "SET status='pending', attempts=0, last_error='' "
            "WHERE status='failed' "
            "AND last_error LIKE :error_pattern"
        ),
        {"error_pattern": "%Thinking mode does not support this tool_choice%"},
    )


def downgrade() -> None:
    # This migration repairs task state; recreating a historical failure is not useful.
    pass
