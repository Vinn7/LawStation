"""Requeue memory jobs that stored the sanitized tool-choice compatibility error."""

import sqlalchemy as sa

from alembic import op

revision = "20260901_06"
down_revision = "20260826_05"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Only retry the historical deterministic error fixed by the new request shape."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "memory_jobs" not in set(inspector.get_table_names()):
        return
    bind.execute(
        sa.text(
            "UPDATE memory_jobs "
            "SET status='pending', attempts=0, last_error='' "
            "WHERE status='failed' AND last_error=:safe_error"
        ),
        {"safe_error": "记忆模型调用方式与模型不兼容"},
    )


def downgrade() -> None:
    # 数据修复无法可靠还原哪些任务在升级后已经重新执行，因此保持 no-op。
    pass
