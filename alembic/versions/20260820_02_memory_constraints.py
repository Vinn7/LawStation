"""Add lookup and idempotency constraints for layered memories."""

import sqlalchemy as sa

from alembic import op

revision = "20260820_02"
down_revision = "20260820_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "user_memories" in tables:
        index_names = {item["name"] for item in inspector.get_indexes("user_memories")}
        unique_names = {
            item["name"] for item in inspector.get_unique_constraints("user_memories")
        }
        if "ix_memory_context" not in index_names:
            op.create_index(
                "ix_memory_context",
                "user_memories",
                ["tenant_id", "user_id", "scope", "status", "conversation_id"],
            )
        if "uq_memory_source_canonical" not in unique_names:
            with op.batch_alter_table("user_memories") as batch:
                batch.create_unique_constraint(
                    "uq_memory_source_canonical",
                    ["tenant_id", "user_id", "source_message_id", "canonical_key"],
                )
    if "conversation_summaries" in tables:
        op.execute(
            "UPDATE conversation_summaries "
            "SET generated_at=COALESCE(generated_at, updated_at) WHERE generated_at IS NULL"
        )


def downgrade() -> None:
    with op.batch_alter_table("user_memories") as batch:
        batch.drop_constraint("uq_memory_source_canonical", type_="unique")
    op.drop_index("ix_memory_context", table_name="user_memories")
