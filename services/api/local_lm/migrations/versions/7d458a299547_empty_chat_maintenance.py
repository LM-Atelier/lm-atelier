"""Record empty-chat cleanup previews and completed deletions.

Revision ID: 7d458a299547
Revises: a8e3f7c24b60
Create Date: 2026-09-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7d458a299547"
down_revision: str | None = "a8e3f7c24b60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Two new tables and nothing else: no existing row or column is touched, so
    # the upgrade cannot lose data and the downgrade only removes what this adds.
    op.create_table(
        "empty_chat_previews",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("digest", sa.String(length=64), nullable=False),
        sa.Column("chat_states_json", sa.JSON(), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_empty_chat_previews_digest"), "empty_chat_previews", ["digest"], unique=False
    )
    op.create_table(
        "empty_chat_deletions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("operation_id", sa.String(length=80), nullable=False),
        sa.Column("digest", sa.String(length=64), nullable=False),
        sa.Column("deleted_ids_json", sa.JSON(), nullable=False),
        sa.Column("deleted_count", sa.Integer(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_empty_chat_deletions_operation_id"),
        "empty_chat_deletions",
        ["operation_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_empty_chat_deletions_operation_id"), table_name="empty_chat_deletions")
    op.drop_table("empty_chat_deletions")
    op.drop_index(op.f("ix_empty_chat_previews_digest"), table_name="empty_chat_previews")
    op.drop_table("empty_chat_previews")
