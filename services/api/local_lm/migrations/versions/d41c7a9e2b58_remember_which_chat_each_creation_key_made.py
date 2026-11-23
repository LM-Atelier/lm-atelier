"""Remember which chat each creation key made

Revision ID: d41c7a9e2b58
Revises: b8d24f6a9c13
Create Date: 2026-11-23

A create request may carry a key of its own, so that a create retried after
its answer was lost returns the chat the first one made instead of a second
one. Each key names the chat it made, with no foreign key: a key is neither
work nor part of the conversation, and it outlives a chat deleted for good.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d41c7a9e2b58"
down_revision: str | None = "b8d24f6a9c13"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_creation_keys",
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("chat_id", sa.String(40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(idempotency_key) BETWEEN 1 AND 200 AND instr(idempotency_key, char(0)) = 0",
            name="ck_chat_creation_key",
        ),
        sa.PrimaryKeyConstraint("idempotency_key"),
    )


def downgrade() -> None:
    op.drop_table("chat_creation_keys")
