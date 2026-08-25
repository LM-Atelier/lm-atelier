"""add durable selective chat-item removal receipts

Revision ID: e7b9c4d12f60
Revises: d6a8f1c42b90
Create Date: 2026-08-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e7b9c4d12f60"
down_revision: str | None = "d6a8f1c42b90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _lowercase_sha256_check(column: str) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = 64 AND lower({column}) = {column} AND {remainder} = ''"


def upgrade() -> None:
    op.create_table(
        "chat_item_removal_receipts",
        sa.Column("id", sa.String(length=48), nullable=False),
        sa.Column("chat_id", sa.String(length=40), nullable=False),
        sa.Column("operation_key", sa.String(length=128), nullable=False),
        sa.Column("message_id", sa.String(length=40), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("message_revision_id", sa.String(length=64), nullable=False),
        sa.Column("content_removed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(operation_key) BETWEEN 1 AND 128 "
            "AND operation_key GLOB '[A-Za-z0-9]*' "
            "AND operation_key NOT GLOB '*[^A-Za-z0-9_.:-]*'",
            name="ck_chat_item_removal_receipt_operation_key",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("request_sha256"),
            name="ck_chat_item_removal_receipt_request_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("message_revision_id"),
            name="ck_chat_item_removal_receipt_revision_id",
        ),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "chat_id",
            "operation_key",
            name="uq_chat_item_removal_receipt_chat_operation",
        ),
    )
    op.create_index(
        "ix_chat_item_removal_receipts_chat_id",
        "chat_item_removal_receipts",
        ["chat_id"],
        unique=False,
    )
    op.create_index(
        "ix_chat_item_removal_receipts_message_id",
        "chat_item_removal_receipts",
        ["message_id"],
        unique=False,
    )


def downgrade() -> None:
    receipt_count = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM chat_item_removal_receipts"))
        .scalar_one()
    )
    if receipt_count:
        raise RuntimeError(
            "Cannot downgrade selective chat-item removal: durable replay receipts exist."
        )
    op.drop_index(
        "ix_chat_item_removal_receipts_message_id",
        table_name="chat_item_removal_receipts",
    )
    op.drop_index(
        "ix_chat_item_removal_receipts_chat_id",
        table_name="chat_item_removal_receipts",
    )
    op.drop_table("chat_item_removal_receipts")
