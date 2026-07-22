"""persist the active chat branch head

Revision ID: 9c4d2a7e1f30
Revises: 71c8f76c3182
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9c4d2a7e1f30"
down_revision: str | None = "71c8f76c3182"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("chats") as batch_op:
        batch_op.add_column(sa.Column("active_head_message_id", sa.String(length=40)))
    op.execute(
        """
        UPDATE chats
        SET active_head_message_id = (
            SELECT runs.assistant_message_id
            FROM runs
            WHERE runs.chat_id = chats.id
            ORDER BY COALESCE(runs.completed_at, runs.updated_at, runs.created_at) DESC,
                     runs.id DESC
            LIMIT 1
        )
        """
    )
    op.execute(
        """
        UPDATE chats
        SET active_head_message_id = (
            SELECT messages.id
            FROM messages
            WHERE messages.chat_id = chats.id
            ORDER BY messages.updated_at DESC, messages.created_at DESC, messages.id DESC
            LIMIT 1
        )
        WHERE active_head_message_id IS NULL
        """
    )


def downgrade() -> None:
    with op.batch_alter_table("chats") as batch_op:
        batch_op.drop_column("active_head_message_id")
