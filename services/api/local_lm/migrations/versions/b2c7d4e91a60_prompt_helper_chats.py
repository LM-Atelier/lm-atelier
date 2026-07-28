"""add isolated prompt helper chats

Revision ID: b2c7d4e91a60
Revises: 8d6f4b2a9c10
Create Date: 2026-07-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "b2c7d4e91a60"
down_revision: str | None = "8d6f4b2a9c10"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "chats",
        sa.Column("scope", sa.String(length=24), nullable=False, server_default="standard"),
    )
    op.add_column(
        "chats",
        sa.Column("draft_prompt", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index("ix_chats_scope", "chats", ["scope"])


def downgrade() -> None:
    op.drop_index("ix_chats_scope", table_name="chats")
    op.drop_column("chats", "draft_prompt")
    op.drop_column("chats", "scope")
