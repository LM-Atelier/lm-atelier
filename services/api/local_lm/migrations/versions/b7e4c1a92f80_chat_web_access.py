"""Per-chat web access consent

Revision ID: b7e4c1a92f80
Revises: a4d7e2b9c150
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "b7e4c1a92f80"
down_revision: str | None = "a4d7e2b9c150"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # Empty rather than a default that grants anything. Every existing chat
    # starts with no web access, which is what its owner consented to when
    # the capability did not exist.
    op.add_column(
        "chats",
        sa.Column("web_settings_json", sa.JSON(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("chats", "web_settings_json")
