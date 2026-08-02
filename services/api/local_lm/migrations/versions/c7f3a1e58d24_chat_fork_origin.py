"""record where a forked chat came from

Revision ID: c7f3a1e58d24
Revises: a3f61c2d8be7
Create Date: 2026-08-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7f3a1e58d24"
down_revision: str | None = "a3f61c2d8be7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Forking a thread has to be traceable from the fork, and the origin is
    # contract rather than a marker hidden in an unrelated settings blob.
    op.add_column(
        "chats",
        sa.Column("origin_json", sa.JSON(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("chats", "origin_json")
