"""pin chats and projects

A pin says "this is where I work"; archiving says "I am done with this".
They are independent, so a pin is its own column rather than a state that
archiving would have to take away.

Revision ID: a4d7e2b9c150
Revises: f1c8a2d47b90
Create Date: 2026-08-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4d7e2b9c150"
down_revision: str | None = "f1c8a2d47b90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for table in ("projects", "chats"):
        op.add_column(
            table,
            sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
        op.create_index(f"ix_{table}_pinned", table, ["pinned"])


def downgrade() -> None:
    for table in ("chats", "projects"):
        op.drop_index(f"ix_{table}_pinned", table_name=table)
        op.drop_column(table, "pinned")
