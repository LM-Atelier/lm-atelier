"""Keep generation dispatch state and exact command retries.

Revision ID: d9b63c17a204
Revises: e7c2a91d6b40
Create Date: 2026-09-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d9b63c17a204"
down_revision: str | None = "e7c2a91d6b40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "generation_queue_policies",
        sa.Column("lane", sa.String(16), nullable=False),
        sa.Column("dispatch_state", sa.String(16), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("lane"),
    )
    op.create_table(
        "generation_queue_receipts",
        sa.Column("lane", sa.String(16), nullable=False),
        sa.Column("command_key", sa.String(128), nullable=False),
        sa.Column("action", sa.String(24), nullable=False),
        sa.Column("expected_revision", sa.Integer(), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["lane"], ["generation_queue_policies.lane"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("lane", "command_key"),
    )


def downgrade() -> None:
    op.drop_table("generation_queue_receipts")
    op.drop_table("generation_queue_policies")
