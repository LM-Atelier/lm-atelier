"""Persist explicit dispatch positions and exact relative-move receipts."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7a9d3c14620"
down_revision: str | None = "a64c0d289e71"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "queue_order_entries",
        sa.Column("lane", sa.String(16), primary_key=True),
        sa.Column("owner_type", sa.String(16), primary_key=True),
        sa.Column("owner_id", sa.String(40), primary_key=True),
        sa.Column("queue_group", sa.String(100), nullable=False),
        sa.Column("queue_resource", sa.String(100), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.CheckConstraint("position >= 0", name="ck_queue_order_position"),
    )
    op.create_table(
        "queue_order_receipts",
        sa.Column(
            "lane",
            sa.String(16),
            sa.ForeignKey("generation_queue_policies.lane", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("command_key", sa.String(128), primary_key=True),
        sa.Column("command_json", sa.JSON(), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    queries = (
        "SELECT 1 FROM queue_order_entries LIMIT 1",
        "SELECT 1 FROM queue_order_receipts LIMIT 1",
    )
    if any(op.get_bind().execute(sa.text(query)).first() for query in queries):
        raise RuntimeError("Manual queue ordering requires the current database version.")
    op.drop_table("queue_order_receipts")
    op.drop_table("queue_order_entries")
