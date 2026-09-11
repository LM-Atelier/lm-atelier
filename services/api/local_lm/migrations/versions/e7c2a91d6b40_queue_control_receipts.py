"""Keep queue control transitions and their exact retry responses.

Revision ID: e7c2a91d6b40
Revises: e4b7c2d91a60
Create Date: 2026-09-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e7c2a91d6b40"
down_revision: str | None = "e4b7c2d91a60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "work_plan_controls",
        sa.Column("plan_id", sa.String(40), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("eligible_since", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["plan_id"], ["work_plans.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("plan_id"),
    )
    op.create_table(
        "work_plan_control_receipts",
        sa.Column("plan_id", sa.String(40), nullable=False),
        sa.Column("command_key", sa.String(128), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("expected_revision", sa.Integer(), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["plan_id"], ["work_plans.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("plan_id", "command_key"),
    )


def downgrade() -> None:
    op.drop_table("work_plan_control_receipts")
    op.drop_table("work_plan_controls")
