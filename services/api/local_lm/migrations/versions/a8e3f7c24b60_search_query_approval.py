"""Persist exact search approval separately from execution provenance.

Revision ID: a8e3f7c24b60
Revises: d9b63c17a204
Create Date: 2026-09-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a8e3f7c24b60"
down_revision: str | None = "d9b63c17a204"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "web_search_proposals",
        sa.Column("id", sa.String(40), nullable=False),
        sa.Column("job_id", sa.String(40), nullable=False),
        sa.Column("run_id", sa.String(40), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("provider_endpoint", sa.Text(), nullable=False),
        sa.Column("provider_revision", sa.String(80), nullable=False),
        sa.Column("approved_automatically", sa.Boolean(), nullable=False),
        sa.Column("dispatch_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dispatch_owner", sa.String(80), nullable=True),
        sa.Column("dispatch_attempt", sa.Integer(), nullable=True),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("error_code", sa.String(80), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
        sa.UniqueConstraint("run_id"),
    )


def downgrade() -> None:
    op.drop_table("web_search_proposals")
