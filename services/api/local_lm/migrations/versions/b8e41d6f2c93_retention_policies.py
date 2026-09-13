"""Keep the retention windows chosen in Settings.

Revision ID: b8e41d6f2c93
Revises: a3f9c2e51d74
Create Date: 2026-09-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8e41d6f2c93"
down_revision: str | None = "a3f9c2e51d74"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # One new table and nothing else. With no row, retention keeps using the
    # installation's configured windows, so upgrading changes nothing about
    # what is cleared.
    op.create_table(
        "retention_policies",
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("media_days", sa.Integer(), nullable=False),
        sa.Column("temporary_hours", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("scope"),
    )


def downgrade() -> None:
    op.drop_table("retention_policies")
