"""Keep bounded recovery selections and their atomic replay results.

Revision ID: 7d39a4c81e20
Revises: 4b6e18d7c092
Create Date: 2026-10-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7d39a4c81e20"
down_revision: str | None = "4b6e18d7c092"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "recovery_batches",
        sa.Column("id", sa.String(40), nullable=False),
        sa.Column("preview_json", sa.JSON(), nullable=False),
        sa.Column("fingerprints_json", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("operation_key", sa.String(128), nullable=True),
        sa.Column("request_sha256", sa.String(64), nullable=True),
        sa.Column("response_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("operation_key"),
    )
    op.create_index("ix_recovery_batches_expires_at", "recovery_batches", ["expires_at"])


def downgrade() -> None:
    op.drop_table("recovery_batches")
