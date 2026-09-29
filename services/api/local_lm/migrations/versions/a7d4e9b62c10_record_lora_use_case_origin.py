"""Record whether a LoRA use case came from provider metadata.

Revision ID: a7d4e9b62c10
Revises: c3b6d91e8a20
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7d4e9b62c10"
down_revision: str | None = "c3b6d91e8a20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_asset_installs",
        sa.Column("use_case_derived", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("model_asset_installs", "use_case_derived")
