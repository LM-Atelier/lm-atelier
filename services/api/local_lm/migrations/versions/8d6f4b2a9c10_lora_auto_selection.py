"""add LoRA automatic-selection metadata

Revision ID: 8d6f4b2a9c10
Revises: 6b3e8c4f1a20
Create Date: 2026-07-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "8d6f4b2a9c10"
down_revision: str | None = "6b3e8c4f1a20"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "model_asset_installs",
        sa.Column("use_case", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "model_asset_installs",
        sa.Column("auto_apply", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "model_asset_installs",
        sa.Column(
            "default_model_strength",
            sa.Float(),
            nullable=False,
            server_default="1.0",
        ),
    )
    op.add_column(
        "model_asset_installs",
        sa.Column(
            "default_clip_strength",
            sa.Float(),
            nullable=False,
            server_default="1.0",
        ),
    )


def downgrade() -> None:
    op.drop_column("model_asset_installs", "default_clip_strength")
    op.drop_column("model_asset_installs", "default_model_strength")
    op.drop_column("model_asset_installs", "auto_apply")
    op.drop_column("model_asset_installs", "use_case")
