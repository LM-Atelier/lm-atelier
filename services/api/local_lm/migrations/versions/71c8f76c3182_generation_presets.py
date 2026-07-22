"""generation presets

Revision ID: 71c8f76c3182
Revises: 266b3b9df743
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "71c8f76c3182"
down_revision: str | None = "266b3b9df743"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "generation_presets",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("settings_json", sa.JSON(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("role", "name", name="uq_preset_role_name"),
    )
    op.create_index("ix_generation_presets_name", "generation_presets", ["name"])
    op.create_index("ix_generation_presets_role", "generation_presets", ["role"])


def downgrade() -> None:
    op.drop_index("ix_generation_presets_role", table_name="generation_presets")
    op.drop_index("ix_generation_presets_name", table_name="generation_presets")
    op.drop_table("generation_presets")
