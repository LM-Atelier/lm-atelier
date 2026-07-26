"""add model asset installs

Revision ID: 6b3e8c4f1a20
Revises: a6d9c4e21f70
Create Date: 2026-07-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "6b3e8c4f1a20"
down_revision: str | None = "a6d9c4e21f70"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "model_asset_installs",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("source_id", sa.String(length=40), nullable=True),
        sa.Column("name", sa.String(length=300), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("family", sa.String(length=100), nullable=True),
        sa.Column("local_path", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "manifest_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["model_sources.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_model_asset_installs_source_id",
        "model_asset_installs",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        "ix_model_asset_installs_name",
        "model_asset_installs",
        ["name"],
        unique=False,
    )
    op.create_index(
        "ix_model_asset_installs_kind",
        "model_asset_installs",
        ["kind"],
        unique=False,
    )
    op.create_index(
        "ix_model_asset_installs_family",
        "model_asset_installs",
        ["family"],
        unique=False,
    )
    op.create_index(
        "ix_model_asset_installs_active",
        "model_asset_installs",
        ["active"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("model_asset_installs")
