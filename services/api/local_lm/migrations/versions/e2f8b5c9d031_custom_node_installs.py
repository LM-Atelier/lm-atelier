"""track pinned custom node installations

Revision ID: e2f8b5c9d031
Revises: d1e7a4b8c920
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e2f8b5c9d031"
down_revision: str | None = "d1e7a4b8c920"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "custom_node_installs",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column("name", sa.String(length=240), nullable=False),
        sa.Column("source_url", sa.String(length=1000), nullable=False, unique=True),
        sa.Column("revision", sa.String(length=40), nullable=False),
        sa.Column("previous_revision", sa.String(length=40)),
        sa.Column("installed_path", sa.Text(), nullable=False),
        sa.Column("tree_hash", sa.String(length=64), nullable=False),
        sa.Column("trusted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("security_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_custom_node_installs_name", "custom_node_installs", ["name"])
    op.create_index("ix_custom_node_installs_trusted", "custom_node_installs", ["trusted"])
    op.create_index("ix_custom_node_installs_active", "custom_node_installs", ["active"])


def downgrade() -> None:
    op.drop_index("ix_custom_node_installs_active", table_name="custom_node_installs")
    op.drop_index("ix_custom_node_installs_trusted", table_name="custom_node_installs")
    op.drop_index("ix_custom_node_installs_name", table_name="custom_node_installs")
    op.drop_table("custom_node_installs")
