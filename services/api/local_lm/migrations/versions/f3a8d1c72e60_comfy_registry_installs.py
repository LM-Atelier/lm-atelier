"""persist exact Comfy Registry install identity

Revision ID: f3a8d1c72e60
Revises: b9d4f2e73a15
Create Date: 2026-08-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f3a8d1c72e60"
down_revision: str | None = "b9d4f2e73a15"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "comfy_registry_installs",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column("package_id", sa.String(length=100), nullable=False),
        sa.Column("package_version", sa.String(length=100), nullable=False),
        sa.Column("registry_record_id", sa.String(length=1000), nullable=False, unique=True),
        sa.Column("repository_url", sa.String(length=1000), nullable=False),
        sa.Column("download_url", sa.String(length=1000), nullable=False),
        sa.Column("archive_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("installed_path", sa.Text(), nullable=False, unique=True),
        sa.Column("node_types_json", sa.JSON(), nullable=False),
        sa.Column("pip_dependencies_json", sa.JSON(), nullable=False),
        sa.Column("review_json", sa.JSON(), nullable=False),
        sa.Column("trusted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "package_id",
            "package_version",
            name="uq_comfy_registry_install_package_version",
        ),
    )
    op.create_index(
        "ix_comfy_registry_installs_package_id",
        "comfy_registry_installs",
        ["package_id"],
    )
    op.create_index("ix_comfy_registry_installs_trusted", "comfy_registry_installs", ["trusted"])
    op.create_index("ix_comfy_registry_installs_active", "comfy_registry_installs", ["active"])


def downgrade() -> None:
    op.drop_index("ix_comfy_registry_installs_active", table_name="comfy_registry_installs")
    op.drop_index("ix_comfy_registry_installs_trusted", table_name="comfy_registry_installs")
    op.drop_index("ix_comfy_registry_installs_package_id", table_name="comfy_registry_installs")
    op.drop_table("comfy_registry_installs")
