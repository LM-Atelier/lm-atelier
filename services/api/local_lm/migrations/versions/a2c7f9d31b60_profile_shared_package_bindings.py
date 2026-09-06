"""Add profile-local shared package references without changing existing installs."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "a2c7f9d31b60"
down_revision: str | None = "f5c2a8d91e40"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "shared_package_bindings",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("library_id", sa.String(36), nullable=False),
        sa.Column("consumer_id", sa.String(64), nullable=False),
        sa.Column("package_digest", sa.String(64), nullable=False),
        sa.Column("member_digests_json", sa.JSON(), nullable=False),
        sa.Column("claim_id", sa.String(32), nullable=True),
        sa.Column("state", sa.String(24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "library_id",
            "consumer_id",
            "package_digest",
            name="uq_shared_package_binding_identity",
        ),
    )
    # Nullable inline references let SQLite extend both tables without copying
    # existing model rows or changing their local paths and activation state.
    for table in ("model_installs", "model_asset_installs"):
        op.execute(
            f"ALTER TABLE {table} ADD COLUMN shared_package_binding_id VARCHAR(40)"
            " REFERENCES shared_package_bindings(id)"
        )
        op.create_index(
            f"ix_{table}_shared_package_binding_id",
            table,
            ["shared_package_binding_id"],
        )


def downgrade() -> None:
    connection = op.get_bind()
    if connection.execute(sa.text("SELECT 1 FROM shared_package_bindings LIMIT 1")).first():
        raise RuntimeError("shared package bindings must be released before downgrade")
    for table in ("model_asset_installs", "model_installs"):
        op.drop_index(f"ix_{table}_shared_package_binding_id", table_name=table)
        op.drop_column(table, "shared_package_binding_id")
    op.drop_table("shared_package_bindings")
