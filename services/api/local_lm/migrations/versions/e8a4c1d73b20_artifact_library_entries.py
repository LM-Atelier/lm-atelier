"""separate Media Library membership from artifact bytes

Revision ID: e8a4c1d73b20
Revises: c7e1d4a83b56
Create Date: 2026-08-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from local_lm.artifact_library_schema import (
    AUDIT_TRIGGER_SQL,
    CREATE_TRIGGER_SQL,
    DROP_TRIGGER_SQL,
    PREMIGRATION_INVALID_SQL,
)

revision: str = "e8a4c1d73b20"
down_revision: str | None = "c7e1d4a83b56"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _triggers() -> None:
    for statement in CREATE_TRIGGER_SQL:
        op.execute(statement)
    for statement in AUDIT_TRIGGER_SQL:
        op.execute(statement)


def _audit_existing_references() -> None:
    if op.get_bind().exec_driver_sql(PREMIGRATION_INVALID_SQL).scalar_one():
        raise sa.exc.IntegrityError(
            "artifact JSON reference is invalid",
            None,
            ValueError("artifact JSON reference is invalid"),
        )


def upgrade() -> None:
    _audit_existing_references()
    op.create_table(
        "artifact_library_entries",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("artifact_id", sa.String(length=80), nullable=False),
        sa.Column("display_name", sa.String(length=500), nullable=False),
        sa.Column("favorite", sa.Boolean(), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("recovery_id", sa.String(length=80), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("artifact_id", name="uq_artifact_library_entry_artifact"),
        sa.CheckConstraint(
            "length(display_name) BETWEEN 1 AND 500 AND instr(display_name, char(0)) = 0",
            name="ck_library_entry_display_name",
        ),
        sa.CheckConstraint("favorite IN (0, 1)", name="ck_library_entry_favorite_boolean"),
        sa.CheckConstraint("state IN ('visible', 'trashed')", name="ck_library_entry_state"),
        sa.CheckConstraint("version > 0", name="ck_library_entry_version_positive"),
        sa.CheckConstraint(
            "(state = 'visible' AND deleted_at IS NULL AND recovery_id IS NULL) OR "
            "(state = 'trashed' AND deleted_at IS NOT NULL AND recovery_id IS NOT NULL)",
            name="ck_library_entry_recovery_consistent",
        ),
    )
    op.create_index(
        "ix_library_entry_state_created", "artifact_library_entries", ["state", "created_at", "id"]
    )
    op.create_index(
        "ix_library_entry_favorite_created",
        "artifact_library_entries",
        ["favorite", "created_at", "id"],
    )
    op.create_index(
        "ux_library_entry_recovery_id",
        "artifact_library_entries",
        ["recovery_id"],
        unique=True,
        sqlite_where=sa.text("recovery_id IS NOT NULL"),
    )
    op.execute("""
        INSERT INTO artifact_library_entries
          (id, artifact_id, display_name, favorite, state, deleted_at, recovery_id,
           version, created_at, updated_at)
        SELECT 'libentry:sha256:' || sha256, id,
               COALESCE(NULLIF(trim(original_name), ''), sha256),
               favorite, 'visible', NULL, NULL, 1, created_at, updated_at
        FROM artifacts WHERE kind IN ('image', 'video')
        ON CONFLICT(artifact_id) DO NOTHING
    """)
    _triggers()


def downgrade() -> None:
    for statement in DROP_TRIGGER_SQL:
        op.execute(statement)
    op.drop_index("ux_library_entry_recovery_id", table_name="artifact_library_entries")
    op.drop_index("ix_library_entry_favorite_created", table_name="artifact_library_entries")
    op.drop_index("ix_library_entry_state_created", table_name="artifact_library_entries")
    op.drop_table("artifact_library_entries")
