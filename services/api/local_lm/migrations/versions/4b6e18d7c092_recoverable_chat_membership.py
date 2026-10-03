"""Keep deletion memberships and enforce immutable recoverable chat history.

Revision ID: 4b6e18d7c092
Revises: fbc390fda45f
Create Date: 2026-10-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from local_lm.artifact_library_schema import ENTRY_DELETE_TRIGGER
from local_lm.chat_recovery_schema import (
    CREATE_CHAT_RECOVERY_TRIGGER_SQL,
    DROP_CHAT_RECOVERY_TRIGGER_SQL,
)
from local_lm.media_recovery_schema import (
    CREATE_MEDIA_RECOVERY_TRIGGER_SQL,
    DROP_MEDIA_RECOVERY_TRIGGER_SQL,
)
from local_lm.project_recovery_schema import (
    CREATE_PROJECT_RECOVERY_TRIGGER_SQL,
    DROP_PROJECT_RECOVERY_TRIGGER_SQL,
)
from local_lm.workflow_recovery_schema import (
    CREATE_WORKFLOW_RECOVERY_TRIGGER_SQL,
    DROP_WORKFLOW_RECOVERY_TRIGGER_SQL,
)

revision: str = "4b6e18d7c092"
down_revision: str | None = "fbc390fda45f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "recovery_items",
        sa.Column("deletion_id", sa.String(40), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("display_label", sa.String(240), nullable=False),
        sa.Column("original_project_id", sa.String(40), nullable=True),
        sa.Column("original_project_label", sa.String(200), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("purge_after", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("subject_revision", sa.String(64), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("delete_generated_media", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("deletion_id"),
        sa.UniqueConstraint("kind", "subject_id", name="uq_recovery_item_subject"),
    )
    op.create_index("ix_recovery_items_kind", "recovery_items", ["kind"])
    op.create_index("ix_recovery_items_subject_id", "recovery_items", ["subject_id"])
    op.create_index("ix_recovery_items_state", "recovery_items", ["state"])
    op.create_index(
        "ix_recovery_item_expiry", "recovery_items", ["state", "purge_after", "deletion_id"]
    )
    op.create_table(
        "recovery_operations",
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("operation_key", sa.String(128), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("deletion_id", sa.String(40), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("kind", "subject_id", "operation_key"),
    )
    op.create_index("ix_recovery_operations_deletion_id", "recovery_operations", ["deletion_id"])
    op.create_table(
        "recovery_previews",
        sa.Column("revision", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("deletion_id", sa.String(40), nullable=True),
        sa.Column("subject_fingerprint", sa.String(64), nullable=False),
        sa.Column("impact_sha256", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("revision"),
    )
    op.create_index("ix_recovery_previews_subject_id", "recovery_previews", ["subject_id"])
    op.create_index("ix_recovery_preview_expiry", "recovery_previews", ["expires_at"])
    for statement in CREATE_CHAT_RECOVERY_TRIGGER_SQL:
        op.execute(statement)
    for statement in CREATE_MEDIA_RECOVERY_TRIGGER_SQL:
        op.execute(statement)
    for statement in CREATE_PROJECT_RECOVERY_TRIGGER_SQL:
        op.execute(statement)
    for statement in CREATE_WORKFLOW_RECOVERY_TRIGGER_SQL:
        op.execute(statement)


def downgrade() -> None:
    retained = (
        op.get_bind()
        .exec_driver_sql("SELECT EXISTS (SELECT 1 FROM recovery_items WHERE state != 'purged')")
        .scalar_one()
    )
    if retained:
        raise RuntimeError("Restore or purge recoverable items before removing recovery storage.")
    for statement in DROP_WORKFLOW_RECOVERY_TRIGGER_SQL:
        op.execute(statement)
    for statement in DROP_PROJECT_RECOVERY_TRIGGER_SQL:
        op.execute(statement)
    for statement in DROP_MEDIA_RECOVERY_TRIGGER_SQL:
        op.execute(statement)
    op.execute(ENTRY_DELETE_TRIGGER)
    for statement in DROP_CHAT_RECOVERY_TRIGGER_SQL:
        op.execute(statement)
    op.drop_table("recovery_previews")
    op.drop_table("recovery_operations")
    op.drop_table("recovery_items")
