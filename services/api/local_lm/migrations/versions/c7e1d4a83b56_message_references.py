"""record what a turn referred to, as it stood at the time

Revision ID: c7e1d4a83b56
Revises: b3f7c2a19d54
Create Date: 2026-08-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7e1d4a83b56"
down_revision: str | None = "b3f7c2a19d54"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "message_references",
        sa.Column("id", sa.String(length=48), nullable=False),
        sa.Column("message_id", sa.String(length=40), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        # Snapshots, deliberately without a foreign key to reference_subjects.
        # A subject can be renamed, archived or deleted long after a turn used
        # it, and the record of what that turn referred to has to survive all
        # three. A live foreign key would erase the history at exactly the
        # moment someone asked why an old picture looks the way it does.
        sa.Column("reference_subject_id", sa.String(length=48), nullable=False),
        sa.Column("mention_slug", sa.String(length=64), nullable=False),
        sa.Column("subject_name", sa.String(length=120), nullable=False),
        sa.Column("subject_kind", sa.String(length=40), nullable=False),
        sa.Column("role", sa.String(length=40), nullable=True),
        sa.Column("strength", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=40), nullable=False),
        # Copied for the same reason and likewise not foreign keys: a cleanup
        # that removes unreferenced bytes must not erase the record that those
        # bytes were once used.
        sa.Column("reference_asset_ids_json", sa.JSON(), nullable=False),
        sa.Column("artifact_ids_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        # The turn is a real foreign key: deleting a user's turn removes what
        # it produced, and a reference record outliving its own message would
        # be an orphan nobody could interpret.
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        # One entry per slot, so re-recording a turn cannot quietly double what
        # that turn referred to.
        sa.UniqueConstraint("message_id", "position", name="uq_message_reference_position"),
        sa.CheckConstraint("position >= 0", name="ck_message_reference_position"),
        sa.CheckConstraint(
            "length(trim(reference_subject_id)) > 0",
            name="ck_message_reference_subject_present",
        ),
    )
    op.create_index("ix_message_references_message", "message_references", ["message_id"])
    op.create_index("ix_message_references_subject", "message_references", ["reference_subject_id"])


def downgrade() -> None:
    op.drop_index("ix_message_references_subject", table_name="message_references")
    op.drop_index("ix_message_references_message", table_name="message_references")
    op.drop_table("message_references")
