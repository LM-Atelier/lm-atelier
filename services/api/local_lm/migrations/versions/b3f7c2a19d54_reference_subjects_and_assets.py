"""record subjects a user can name, and the images that show them

Revision ID: b3f7c2a19d54
Revises: f4d9e6a20c81
Create Date: 2026-08-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b3f7c2a19d54"
down_revision: str | None = "f4d9e6a20c81"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "reference_subjects",
        sa.Column("id", sa.String(length=48), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        # The addressing token, canonicalised before it arrives so a plain
        # unique index is enough to stop two subjects sharing one mention.
        sa.Column("mention_slug", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("aliases_json", sa.JSON(), nullable=False),
        sa.Column("tags_json", sa.JSON(), nullable=False),
        sa.Column("cover_artifact_id", sa.String(length=80), nullable=True),
        sa.Column("favorite", sa.Boolean(), nullable=False),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        # Cleared rather than cascading: losing a cover image must not lose the
        # subject, because images are replaceable and the identity is not.
        sa.ForeignKeyConstraint(["cover_artifact_id"], ["artifacts.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("mention_slug", name="uq_reference_subject_mention_slug"),
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_reference_subject_name_present"),
        sa.CheckConstraint(
            "length(trim(mention_slug)) > 0", name="ck_reference_subject_slug_present"
        ),
    )
    op.create_index("ix_reference_subjects_mention_slug", "reference_subjects", ["mention_slug"])
    op.create_index("ix_reference_subjects_kind", "reference_subjects", ["kind"])
    op.create_index("ix_reference_subjects_favorite", "reference_subjects", ["favorite"])
    op.create_index("ix_reference_subjects_archived", "reference_subjects", ["archived"])

    op.create_table(
        "reference_assets",
        sa.Column("id", sa.String(length=48), nullable=False),
        sa.Column("reference_subject_id", sa.String(length=48), nullable=False),
        sa.Column("artifact_id", sa.String(length=80), nullable=False),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("purpose", sa.String(length=40), nullable=False),
        sa.Column("view_label", sa.String(length=60), nullable=True),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("validation_state", sa.String(length=30), nullable=False),
        sa.Column("validation_reasons_json", sa.JSON(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["reference_subject_id"], ["reference_subjects.id"], ondelete="CASCADE"
        ),
        # Restricted rather than cascading: an artifact a Reference still uses
        # must not be removable out from under it by an unrelated cleanup.
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        # The same image twice under one subject is a duplicate, not a second
        # view, and would silently weight the set toward one picture.
        sa.UniqueConstraint(
            "reference_subject_id", "artifact_id", name="uq_reference_asset_membership"
        ),
    )
    op.create_index("ix_reference_assets_subject", "reference_assets", ["reference_subject_id"])
    op.create_index("ix_reference_assets_artifact", "reference_assets", ["artifact_id"])


def downgrade() -> None:
    op.drop_index("ix_reference_assets_artifact", table_name="reference_assets")
    op.drop_index("ix_reference_assets_subject", table_name="reference_assets")
    op.drop_table("reference_assets")
    op.drop_index("ix_reference_subjects_archived", table_name="reference_subjects")
    op.drop_index("ix_reference_subjects_favorite", table_name="reference_subjects")
    op.drop_index("ix_reference_subjects_kind", table_name="reference_subjects")
    op.drop_index("ix_reference_subjects_mention_slug", table_name="reference_subjects")
    op.drop_table("reference_subjects")
