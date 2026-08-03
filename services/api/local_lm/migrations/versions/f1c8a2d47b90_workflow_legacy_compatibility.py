"""add workflow-first legacy compatibility selections

Revision ID: f1c8a2d47b90
Revises: c8f2d7a91e64
Create Date: 2026-08-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f1c8a2d47b90"
down_revision: str | None = "c8f2d7a91e64"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _lowercase_sha256_check(column: str) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = 64 AND lower({column}) = {column} AND {remainder} = ''"


def upgrade() -> None:
    op.create_table(
        "workflow_profile_compatibility",
        sa.Column(
            "model_profile_id",
            sa.String(length=40),
            sa.ForeignKey("model_profiles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "workflow_family_id",
            sa.String(length=64),
            sa.ForeignKey("workflow_families.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_fingerprint_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            _lowercase_sha256_check("source_fingerprint_sha256"),
            name="ck_workflow_profile_compatibility_fingerprint_sha256",
        ),
    )
    op.create_index(
        "ix_workflow_profile_compatibility_workflow_family_id",
        "workflow_profile_compatibility",
        ["workflow_family_id"],
        unique=True,
    )

    op.create_table(
        "chat_workflow_selections",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column(
            "chat_id",
            sa.String(length=40),
            sa.ForeignKey("chats.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("selector_capability", sa.String(length=32), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column(
            "workflow_family_id",
            sa.String(length=64),
            sa.ForeignKey("workflow_families.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "chat_id",
            "selector_capability",
            name="uq_chat_workflow_selection_capability",
        ),
        sa.CheckConstraint(
            "length(trim(selector_capability)) > 0",
            name="ck_chat_workflow_selection_capability_nonempty",
        ),
        sa.CheckConstraint(
            "(mode = 'automatic' AND workflow_family_id IS NULL) OR "
            "(mode = 'family' AND workflow_family_id IS NOT NULL)",
            name="ck_chat_workflow_selection_mode_target",
        ),
    )
    op.create_index(
        "ix_chat_workflow_selections_chat_id",
        "chat_workflow_selections",
        ["chat_id"],
    )
    op.create_index(
        "ix_chat_workflow_selections_workflow_family_id",
        "chat_workflow_selections",
        ["workflow_family_id"],
    )

    op.create_table(
        "project_workflow_selections",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(length=40),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("selector_capability", sa.String(length=32), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column(
            "workflow_family_id",
            sa.String(length=64),
            sa.ForeignKey("workflow_families.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column(
            "workflow_revision_id",
            sa.String(length=40),
            sa.ForeignKey("workflow_revisions.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "project_id",
            "selector_capability",
            name="uq_project_workflow_selection_capability",
        ),
        sa.CheckConstraint(
            "length(trim(selector_capability)) > 0",
            name="ck_project_workflow_selection_capability_nonempty",
        ),
        sa.CheckConstraint(
            "(mode = 'automatic' AND workflow_family_id IS NULL "
            "AND workflow_revision_id IS NULL) OR "
            "(mode = 'family' AND workflow_family_id IS NOT NULL "
            "AND workflow_revision_id IS NULL) OR "
            "(mode = 'revision' AND workflow_family_id IS NULL "
            "AND workflow_revision_id IS NOT NULL)",
            name="ck_project_workflow_selection_mode_target",
        ),
    )
    op.create_index(
        "ix_project_workflow_selections_project_id",
        "project_workflow_selections",
        ["project_id"],
    )
    op.create_index(
        "ix_project_workflow_selections_workflow_family_id",
        "project_workflow_selections",
        ["workflow_family_id"],
    )
    op.create_index(
        "ix_project_workflow_selections_workflow_revision_id",
        "project_workflow_selections",
        ["workflow_revision_id"],
    )


def downgrade() -> None:
    connection = op.get_bind()
    populated = {
        table: connection.execute(sa.text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
        for table in (
            "workflow_profile_compatibility",
            "chat_workflow_selections",
            "project_workflow_selections",
        )
    }
    if any(populated.values()):
        raise RuntimeError(
            "Cannot downgrade workflow compatibility while mirrored selections exist; "
            "run a lossless compatibility cleanup before retrying."
        )

    op.drop_index(
        "ix_project_workflow_selections_workflow_revision_id",
        table_name="project_workflow_selections",
    )
    op.drop_index(
        "ix_project_workflow_selections_workflow_family_id",
        table_name="project_workflow_selections",
    )
    op.drop_index(
        "ix_project_workflow_selections_project_id",
        table_name="project_workflow_selections",
    )
    op.drop_table("project_workflow_selections")

    op.drop_index(
        "ix_chat_workflow_selections_workflow_family_id",
        table_name="chat_workflow_selections",
    )
    op.drop_index(
        "ix_chat_workflow_selections_chat_id",
        table_name="chat_workflow_selections",
    )
    op.drop_table("chat_workflow_selections")

    op.drop_index(
        "ix_workflow_profile_compatibility_workflow_family_id",
        table_name="workflow_profile_compatibility",
    )
    op.drop_table("workflow_profile_compatibility")
