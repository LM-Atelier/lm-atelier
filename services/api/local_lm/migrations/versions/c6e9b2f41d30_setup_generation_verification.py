"""add setup generation verification evidence

Revision ID: c6e9b2f41d30
Revises: b2c7d4e91a60
Create Date: 2026-07-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "c6e9b2f41d30"
down_revision: str | None = "b2c7d4e91a60"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "setup_verifications",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("evidence_key", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=24), nullable=False),
        sa.Column("model_install_id", sa.String(length=40), nullable=False),
        sa.Column("profile_id", sa.String(length=40), nullable=False),
        sa.Column("workflow_revision_id", sa.String(length=40), nullable=True),
        sa.Column("chat_id", sa.String(length=40), nullable=True),
        sa.Column("run_id", sa.String(length=40), nullable=True),
        sa.Column("job_id", sa.String(length=40), nullable=True),
        sa.Column("input_artifact_id", sa.String(length=40), nullable=True),
        sa.Column("failure_code", sa.String(length=80), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_install_id"],
            ["model_installs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["profile_id"],
            ["model_profiles.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workflow_revision_id"],
            ["workflow_revisions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("evidence_key", name="uq_setup_verification_evidence_key"),
    )
    op.create_index(
        "ix_setup_verifications_role_state",
        "setup_verifications",
        ["role", "state"],
    )
    op.create_index(
        "ix_setup_verifications_chat_id",
        "setup_verifications",
        ["chat_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_setup_verifications_chat_id", table_name="setup_verifications")
    op.drop_index("ix_setup_verifications_role_state", table_name="setup_verifications")
    op.drop_table("setup_verifications")
