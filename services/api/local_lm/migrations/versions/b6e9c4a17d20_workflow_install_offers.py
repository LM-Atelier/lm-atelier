"""add durable workflow install offers

Revision ID: b6e9c4a17d20
Revises: a4d7e2b9c150
Create Date: 2026-08-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b6e9c4a17d20"
down_revision: str | None = "a4d7e2b9c150"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _lowercase_sha256_check(column: str) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = 64 AND lower({column}) = {column} AND {remainder} = ''"


def upgrade() -> None:
    op.create_table(
        "workflow_install_offers",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("workflow_revision_id", sa.String(length=40), nullable=False),
        sa.Column("workflow_artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("dependency_contract_sha256", sa.String(length=64), nullable=False),
        sa.Column("binding_plan_sha256", sa.String(length=64), nullable=False),
        sa.Column("offer_sha256", sa.String(length=64), nullable=False),
        sa.Column("selections_json", sa.JSON(), nullable=False),
        sa.Column("assets_json", sa.JSON(), nullable=False),
        sa.Column("plan_count", sa.Integer(), nullable=False),
        sa.Column("total_bytes", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="ready", nullable=False),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("invalidated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("invalidation_code", sa.String(length=80), nullable=True),
        sa.Column("invalidation_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            _lowercase_sha256_check("workflow_artifact_sha256"),
            name="ck_workflow_install_offer_artifact_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("dependency_contract_sha256"),
            name="ck_workflow_install_offer_contract_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("binding_plan_sha256"),
            name="ck_workflow_install_offer_binding_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("offer_sha256"),
            name="ck_workflow_install_offer_sha256",
        ),
        sa.CheckConstraint(
            "status IN ('ready', 'queued', 'invalidated', 'completed', 'expired')",
            name="ck_workflow_install_offer_status",
        ),
        sa.CheckConstraint(
            "plan_count > 0",
            name="ck_workflow_install_offer_plan_count",
        ),
        sa.CheckConstraint(
            "total_bytes > 0",
            name="ck_workflow_install_offer_total_bytes",
        ),
        sa.ForeignKeyConstraint(
            ["workflow_revision_id"],
            ["workflow_revisions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_workflow_install_offers_workflow_revision_id",
        "workflow_install_offers",
        ["workflow_revision_id"],
    )
    op.create_index(
        "ix_workflow_install_offers_offer_sha256",
        "workflow_install_offers",
        ["offer_sha256"],
    )
    op.create_index(
        "ix_workflow_install_offers_status",
        "workflow_install_offers",
        ["status"],
    )
    op.create_index(
        "ix_workflow_install_offer_revision_status",
        "workflow_install_offers",
        ["workflow_revision_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_workflow_install_offer_revision_status",
        table_name="workflow_install_offers",
    )
    op.drop_index(
        "ix_workflow_install_offers_status",
        table_name="workflow_install_offers",
    )
    op.drop_index(
        "ix_workflow_install_offers_offer_sha256",
        table_name="workflow_install_offers",
    )
    op.drop_index(
        "ix_workflow_install_offers_workflow_revision_id",
        table_name="workflow_install_offers",
    )
    op.drop_table("workflow_install_offers")
