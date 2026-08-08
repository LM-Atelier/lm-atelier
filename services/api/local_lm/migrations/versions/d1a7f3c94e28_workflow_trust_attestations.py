"""record what this machine verified about a workflow revision

Revision ID: d1a7f3c94e28
Revises: c4f7a2e81b60
Create Date: 2026-08-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d1a7f3c94e28"
down_revision: str | None = "c4f7a2e81b60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _lowercase_sha256_check(column: str) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = 64 AND lower({column}) = {column} AND {remainder} = ''"


def upgrade() -> None:
    op.create_table(
        "workflow_trust_attestations",
        sa.Column("id", sa.String(length=48), nullable=False),
        sa.Column("workflow_revision_id", sa.String(length=40), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("runtime_contract_sha256", sa.String(length=64), nullable=True),
        sa.Column("runtime_managed", sa.Boolean(), nullable=False),
        sa.Column("node_inventory_sha256", sa.String(length=64), nullable=False),
        sa.Column("whitelist_sha256", sa.String(length=64), nullable=False),
        sa.Column("launch_scope_sha256", sa.String(length=64), nullable=True),
        sa.Column("required_node_types_json", sa.JSON(), nullable=False),
        sa.Column("declared_dependencies_json", sa.JSON(), nullable=False),
        sa.Column("resolution_json", sa.JSON(), nullable=False),
        sa.Column("attested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["workflow_revision_id"],
            ["workflow_revisions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        # One attestation per revision. A second would raise the question of
        # which one the run path believed, and the answer has to be that there
        # is only ever one.
        sa.UniqueConstraint("workflow_revision_id", name="uq_workflow_trust_attestation_revision"),
        sa.CheckConstraint(
            _lowercase_sha256_check("artifact_sha256"),
            name="ck_workflow_trust_attestation_artifact_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("node_inventory_sha256"),
            name="ck_workflow_trust_attestation_node_inventory_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("whitelist_sha256"),
            name="ck_workflow_trust_attestation_whitelist_sha256",
        ),
        sa.CheckConstraint(
            "runtime_contract_sha256 IS NULL OR ("
            + _lowercase_sha256_check("runtime_contract_sha256")
            + ")",
            name="ck_workflow_trust_attestation_runtime_contract_sha256",
        ),
        sa.CheckConstraint(
            "launch_scope_sha256 IS NULL OR ("
            + _lowercase_sha256_check("launch_scope_sha256")
            + ")",
            name="ck_workflow_trust_attestation_launch_scope_sha256",
        ),
    )
    op.create_index(
        "ix_workflow_trust_attestations_workflow_revision_id",
        "workflow_trust_attestations",
        ["workflow_revision_id"],
    )
    op.create_index(
        "ix_workflow_trust_attestations_artifact_sha256",
        "workflow_trust_attestations",
        ["artifact_sha256"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_workflow_trust_attestations_artifact_sha256",
        table_name="workflow_trust_attestations",
    )
    op.drop_index(
        "ix_workflow_trust_attestations_workflow_revision_id",
        table_name="workflow_trust_attestations",
    )
    op.drop_table("workflow_trust_attestations")
