"""add install plans, components, and capability evidence

Revision ID: e5c8a1b7f240
Revises: d4b7e2a61f90
Create Date: 2026-07-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "e5c8a1b7f240"
down_revision: str | None = "d4b7e2a61f90"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "install_plans",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("remote_id", sa.String(length=500), nullable=False),
        sa.Column("revision", sa.String(length=200), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("engine", sa.String(length=32), nullable=False),
        sa.Column("architecture", sa.String(length=200), nullable=True),
        sa.Column("family", sa.String(length=100), nullable=True),
        sa.Column("plan_hash", sa.String(length=64), nullable=False),
        sa.Column("resolver_version", sa.String(length=40), nullable=False),
        sa.Column("compatibility", sa.String(length=40), nullable=False),
        sa.Column(
            "artifacts_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "runtime_contract_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "activation_probe_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("failure_code", sa.String(length=80), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("plan_hash", name="uq_install_plan_hash"),
    )
    op.create_index(
        "ix_install_plan_source",
        "install_plans",
        ["provider", "remote_id", "revision", "role"],
        unique=False,
    )
    op.create_index(
        op.f("ix_install_plans_plan_hash"),
        "install_plans",
        ["plan_hash"],
        unique=True,
    )
    op.create_index(
        op.f("ix_install_plans_status"),
        "install_plans",
        ["status"],
        unique=False,
    )

    op.create_table(
        "model_component_manifests",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("model_install_id", sa.String(length=40), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("target_folder", sa.String(length=80), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("required", sa.Boolean(), nullable=False),
        sa.Column(
            "metadata_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_install_id"],
            ["model_installs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_install_id",
            "relative_path",
            name="uq_model_component_path",
        ),
    )
    op.create_index(
        op.f("ix_model_component_manifests_model_install_id"),
        "model_component_manifests",
        ["model_install_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_model_component_manifests_sha256"),
        "model_component_manifests",
        ["sha256"],
        unique=False,
    )

    op.create_table(
        "model_capability_evidence",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("model_install_id", sa.String(length=40), nullable=False),
        sa.Column("evidence_key", sa.String(length=64), nullable=False),
        sa.Column("result", sa.String(length=24), nullable=False),
        sa.Column(
            "component_hashes_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("runtime_build", sa.String(length=200), nullable=False),
        sa.Column("adapter_contract_version", sa.Integer(), nullable=False),
        sa.Column("launch_contract_version", sa.String(length=40), nullable=False),
        sa.Column("workflow_contract_version", sa.String(length=100), nullable=True),
        sa.Column("hardware_class", sa.String(length=200), nullable=False),
        sa.Column("probe_version", sa.String(length=40), nullable=False),
        sa.Column("failure_code", sa.String(length=80), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column(
            "details_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("probed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_install_id"],
            ["model_installs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_install_id",
            "evidence_key",
            name="uq_model_capability_evidence_install_key",
        ),
    )
    op.create_index(
        "ix_model_capability_evidence_install_result",
        "model_capability_evidence",
        ["model_install_id", "result"],
        unique=False,
    )
    op.create_index(
        op.f("ix_model_capability_evidence_evidence_key"),
        "model_capability_evidence",
        ["evidence_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_model_capability_evidence_model_install_id"),
        "model_capability_evidence",
        ["model_install_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_model_capability_evidence_model_install_id"),
        table_name="model_capability_evidence",
    )
    op.drop_index(
        op.f("ix_model_capability_evidence_evidence_key"),
        table_name="model_capability_evidence",
    )
    op.drop_index(
        "ix_model_capability_evidence_install_result",
        table_name="model_capability_evidence",
    )
    op.drop_table("model_capability_evidence")
    op.drop_index(
        op.f("ix_model_component_manifests_sha256"),
        table_name="model_component_manifests",
    )
    op.drop_index(
        op.f("ix_model_component_manifests_model_install_id"),
        table_name="model_component_manifests",
    )
    op.drop_table("model_component_manifests")
    op.drop_index(op.f("ix_install_plans_status"), table_name="install_plans")
    op.drop_index(op.f("ix_install_plans_plan_hash"), table_name="install_plans")
    op.drop_index("ix_install_plan_source", table_name="install_plans")
    op.drop_table("install_plans")
