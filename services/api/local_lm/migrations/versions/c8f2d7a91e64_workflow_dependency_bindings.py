"""persist exact workflow dependency activations

Revision ID: c8f2d7a91e64
Revises: b6a1e4d92c70
Create Date: 2026-08-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c8f2d7a91e64"
down_revision: str | None = "b6a1e4d92c70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _lowercase_sha256_check(column: str) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = 64 AND lower({column}) = {column} AND {remainder} = ''"


def upgrade() -> None:
    with op.batch_alter_table("workflow_revisions") as batch_op:
        batch_op.add_column(
            sa.Column("dependency_contract_sha256", sa.String(length=64), nullable=True)
        )
        batch_op.create_check_constraint(
            "ck_workflow_revision_dependency_contract_sha256",
            "dependency_contract_sha256 IS NULL OR ("
            + _lowercase_sha256_check("dependency_contract_sha256")
            + ")",
        )

    op.create_table(
        "workflow_dependency_slots",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column(
            "workflow_revision_id",
            sa.String(length=40),
            sa.ForeignKey("workflow_revisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("resource_kind", sa.String(length=32), nullable=False),
        sa.Column("required", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "satisfaction",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'all_of'"),
        ),
        sa.Column(
            "requirements_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("contract_sha256", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "workflow_revision_id",
            "name",
            name="uq_workflow_dependency_slot_revision_name",
        ),
        sa.UniqueConstraint(
            "workflow_revision_id",
            "ordinal",
            name="uq_workflow_dependency_slot_revision_ordinal",
        ),
        sa.UniqueConstraint(
            "id",
            "workflow_revision_id",
            name="uq_workflow_dependency_slot_id_revision",
        ),
        sa.CheckConstraint(
            "length(trim(name)) > 0",
            name="ck_workflow_dependency_slot_name_nonempty",
        ),
        sa.CheckConstraint(
            "resource_kind IN ('model_profile', 'model_install', 'model_asset', "
            "'custom_node', 'registry_package', 'runtime')",
            name="ck_workflow_dependency_slot_resource_kind",
        ),
        sa.CheckConstraint(
            "satisfaction IN ('all_of', 'any_of')",
            name="ck_workflow_dependency_slot_satisfaction",
        ),
        sa.CheckConstraint(
            "required IN (false, true)",
            name="ck_workflow_dependency_slot_required",
        ),
        sa.CheckConstraint(
            "ordinal >= 0",
            name="ck_workflow_dependency_slot_ordinal",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("contract_sha256"),
            name="ck_workflow_dependency_slot_contract_sha256",
        ),
    )
    op.create_index(
        "ix_workflow_dependency_slots_workflow_revision_id",
        "workflow_dependency_slots",
        ["workflow_revision_id"],
    )

    op.create_table(
        "workflow_activations",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column(
            "workflow_revision_id",
            sa.String(length=40),
            sa.ForeignKey("workflow_revisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("resolver_version", sa.String(length=40), nullable=False),
        sa.Column("dependency_contract_sha256", sa.String(length=64), nullable=False),
        sa.Column("binding_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "state",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'ready'"),
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("last_validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("invalidated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("invalidation_code", sa.String(length=80), nullable=True),
        sa.Column("invalidation_reason", sa.Text(), nullable=True),
        sa.Column("details_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "workflow_revision_id",
            "binding_sha256",
            name="uq_workflow_activation_revision_binding",
        ),
        sa.UniqueConstraint(
            "id",
            "workflow_revision_id",
            name="uq_workflow_activation_id_revision",
        ),
        sa.CheckConstraint(
            "state IN ('ready', 'stale', 'disabled')",
            name="ck_workflow_activation_state",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("dependency_contract_sha256"),
            name="ck_workflow_activation_contract_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("binding_sha256"),
            name="ck_workflow_activation_binding_sha256",
        ),
        sa.CheckConstraint(
            "NOT is_active OR (state = 'ready' AND invalidated_at IS NULL)",
            name="ck_workflow_activation_active_ready",
        ),
        sa.CheckConstraint(
            "is_active IN (false, true)",
            name="ck_workflow_activation_is_active",
        ),
    )
    op.create_index(
        "ix_workflow_activations_workflow_revision_id",
        "workflow_activations",
        ["workflow_revision_id"],
    )
    op.create_index("ix_workflow_activations_state", "workflow_activations", ["state"])
    op.create_index("ix_workflow_activations_is_active", "workflow_activations", ["is_active"])
    op.create_index(
        "uq_workflow_activation_active_revision",
        "workflow_activations",
        ["workflow_revision_id"],
        unique=True,
        sqlite_where=sa.text("is_active = 1"),
        postgresql_where=sa.text("is_active IS TRUE"),
    )

    op.create_table(
        "workflow_dependency_bindings",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column("workflow_revision_id", sa.String(length=40), nullable=False),
        sa.Column("workflow_activation_id", sa.String(length=40), nullable=False),
        sa.Column("workflow_dependency_slot_id", sa.String(length=40), nullable=False),
        sa.Column("requirement_key", sa.String(length=100), nullable=False),
        sa.Column(
            "model_profile_id",
            sa.String(length=40),
            sa.ForeignKey("model_profiles.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column(
            "model_install_id",
            sa.String(length=40),
            sa.ForeignKey("model_installs.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column(
            "model_asset_install_id",
            sa.String(length=40),
            sa.ForeignKey("model_asset_installs.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column(
            "custom_node_install_id",
            sa.String(length=40),
            sa.ForeignKey("custom_node_installs.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column(
            "comfy_registry_install_id",
            sa.String(length=64),
            sa.ForeignKey("comfy_registry_installs.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("runtime_key", sa.String(length=100), nullable=True),
        sa.Column("mount_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "resource_identity_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("resource_identity_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["workflow_activation_id", "workflow_revision_id"],
            ["workflow_activations.id", "workflow_activations.workflow_revision_id"],
            name="fk_workflow_dependency_binding_activation_revision",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workflow_dependency_slot_id", "workflow_revision_id"],
            ["workflow_dependency_slots.id", "workflow_dependency_slots.workflow_revision_id"],
            name="fk_workflow_dependency_binding_slot_revision",
        ),
        sa.UniqueConstraint(
            "workflow_activation_id",
            "workflow_dependency_slot_id",
            "requirement_key",
            name="uq_workflow_dependency_binding_assignment",
        ),
        sa.CheckConstraint(
            "length(trim(requirement_key)) > 0",
            name="ck_workflow_dependency_binding_requirement_nonempty",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("resource_identity_sha256"),
            name="ck_workflow_dependency_binding_identity_sha256",
        ),
        sa.CheckConstraint(
            "(CASE WHEN model_profile_id IS NOT NULL THEN 1 ELSE 0 END + "
            "CASE WHEN model_install_id IS NOT NULL THEN 1 ELSE 0 END + "
            "CASE WHEN model_asset_install_id IS NOT NULL THEN 1 ELSE 0 END + "
            "CASE WHEN custom_node_install_id IS NOT NULL THEN 1 ELSE 0 END + "
            "CASE WHEN comfy_registry_install_id IS NOT NULL THEN 1 ELSE 0 END + "
            "CASE WHEN runtime_key IS NOT NULL THEN 1 ELSE 0 END) = 1",
            name="ck_workflow_dependency_binding_one_locator",
        ),
    )
    for column in (
        "workflow_revision_id",
        "workflow_activation_id",
        "workflow_dependency_slot_id",
        "model_profile_id",
        "model_install_id",
        "model_asset_install_id",
        "custom_node_install_id",
        "comfy_registry_install_id",
    ):
        op.create_index(
            f"ix_workflow_dependency_bindings_{column}",
            "workflow_dependency_bindings",
            [column],
        )


def downgrade() -> None:
    connection = op.get_bind()
    binding_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM workflow_dependency_bindings")
    ).scalar_one()
    activation_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM workflow_activations")
    ).scalar_one()
    slot_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM workflow_dependency_slots")
    ).scalar_one()
    revision_contract_count = connection.execute(
        sa.text(
            "SELECT COUNT(*) FROM workflow_revisions WHERE dependency_contract_sha256 IS NOT NULL"
        )
    ).scalar_one()
    if binding_count or activation_count or slot_count or revision_contract_count:
        raise RuntimeError(
            "Cannot downgrade workflow dependency bindings while dependency data exists; "
            "remove or migrate those records before retrying."
        )

    for column in (
        "comfy_registry_install_id",
        "custom_node_install_id",
        "model_asset_install_id",
        "model_install_id",
        "model_profile_id",
        "workflow_dependency_slot_id",
        "workflow_activation_id",
        "workflow_revision_id",
    ):
        op.drop_index(
            f"ix_workflow_dependency_bindings_{column}",
            table_name="workflow_dependency_bindings",
        )
    op.drop_table("workflow_dependency_bindings")

    op.drop_index("uq_workflow_activation_active_revision", table_name="workflow_activations")
    op.drop_index("ix_workflow_activations_is_active", table_name="workflow_activations")
    op.drop_index("ix_workflow_activations_state", table_name="workflow_activations")
    op.drop_index(
        "ix_workflow_activations_workflow_revision_id",
        table_name="workflow_activations",
    )
    op.drop_table("workflow_activations")

    op.drop_index(
        "ix_workflow_dependency_slots_workflow_revision_id",
        table_name="workflow_dependency_slots",
    )
    op.drop_table("workflow_dependency_slots")

    with op.batch_alter_table("workflow_revisions") as batch_op:
        batch_op.drop_constraint("ck_workflow_revision_dependency_contract_sha256", type_="check")
        batch_op.drop_column("dependency_contract_sha256")
