"""group operation workflows under selectable families

Revision ID: b6a1e4d92c70
Revises: d8b3e6f92a41
Create Date: 2026-08-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b6a1e4d92c70"
down_revision: str | None = "d8b3e6f92a41"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_families",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("name", sa.String(length=240), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("use_case", sa.Text(), nullable=False),
        sa.Column("tags_json", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_workflow_families_name", "workflow_families", ["name"])
    op.create_index("ix_workflow_families_archived", "workflow_families", ["archived"])

    with op.batch_alter_table("workflow_definitions") as batch_op:
        batch_op.add_column(sa.Column("family_id", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("variant_key", sa.String(length=100), nullable=True))
        batch_op.create_foreign_key(
            "fk_workflow_definitions_family",
            "workflow_families",
            ["family_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_unique_constraint(
            "uq_workflow_definition_family_variant",
            ["family_id", "variant_key"],
        )
        batch_op.create_check_constraint(
            "ck_workflow_definition_family_variant",
            "family_id IS NULL OR variant_key IS NOT NULL",
        )
        batch_op.create_index("ix_workflow_definitions_family_id", ["family_id"], unique=False)

    with op.batch_alter_table("workflow_revisions") as batch_op:
        batch_op.add_column(
            sa.Column(
                "capabilities_json",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'[]'"),
            )
        )

    op.create_table(
        "workflow_preferences",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column(
            "workflow_family_id",
            sa.String(length=64),
            sa.ForeignKey("workflow_families.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("selector_capability", sa.String(length=32), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "workflow_family_id",
            "selector_capability",
            name="uq_workflow_preference_family_selector",
        ),
        sa.CheckConstraint(
            "length(trim(selector_capability)) > 0",
            name="ck_workflow_preference_selector_nonempty",
        ),
        sa.CheckConstraint(
            "NOT is_default OR enabled",
            name="ck_workflow_preference_default_enabled",
        ),
    )
    op.create_index(
        "ix_workflow_preferences_workflow_family_id",
        "workflow_preferences",
        ["workflow_family_id"],
    )
    op.create_index(
        "ix_workflow_preferences_selector_order",
        "workflow_preferences",
        ["selector_capability", "enabled", "sort_order"],
    )
    op.create_index(
        "uq_workflow_preferences_default_selector",
        "workflow_preferences",
        ["selector_capability"],
        unique=True,
        sqlite_where=sa.text("is_default = 1"),
        postgresql_where=sa.text("is_default IS TRUE"),
    )


def downgrade() -> None:
    connection = op.get_bind()
    family_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM workflow_families")
    ).scalar_one()
    preference_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM workflow_preferences")
    ).scalar_one()
    assignment_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM workflow_definitions WHERE family_id IS NOT NULL")
    ).scalar_one()
    if family_count or preference_count or assignment_count:
        raise RuntimeError(
            "Cannot downgrade workflow families while family data exists; "
            "archive or migrate those records before retrying."
        )

    op.drop_index(
        "uq_workflow_preferences_default_selector",
        table_name="workflow_preferences",
    )
    op.drop_index(
        "ix_workflow_preferences_selector_order",
        table_name="workflow_preferences",
    )
    op.drop_index(
        "ix_workflow_preferences_workflow_family_id",
        table_name="workflow_preferences",
    )
    op.drop_table("workflow_preferences")

    with op.batch_alter_table("workflow_definitions") as batch_op:
        batch_op.drop_index("ix_workflow_definitions_family_id")
        batch_op.drop_constraint("ck_workflow_definition_family_variant", type_="check")
        batch_op.drop_constraint("uq_workflow_definition_family_variant", type_="unique")
        batch_op.drop_constraint("fk_workflow_definitions_family", type_="foreignkey")
        batch_op.drop_column("variant_key")
        batch_op.drop_column("family_id")

    with op.batch_alter_table("workflow_revisions") as batch_op:
        batch_op.drop_column("capabilities_json")

    op.drop_index("ix_workflow_families_archived", table_name="workflow_families")
    op.drop_index("ix_workflow_families_name", table_name="workflow_families")
    op.drop_table("workflow_families")
