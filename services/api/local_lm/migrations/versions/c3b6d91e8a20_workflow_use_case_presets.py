"""Persist use-case recipes and independent project and chat choices."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3b6d91e8a20"
down_revision: str | None = "b7a9d3c14620"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_use_case_presets",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("use_case", sa.String(32), nullable=False),
        sa.Column("settings_json", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("builtin", sa.Boolean(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("use_case", "name", name="uq_workflow_use_case_preset_name"),
        sa.UniqueConstraint("id", "use_case", name="uq_workflow_use_case_preset_target"),
        sa.CheckConstraint(
            "is_default = 0 OR enabled = 1", name="ck_workflow_use_case_default_enabled"
        ),
    )
    op.create_index(
        "uq_workflow_use_case_default",
        "workflow_use_case_presets",
        ["use_case"],
        unique=True,
        sqlite_where=sa.text("is_default = 1"),
    )
    for scope in ("project", "chat"):
        op.create_table(
            f"{scope}_workflow_use_case_selections",
            sa.Column(
                f"{scope}_id",
                sa.String(40),
                sa.ForeignKey(f"{scope}s.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("use_case", sa.String(32), primary_key=True),
            sa.Column("preset_id", sa.String(40), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(
                ["preset_id", "use_case"],
                ["workflow_use_case_presets.id", "workflow_use_case_presets.use_case"],
                ondelete="RESTRICT",
            ),
        )


def downgrade() -> None:
    tables = (
        "chat_workflow_use_case_selections",
        "project_workflow_use_case_selections",
        "workflow_use_case_presets",
    )
    queries = (
        "SELECT 1 FROM chat_workflow_use_case_selections LIMIT 1",
        "SELECT 1 FROM project_workflow_use_case_selections LIMIT 1",
        "SELECT 1 FROM workflow_use_case_presets LIMIT 1",
    )
    if any(op.get_bind().execute(sa.text(query)).first() for query in queries):
        raise RuntimeError("Saved workflow use-case choices require the current database version.")
    for table in tables[:2]:
        op.drop_table(table)
    op.drop_index("uq_workflow_use_case_default", table_name="workflow_use_case_presets")
    op.drop_table("workflow_use_case_presets")
