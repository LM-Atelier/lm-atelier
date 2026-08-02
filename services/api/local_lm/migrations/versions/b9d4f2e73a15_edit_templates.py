"""one-click edit templates

Revision ID: b9d4f2e73a15
Revises: c7f3a1e58d24
Create Date: 2026-08-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b9d4f2e73a15"
down_revision: str | None = "c7f3a1e58d24"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "edit_templates",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("instruction", sa.Text(), nullable=False),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("settings_json", sa.JSON(), nullable=False),
        sa.Column("trigger_words_json", sa.JSON(), nullable=False),
        sa.Column("content_rating", sa.String(length=16), nullable=False),
        sa.Column("builtin", sa.Boolean(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name", name="uq_edit_template_name"),
    )
    op.create_index("ix_edit_templates_name", "edit_templates", ["name"])
    op.create_index("ix_edit_templates_content_rating", "edit_templates", ["content_rating"])


def downgrade() -> None:
    op.drop_index("ix_edit_templates_content_rating", table_name="edit_templates")
    op.drop_index("ix_edit_templates_name", table_name="edit_templates")
    op.drop_table("edit_templates")
