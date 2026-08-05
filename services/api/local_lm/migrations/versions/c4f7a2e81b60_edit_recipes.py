"""Edit templates become recipes

Revision ID: c4f7a2e81b60
Revises: b7e4c1a92f80

A template carried an instruction and some settings, so applying one
reproduced the words but not the run: the workflow, the model, and whether
the edit was scoped to a selection were whatever happened to be current. The
new columns are all nullable, which is the honest state for every template
that already exists - nobody recorded what produced them, and inventing a
binding now would claim knowledge the record does not have.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "c4f7a2e81b60"
down_revision: str | None = "b7e4c1a92f80"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "edit_templates",
        sa.Column("workflow_revision_id", sa.String(length=40), nullable=True),
    )
    op.add_column(
        "edit_templates",
        sa.Column("model_profile_id", sa.String(length=40), nullable=True),
    )
    # "none" rather than nullable: every existing template edits the whole
    # picture, and that is a fact about them rather than a gap in the record.
    op.add_column(
        "edit_templates",
        sa.Column("mask_mode", sa.String(length=16), nullable=False, server_default="none"),
    )


def downgrade() -> None:
    op.drop_column("edit_templates", "mask_mode")
    op.drop_column("edit_templates", "model_profile_id")
    op.drop_column("edit_templates", "workflow_revision_id")
