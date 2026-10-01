"""Keep each viewing of a blind comparison: its order of the pictures and its saying.

Revision ID: fbc390fda45f
Revises: 53c96f3f738a
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "fbc390fda45f"
down_revision: str | None = "53c96f3f738a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "generation_experiment_blind_views",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("experiment_id", sa.String(length=40), nullable=False),
        sa.Column("order_json", sa.JSON(), nullable=False),
        sa.Column("evaluation_id", sa.String(length=40), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"], ["generation_experiments.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["evaluation_id"], ["generation_experiment_evaluations.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("evaluation_id"),
    )
    op.create_index(
        "ix_generation_experiment_blind_views_experiment_id",
        "generation_experiment_blind_views",
        ["experiment_id"],
    )


def downgrade() -> None:
    viewed = op.get_bind().execute(
        sa.text("SELECT 1 FROM generation_experiment_blind_views LIMIT 1")
    )
    if viewed.first():
        raise RuntimeError("Blind comparison views require the current database version.")
    op.drop_index(
        "ix_generation_experiment_blind_views_experiment_id",
        table_name="generation_experiment_blind_views",
    )
    op.drop_table("generation_experiment_blind_views")
