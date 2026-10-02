"""Keep which of a comparison's pictures the person preferred, each time they say.

Revision ID: 53c96f3f738a
Revises: 374ee9525865
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "53c96f3f738a"
down_revision: str | None = "374ee9525865"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "generation_experiment_evaluations",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("experiment_id", sa.String(length=40), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("preference", sa.String(length=16), nullable=False),
        sa.Column("preferred_arm_id", sa.String(length=40), nullable=True),
        sa.Column("note", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"], ["generation_experiments.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["preferred_arm_id"], ["generation_experiment_arms.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "experiment_id", "sequence", name="uq_generation_experiment_evaluation_sequence"
        ),
    )
    op.create_index(
        "ix_generation_experiment_evaluations_experiment_id",
        "generation_experiment_evaluations",
        ["experiment_id"],
    )


def downgrade() -> None:
    said = op.get_bind().execute(sa.text("SELECT 1 FROM generation_experiment_evaluations LIMIT 1"))
    if said.first():
        raise RuntimeError("Recorded comparison preferences require the current database version.")
    op.drop_index(
        "ix_generation_experiment_evaluations_experiment_id",
        table_name="generation_experiment_evaluations",
    )
    op.drop_table("generation_experiment_evaluations")
