"""Link a started comparison to the work plan, steps and runs that make its pictures.

Revision ID: 374ee9525865
Revises: b55b291b1bfb
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "374ee9525865"
down_revision: str | None = "b55b291b1bfb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("generation_experiments") as batch:
        batch.add_column(sa.Column("work_plan_id", sa.String(length=40), nullable=True))
        batch.add_column(sa.Column("start_idempotency_key", sa.String(length=200), nullable=True))
        batch.add_column(sa.Column("start_request_sha256", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_foreign_key(
            "fk_generation_experiments_work_plan_id",
            "work_plans",
            ["work_plan_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_unique_constraint("uq_generation_experiment_work_plan_id", ["work_plan_id"])
    with op.batch_alter_table("generation_experiment_trials") as batch:
        batch.add_column(sa.Column("work_step_id", sa.String(length=40), nullable=True))
        batch.add_column(sa.Column("run_id", sa.String(length=40), nullable=True))
        batch.create_foreign_key(
            "fk_generation_experiment_trials_work_step_id",
            "work_steps",
            ["work_step_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_foreign_key(
            "fk_generation_experiment_trials_run_id",
            "runs",
            ["run_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_unique_constraint(
            "uq_generation_experiment_trial_work_step_id", ["work_step_id"]
        )
        batch.create_unique_constraint("uq_generation_experiment_trial_run_id", ["run_id"])


def downgrade() -> None:
    started = op.get_bind().execute(
        sa.text(
            "SELECT 1 FROM generation_experiments "
            "WHERE state != 'ready' OR work_plan_id IS NOT NULL LIMIT 1"
        )
    )
    if started.first():
        raise RuntimeError("Started generation comparisons require the current database version.")
    with op.batch_alter_table("generation_experiment_trials") as batch:
        batch.drop_constraint("uq_generation_experiment_trial_run_id", type_="unique")
        batch.drop_constraint("uq_generation_experiment_trial_work_step_id", type_="unique")
        batch.drop_constraint("fk_generation_experiment_trials_run_id", type_="foreignkey")
        batch.drop_constraint("fk_generation_experiment_trials_work_step_id", type_="foreignkey")
        batch.drop_column("run_id")
        batch.drop_column("work_step_id")
    with op.batch_alter_table("generation_experiments") as batch:
        batch.drop_constraint("uq_generation_experiment_work_plan_id", type_="unique")
        batch.drop_constraint("fk_generation_experiments_work_plan_id", type_="foreignkey")
        batch.drop_column("started_at")
        batch.drop_column("start_request_sha256")
        batch.drop_column("start_idempotency_key")
        batch.drop_column("work_plan_id")
