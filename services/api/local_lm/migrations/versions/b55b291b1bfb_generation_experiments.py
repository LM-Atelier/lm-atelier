"""Keep comparisons of two generation choices against one frozen request.

Revision ID: b55b291b1bfb
Revises: a7d4e9b62c10
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b55b291b1bfb"
down_revision: str | None = "a7d4e9b62c10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "generation_experiments",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("operation", sa.String(32), nullable=False),
        sa.Column("contract_version", sa.Integer(), nullable=False),
        sa.Column("app_version", sa.String(32), nullable=False),
        sa.Column("seed_policy", sa.String(32), nullable=False),
        sa.Column("seed_equivalence", sa.String(16), nullable=False),
        sa.Column("common_json", sa.JSON(), nullable=False),
        sa.Column("estimate_json", sa.JSON(), nullable=False),
        sa.Column("preflight_sha256", sa.String(64), nullable=False),
        sa.Column("snapshot_sha256", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("idempotency_key", name="uq_generation_experiment_idempotency"),
    )
    op.create_table(
        "generation_experiment_arms",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column(
            "experiment_id",
            sa.String(40),
            sa.ForeignKey("generation_experiments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(80), nullable=False),
        sa.Column("profile_id", sa.String(40), nullable=False),
        sa.Column("workflow_revision_id", sa.String(40), nullable=False),
        sa.Column("workflow_activation_id", sa.String(40), nullable=True),
        sa.Column("model_family", sa.String(120), nullable=True),
        sa.Column("requested_settings_json", sa.JSON(), nullable=False),
        sa.Column("effective_settings_json", sa.JSON(), nullable=False),
        sa.Column("snapshot_json", sa.JSON(), nullable=False),
        sa.Column("snapshot_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "experiment_id", "ordinal", name="uq_generation_experiment_arm_ordinal"
        ),
        sa.UniqueConstraint("experiment_id", "label", name="uq_generation_experiment_arm_label"),
    )
    op.create_table(
        "generation_experiment_trials",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column(
            "arm_id",
            sa.String(40),
            sa.ForeignKey("generation_experiment_arms.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("arm_id", "ordinal", name="uq_generation_experiment_trial_ordinal"),
    )


def downgrade() -> None:
    if op.get_bind().execute(sa.text("SELECT 1 FROM generation_experiments LIMIT 1")).first():
        raise RuntimeError("Generation comparisons require the current database version.")
    op.drop_table("generation_experiment_trials")
    op.drop_table("generation_experiment_arms")
    op.drop_table("generation_experiments")
