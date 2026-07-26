"""add work plans progress and durable dispatch fields

Revision ID: d4b7e2a61f90
Revises: c3a9d5f27e10
Create Date: 2026-07-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "d4b7e2a61f90"
down_revision: str | None = "c3a9d5f27e10"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "work_plans",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("chat_id", sa.String(length=40), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=True),
        sa.Column("source_action", sa.String(length=32), nullable=False),
        sa.Column("persistence_scope", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("context_head_message_id", sa.String(length=40), nullable=True),
        sa.Column("transcript_sequence", sa.Integer(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("planner_version", sa.String(length=32), nullable=False),
        sa.Column("failure_policy", sa.String(length=32), nullable=False),
        sa.Column(
            "summary_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["context_head_message_id"],
            ["messages.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "chat_id",
            "idempotency_key",
            name="uq_work_plan_chat_id_idempotency_key",
        ),
        sa.UniqueConstraint(
            "chat_id",
            "transcript_sequence",
            name="uq_work_plan_transcript_sequence",
        ),
    )
    op.create_index("ix_work_plans_chat_id", "work_plans", ["chat_id"], unique=False)
    op.create_index("ix_work_plans_status", "work_plans", ["status"], unique=False)

    op.create_table(
        "work_steps",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("plan_id", sa.String(length=40), nullable=False),
        sa.Column("run_id", sa.String(length=40), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("display_group", sa.String(length=80), nullable=True),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("profile_id", sa.String(length=40), nullable=True),
        sa.Column("workflow_revision_id", sa.String(length=40), nullable=True),
        sa.Column(
            "settings_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "input_bindings_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "output_contract_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("queue_class", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["plan_id"], ["work_plans.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("plan_id", "ordinal", name="uq_work_step_ordinal"),
        sa.UniqueConstraint("run_id", name="uq_work_step_run"),
    )
    op.create_index("ix_work_steps_plan_id", "work_steps", ["plan_id"], unique=False)
    op.create_index("ix_work_steps_run_id", "work_steps", ["run_id"], unique=False)
    op.create_index("ix_work_steps_status", "work_steps", ["status"], unique=False)

    op.create_table(
        "work_step_dependencies",
        sa.Column("step_id", sa.String(length=40), nullable=False),
        sa.Column("depends_on_step_id", sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(["step_id"], ["work_steps.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["depends_on_step_id"],
            ["work_steps.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("step_id", "depends_on_step_id"),
    )

    with op.batch_alter_table("runs") as batch_op:
        batch_op.add_column(sa.Column("work_plan_id", sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column("work_step_id", sa.String(length=40), nullable=True))
        batch_op.create_foreign_key(
            "fk_runs_work_plan_id",
            "work_plans",
            ["work_plan_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_runs_work_step_id",
            "work_steps",
            ["work_step_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index("ix_runs_work_plan_id", ["work_plan_id"], unique=False)
        batch_op.create_index("uq_runs_work_step_id", ["work_step_id"], unique=True)

    with op.batch_alter_table("jobs") as batch_op:
        batch_op.add_column(sa.Column("work_plan_id", sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column("work_step_id", sa.String(length=40), nullable=True))
        batch_op.add_column(
            sa.Column(
                "progress_json",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            )
        )
        batch_op.add_column(sa.Column("queue_resource", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("queue_group", sa.String(length=32), nullable=True))
        batch_op.add_column(
            sa.Column("queue_priority", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(sa.Column("queue_ticket", sa.String(length=80), nullable=True))
        batch_op.add_column(sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("claim_owner", sa.String(length=80), nullable=True))
        batch_op.add_column(
            sa.Column("claim_expires_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_foreign_key(
            "fk_jobs_work_plan_id",
            "work_plans",
            ["work_plan_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_jobs_work_step_id",
            "work_steps",
            ["work_step_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index("ix_jobs_work_plan_id", ["work_plan_id"], unique=False)
        batch_op.create_index("ix_jobs_work_step_id", ["work_step_id"], unique=False)
        batch_op.create_index("ix_jobs_queue_resource", ["queue_resource"], unique=False)
        batch_op.create_index("ix_jobs_queue_group", ["queue_group"], unique=False)
        batch_op.create_index("ix_jobs_queue_priority", ["queue_priority"], unique=False)
        batch_op.create_index("ix_jobs_queue_ticket", ["queue_ticket"], unique=False)
        batch_op.create_index("ix_jobs_claim_owner", ["claim_owner"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.drop_index("ix_jobs_claim_owner")
        batch_op.drop_index("ix_jobs_queue_ticket")
        batch_op.drop_index("ix_jobs_queue_priority")
        batch_op.drop_index("ix_jobs_queue_group")
        batch_op.drop_index("ix_jobs_queue_resource")
        batch_op.drop_index("ix_jobs_work_step_id")
        batch_op.drop_index("ix_jobs_work_plan_id")
        batch_op.drop_constraint("fk_jobs_work_step_id", type_="foreignkey")
        batch_op.drop_constraint("fk_jobs_work_plan_id", type_="foreignkey")
        batch_op.drop_column("heartbeat_at")
        batch_op.drop_column("claim_expires_at")
        batch_op.drop_column("claim_owner")
        batch_op.drop_column("enqueued_at")
        batch_op.drop_column("queue_ticket")
        batch_op.drop_column("queue_priority")
        batch_op.drop_column("queue_group")
        batch_op.drop_column("queue_resource")
        batch_op.drop_column("progress_json")
        batch_op.drop_column("work_step_id")
        batch_op.drop_column("work_plan_id")

    with op.batch_alter_table("runs") as batch_op:
        batch_op.drop_index("uq_runs_work_step_id")
        batch_op.drop_index("ix_runs_work_plan_id")
        batch_op.drop_constraint("fk_runs_work_step_id", type_="foreignkey")
        batch_op.drop_constraint("fk_runs_work_plan_id", type_="foreignkey")
        batch_op.drop_column("work_step_id")
        batch_op.drop_column("work_plan_id")

    op.drop_table("work_step_dependencies")
    op.drop_index("ix_work_steps_status", table_name="work_steps")
    op.drop_index("ix_work_steps_run_id", table_name="work_steps")
    op.drop_index("ix_work_steps_plan_id", table_name="work_steps")
    op.drop_table("work_steps")
    op.drop_index("ix_work_plans_status", table_name="work_plans")
    op.drop_index("ix_work_plans_chat_id", table_name="work_plans")
    op.drop_table("work_plans")
