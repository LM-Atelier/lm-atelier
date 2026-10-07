"""Retain the durable job for each accepted workflow installation."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e9a5c7d31b62"
down_revision: str | None = "a4c6e82b910f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("workflow_install_offers") as batch:
        batch.add_column(sa.Column("completion_job_id", sa.String(40), nullable=True))
        batch.create_foreign_key(
            "fk_workflow_install_offer_completion_job",
            "jobs",
            ["completion_job_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_unique_constraint(
            "uq_workflow_install_offer_completion_job_id", ["completion_job_id"]
        )


def downgrade() -> None:
    if (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT 1 FROM workflow_install_offers WHERE completion_job_id IS NOT NULL LIMIT 1"
            )
        )
        .first()
    ):
        raise RuntimeError("Workflow installation jobs require the current database version.")
    with op.batch_alter_table("workflow_install_offers") as batch:
        batch.drop_constraint("uq_workflow_install_offer_completion_job_id", type_="unique")
        batch.drop_constraint("fk_workflow_install_offer_completion_job", type_="foreignkey")
        batch.drop_column("completion_job_id")
