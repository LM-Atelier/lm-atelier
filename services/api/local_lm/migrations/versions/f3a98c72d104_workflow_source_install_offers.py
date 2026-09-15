"""Bind accepted workflow offers to their preflighted source plans."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f3a98c72d104"
down_revision: str | None = "e48d903a7b26"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("workflow_install_offers") as batch:
        batch.add_column(sa.Column("source_plan_id", sa.String(40), nullable=True))
        batch.create_foreign_key(
            "fk_workflow_install_offer_source_plan",
            "workflow_package_install_plans",
            ["source_plan_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_unique_constraint(
            "uq_workflow_install_offer_source_plan_id", ["source_plan_id"]
        )
        batch.drop_constraint("ck_workflow_install_offer_plan_count", type_="check")
        batch.drop_constraint("ck_workflow_install_offer_total_bytes", type_="check")
        batch.create_check_constraint(
            "ck_workflow_install_offer_plan_count",
            "plan_count >= 0 AND (source_plan_id IS NOT NULL OR plan_count > 0)",
        )
        batch.create_check_constraint(
            "ck_workflow_install_offer_total_bytes",
            "total_bytes >= 0 AND (source_plan_id IS NOT NULL OR total_bytes > 0)",
        )


def downgrade() -> None:
    # A source-bound acceptance cannot be represented by the older offer schema.
    if (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT 1 FROM workflow_install_offers WHERE source_plan_id IS NOT NULL LIMIT 1"
            )
        )
        .first()
    ):
        raise RuntimeError("Accepted workflow source plans require the current database version.")
    with op.batch_alter_table("workflow_install_offers") as batch:
        batch.drop_constraint("ck_workflow_install_offer_plan_count", type_="check")
        batch.drop_constraint("ck_workflow_install_offer_total_bytes", type_="check")
        batch.drop_constraint("uq_workflow_install_offer_source_plan_id", type_="unique")
        batch.drop_constraint("fk_workflow_install_offer_source_plan", type_="foreignkey")
        batch.drop_column("source_plan_id")
        batch.create_check_constraint("ck_workflow_install_offer_plan_count", "plan_count > 0")
        batch.create_check_constraint("ck_workflow_install_offer_total_bytes", "total_bytes > 0")
