"""Bind accepted extension plans to durable preparation jobs and exact installs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4c6e82b910f"
down_revision: str | None = "f3a98c72d104"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_install_offer_packages",
        sa.Column("id", sa.String(40), nullable=False),
        sa.Column("offer_id", sa.String(40), nullable=False),
        sa.Column("offer_sha256", sa.String(64), nullable=False),
        sa.Column("package_id", sa.String(100), nullable=False),
        sa.Column("execution_plan_sha256", sa.String(64), nullable=False),
        sa.Column("job_id", sa.String(40), nullable=True),
        sa.Column("registry_install_id", sa.String(64), nullable=True),
        sa.Column("preparation_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["offer_id"], ["workflow_install_offers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["registry_install_id"], ["comfy_registry_installs.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("offer_id", "package_id"),
    )
    for column in ("offer_id", "job_id", "registry_install_id"):
        op.create_index(
            "ix_workflow_install_offer_packages_" + column,
            "workflow_install_offer_packages",
            [column],
        )


def downgrade() -> None:
    if (
        op.get_bind()
        .execute(sa.text("SELECT 1 FROM workflow_install_offer_packages LIMIT 1"))
        .first()
    ):
        raise RuntimeError("Accepted extension preparations require the current database version.")
    for column in ("registry_install_id", "job_id", "offer_id"):
        op.drop_index(
            "ix_workflow_install_offer_packages_" + column,
            table_name="workflow_install_offer_packages",
        )
    op.drop_table("workflow_install_offer_packages")
