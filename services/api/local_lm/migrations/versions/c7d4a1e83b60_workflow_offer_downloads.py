"""Persist the jobs and accepted requests of workflow install offers."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7d4a1e83b60"
down_revision: str | None = "b8e41d6f2c93"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_install_offer_downloads",
        sa.Column("id", sa.String(40), nullable=False),
        sa.Column("offer_id", sa.String(40), nullable=False),
        sa.Column("offer_sha256", sa.String(64), nullable=False),
        sa.Column("job_id", sa.String(40), nullable=True),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["offer_id"], ["workflow_install_offers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("offer_id", "request_sha256"),
    )
    op.create_index(
        "ix_workflow_install_offer_downloads_offer_id",
        "workflow_install_offer_downloads",
        ["offer_id"],
    )
    op.create_index(
        "ix_workflow_install_offer_downloads_job_id", "workflow_install_offer_downloads", ["job_id"]
    )


def downgrade() -> None:
    op.drop_index(
        "ix_workflow_install_offer_downloads_job_id", table_name="workflow_install_offer_downloads"
    )
    op.drop_index(
        "ix_workflow_install_offer_downloads_offer_id",
        table_name="workflow_install_offer_downloads",
    )
    op.drop_table("workflow_install_offer_downloads")
