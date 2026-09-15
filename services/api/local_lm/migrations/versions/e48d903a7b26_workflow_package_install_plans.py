"""Preserve workflow source and dependency plans before approval."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e48d903a7b26"
down_revision: str | None = "d6b2f09c4a71"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _sha256_check() -> str:
    remainder = "plan_sha256"
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length(plan_sha256) = 64 AND lower(plan_sha256) = plan_sha256 AND {remainder} = ''"


def upgrade() -> None:
    op.create_table(
        "workflow_package_install_plans",
        sa.Column("id", sa.String(40), nullable=False),
        sa.Column("plan_sha256", sa.String(64), nullable=False),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("preflight_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            _sha256_check(),
            name="ck_workflow_package_install_plan_sha256",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("plan_sha256", name="uq_workflow_package_install_plan_sha256"),
    )


def downgrade() -> None:
    op.drop_table("workflow_package_install_plans")
