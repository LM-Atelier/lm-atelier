"""favorite flag on artifacts

Revision ID: c2e9f4a71d58
Revises: a4f6d2c91e70
Create Date: 2026-08-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c2e9f4a71d58"
down_revision: str | None = "a4f6d2c91e70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("artifacts") as batch:
        batch.add_column(
            sa.Column("favorite", sa.Boolean(), nullable=False, server_default=sa.false())
        )
    op.create_index("ix_artifacts_favorite", "artifacts", ["favorite"])


def downgrade() -> None:
    op.drop_index("ix_artifacts_favorite", table_name="artifacts")
    with op.batch_alter_table("artifacts") as batch:
        batch.drop_column("favorite")
