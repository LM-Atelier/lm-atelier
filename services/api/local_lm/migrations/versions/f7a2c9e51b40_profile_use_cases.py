"""add profile use cases and simplify default names

Revision ID: f7a2c9e51b40
Revises: e2f8b5c9d031
Create Date: 2026-07-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f7a2c9e51b40"
down_revision: str | None = "e2f8b5c9d031"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_profiles",
        sa.Column("use_case", sa.Text(), nullable=False, server_default=""),
    )
    op.execute(
        """
        UPDATE model_profiles
        SET name = 'Default'
        WHERE is_default = 1
          AND model_install_id IS NULL
          AND name IN ('Default chat', 'Default image', 'Default video')
        """
    )


def downgrade() -> None:
    op.drop_column("model_profiles", "use_case")
