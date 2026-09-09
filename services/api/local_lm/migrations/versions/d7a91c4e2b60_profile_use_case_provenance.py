"""Record whether a profile use case came from provider metadata.

Revision ID: d7a91c4e2b60
Revises: c3e1d7a94f20
Create Date: 2026-09-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d7a91c4e2b60"
down_revision: str | None = "c3e1d7a94f20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_profiles",
        sa.Column("use_case_derived", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("model_profiles", "use_case_derived")
