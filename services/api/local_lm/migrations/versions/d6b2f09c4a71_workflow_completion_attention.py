"""Keep content-free workflow installation failures across reloads."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d6b2f09c4a71"
down_revision: str | None = "c7d4a1e83b60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "workflow_install_offers", sa.Column("completion_error_code", sa.String(80), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("workflow_install_offers", "completion_error_code")
