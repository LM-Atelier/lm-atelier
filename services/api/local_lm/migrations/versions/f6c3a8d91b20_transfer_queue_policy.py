"""Require transfer-aware dispatch before opening databases with queue controls."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f6c3a8d91b20"
down_revision: str | None = "e9a5c7d31b62"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Advance the version so older schedulers cannot ignore transfer policy rows."""
    # The existing tables already key policies and command receipts by lane.


def downgrade() -> None:
    if (
        op.get_bind()
        .execute(sa.text("SELECT 1 FROM generation_queue_policies WHERE lane = 'transfer' LIMIT 1"))
        .first()
    ):
        raise RuntimeError("Transfer queue controls require the current database version.")
