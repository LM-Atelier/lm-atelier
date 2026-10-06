"""Require a scheduler that knows the video utility lane and its job kind."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "2a07d89a3b46"
down_revision: str | None = "9c4e7a2d1830"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Advance the version before utility jobs and their lane policy can be stored."""
    # The existing job and lane tables already hold this state.


def downgrade() -> None:
    queries = (
        "SELECT 1 FROM generation_queue_policies WHERE lane = 'utility' LIMIT 1",
        "SELECT 1 FROM generation_queue_receipts WHERE lane = 'utility' LIMIT 1",
        # An earlier version cannot describe any utility job, finished or not.
        "SELECT 1 FROM jobs WHERE kind = 'media_utility' LIMIT 1",
    )
    if any(op.get_bind().execute(sa.text(query)).first() for query in queries):
        raise RuntimeError("Video utility jobs require the current database version.")
