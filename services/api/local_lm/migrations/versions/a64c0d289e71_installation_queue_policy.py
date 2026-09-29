"""Require installation-aware dispatch for retained queue and completion state."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a64c0d289e71"
down_revision: str | None = "ffccb898c9c6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Advance the version before installations can retain policies and durable claims."""
    # The existing lane tables and completion links already represent this state.


def downgrade() -> None:
    queries = (
        "SELECT 1 FROM generation_queue_policies WHERE lane = 'install' LIMIT 1",
        "SELECT 1 FROM generation_queue_receipts WHERE lane = 'install' LIMIT 1",
        "SELECT 1 FROM workflow_install_offers "
        "WHERE source_plan_id IS NULL AND completion_job_id IS NOT NULL LIMIT 1",
        "SELECT 1 FROM jobs WHERE kind IN ('activate', 'registry_prepare', 'workflow_install') "
        "AND (status IN ('queued', 'paused', 'running') OR claim_owner IS NOT NULL) LIMIT 1",
    )
    if any(op.get_bind().execute(sa.text(query)).first() for query in queries):
        raise RuntimeError("Installation queue controls require the current database version.")
