"""Record what a machine offered when it proved a capability.

Revision ID: a3f61c2d8be7
Revises: d4b81c37e905
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a3f61c2d8be7"
down_revision = "d4b81c37e905"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Deliberately not backfilled. The envelope describes the machine that ran
    # the probe, and that cannot be reconstructed from a hash of it. Existing
    # rows keep comparing `hardware_class` for equality, exactly as before, and
    # gain an envelope the next time they are proven.
    with op.batch_alter_table("model_capability_evidence") as batch:
        batch.add_column(sa.Column("hardware_envelope_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("model_capability_evidence") as batch:
        batch.drop_column("hardware_envelope_json")
