"""Store accepted conversation context and its retained media.

Revision ID: b8f31d0a6c42
Revises: b9c6d1e47a20
"""

import sqlalchemy as sa
from alembic import op

revision = "b8f31d0a6c42"
down_revision = "b9c6d1e47a20"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_context_snapshots",
        sa.Column(
            "run_id", sa.String(40), sa.ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
    )
    op.create_table(
        "run_context_artifacts",
        sa.Column(
            "run_id",
            sa.String(40),
            sa.ForeignKey("run_context_snapshots.run_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "artifact_id",
            sa.String(80),
            sa.ForeignKey("artifacts.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )
    op.create_index(
        "ix_run_context_artifacts_artifact_id", "run_context_artifacts", ["artifact_id"]
    )


def downgrade() -> None:
    op.drop_table("run_context_artifacts")
    op.drop_table("run_context_snapshots")
