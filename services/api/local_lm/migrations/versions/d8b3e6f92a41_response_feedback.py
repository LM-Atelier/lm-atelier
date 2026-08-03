"""local preference feedback on responses

Revision ID: d8b3e6f92a41
Revises: c2e9f4a71d58
Create Date: 2026-08-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d8b3e6f92a41"
down_revision: str | None = "c2e9f4a71d58"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "response_feedback",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column(
            "message_id",
            sa.String(length=40),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "response_revision_id",
            sa.String(length=40),
            sa.ForeignKey("response_revisions.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "run_id",
            sa.String(length=40),
            sa.ForeignKey("runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("rating", sa.String(length=8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "message_id", "response_revision_id", name="uq_response_feedback_target"
        ),
    )
    op.create_index("ix_response_feedback_message_id", "response_feedback", ["message_id"])
    op.create_index(
        "ix_response_feedback_response_revision_id", "response_feedback", ["response_revision_id"]
    )
    op.create_index("ix_response_feedback_run_id", "response_feedback", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_response_feedback_run_id", table_name="response_feedback")
    op.drop_index("ix_response_feedback_response_revision_id", table_name="response_feedback")
    op.drop_index("ix_response_feedback_message_id", table_name="response_feedback")
    op.drop_table("response_feedback")
