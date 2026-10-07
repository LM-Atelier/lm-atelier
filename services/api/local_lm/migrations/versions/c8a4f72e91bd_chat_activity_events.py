"""Persist completion identities with the response snapshots they describe."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c8a4f72e91bd"
down_revision: str | None = "f6c3a8d91b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("response_revisions", sa.Column("activity_json", sa.JSON(), nullable=True))
    op.create_table(
        "chat_activity_events",
        sa.Column("sequence", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("id", sa.String(40), nullable=False, unique=True),
        sa.Column(
            "chat_id", sa.String(40), sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "message_id",
            sa.String(40),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "response_revision_id",
            sa.String(40),
            sa.ForeignKey("response_revisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("job_id", sa.String(40), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("job_id", "attempt", name="uq_chat_activity_attempt"),
        sqlite_autoincrement=True,
    )
    op.create_index(
        "ix_chat_activity_chat_sequence", "chat_activity_events", ["chat_id", "sequence"]
    )
    op.create_index("ix_chat_activity_events_message_id", "chat_activity_events", ["message_id"])
    op.create_index(
        "ix_chat_activity_events_response_revision_id",
        "chat_activity_events",
        ["response_revision_id"],
    )


def downgrade() -> None:
    if op.get_bind().execute(sa.text("SELECT 1 FROM chat_activity_events LIMIT 1")).first():
        raise RuntimeError("Chat activity requires the current database version.")
    op.drop_table("chat_activity_events")
    op.drop_column("response_revisions", "activity_json")
