"""Add durable content tombstones for individual chat messages.

Revision ID: c5a8e1d72f40
Revises: f2a7c9d41e63
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "c5a8e1d72f40"
down_revision: str | None = "f2a7c9d41e63"
branch_labels: str | None = None
depends_on: str | None = None

_CREATE_TRIGGER_SQL = (
    """
CREATE TRIGGER chat_item_content_tombstone_update_guard
BEFORE UPDATE OF content_removed_at ON messages
BEGIN
  SELECT CASE WHEN OLD.content_removed_at IS NOT NULL
                        AND NEW.content_removed_at IS NOT OLD.content_removed_at
    THEN RAISE(ABORT, 'chat item content tombstone is immutable') END;
  SELECT CASE WHEN OLD.content_removed_at IS NULL
                        AND NEW.content_removed_at IS NOT NULL
                        AND (
                          EXISTS (
                            SELECT 1 FROM message_parts
                            WHERE message_id = OLD.id
                          )
                          OR EXISTS (
                            SELECT 1 FROM message_references
                            WHERE message_id = OLD.id
                          )
                          OR EXISTS (
                            SELECT 1
                            FROM response_revision_parts AS part
                            JOIN response_revisions AS revision
                              ON revision.id = part.response_revision_id
                            WHERE revision.message_id = OLD.id
                          )
                        )
    THEN RAISE(ABORT, 'chat item payload must be detached before tombstoning') END;
END
""",
    """
CREATE TRIGGER chat_item_content_tombstone_message_part_guard
BEFORE INSERT ON message_parts
WHEN EXISTS (
  SELECT 1 FROM messages
  WHERE id = NEW.message_id AND content_removed_at IS NOT NULL
)
BEGIN
  SELECT RAISE(ABORT, 'removed chat item cannot receive message parts');
END
""",
    """
CREATE TRIGGER chat_item_content_tombstone_message_part_reparent_guard
BEFORE UPDATE OF message_id ON message_parts
WHEN EXISTS (
  SELECT 1 FROM messages
  WHERE id = NEW.message_id AND content_removed_at IS NOT NULL
)
BEGIN
  SELECT RAISE(ABORT, 'removed chat item cannot receive message parts');
END
""",
    """
CREATE TRIGGER chat_item_content_tombstone_revision_part_guard
BEFORE INSERT ON response_revision_parts
WHEN EXISTS (
  SELECT 1
  FROM response_revisions AS revision
  JOIN messages AS message ON message.id = revision.message_id
  WHERE revision.id = NEW.response_revision_id
    AND message.content_removed_at IS NOT NULL
)
BEGIN
  SELECT RAISE(ABORT, 'removed chat item cannot receive revision parts');
END
""",
    """
CREATE TRIGGER chat_item_content_tombstone_revision_part_reparent_guard
BEFORE UPDATE OF response_revision_id ON response_revision_parts
WHEN EXISTS (
  SELECT 1
  FROM response_revisions AS revision
  JOIN messages AS message ON message.id = revision.message_id
  WHERE revision.id = NEW.response_revision_id
    AND message.content_removed_at IS NOT NULL
)
BEGIN
  SELECT RAISE(ABORT, 'removed chat item cannot receive revision parts');
END
""",
    """
CREATE TRIGGER chat_item_content_tombstone_revision_reparent_guard
BEFORE UPDATE OF message_id ON response_revisions
WHEN EXISTS (
  SELECT 1 FROM messages
  WHERE id = NEW.message_id AND content_removed_at IS NOT NULL
)
AND EXISTS (
  SELECT 1 FROM response_revision_parts
  WHERE response_revision_id = OLD.id
)
BEGIN
  SELECT RAISE(ABORT, 'removed chat item cannot receive a populated revision');
END
""",
    """
CREATE TRIGGER chat_item_content_tombstone_reference_guard
BEFORE INSERT ON message_references
WHEN EXISTS (
  SELECT 1 FROM messages
  WHERE id = NEW.message_id AND content_removed_at IS NOT NULL
)
BEGIN
  SELECT RAISE(ABORT, 'removed chat item cannot receive references');
END
""",
    """
CREATE TRIGGER chat_item_content_tombstone_reference_reparent_guard
BEFORE UPDATE OF message_id ON message_references
WHEN EXISTS (
  SELECT 1 FROM messages
  WHERE id = NEW.message_id AND content_removed_at IS NOT NULL
)
BEGIN
  SELECT RAISE(ABORT, 'removed chat item cannot receive references');
END
""",
)

_DROP_TRIGGER_SQL = tuple(
    f"DROP TRIGGER {name}"
    for name in (
        "chat_item_content_tombstone_reference_reparent_guard",
        "chat_item_content_tombstone_reference_guard",
        "chat_item_content_tombstone_revision_part_reparent_guard",
        "chat_item_content_tombstone_revision_part_guard",
        "chat_item_content_tombstone_revision_reparent_guard",
        "chat_item_content_tombstone_message_part_reparent_guard",
        "chat_item_content_tombstone_message_part_guard",
        "chat_item_content_tombstone_update_guard",
    )
)


def upgrade() -> None:
    with op.batch_alter_table("messages") as batch:
        batch.add_column(sa.Column("content_removed_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_index(
            "ix_messages_content_removed_at",
            ["content_removed_at"],
            unique=False,
        )
    for statement in _CREATE_TRIGGER_SQL:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DROP_TRIGGER_SQL:
        op.execute(statement)
    with op.batch_alter_table("messages") as batch:
        batch.drop_index("ix_messages_content_removed_at")
        batch.drop_column("content_removed_at")
