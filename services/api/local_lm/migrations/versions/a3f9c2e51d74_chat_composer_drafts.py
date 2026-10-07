"""Keep each chat's unsent composer draft, and the files it holds.

Revision ID: a3f9c2e51d74
Revises: 7d458a299547
Create Date: 2026-09-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3f9c2e51d74"
down_revision: str | None = "7d458a299547"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Two new tables and nothing else: no existing row or column is touched, so
    # the upgrade cannot lose data and the downgrade only removes what this adds.
    op.create_table(
        "chat_composer_drafts",
        sa.Column("chat_id", sa.String(length=40), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("prompt_source_json", sa.JSON(), nullable=True),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("output_count", sa.Integer(), nullable=False),
        sa.Column("mentions_json", sa.JSON(), nullable=False),
        sa.Column("template_settings_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chat_id"),
    )
    op.create_table(
        "chat_composer_draft_attachments",
        sa.Column("chat_id", sa.String(length=40), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("artifact_id", sa.String(length=80), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("origin", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(["chat_id"], ["chat_composer_drafts.chat_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("chat_id", "position"),
        sa.UniqueConstraint("chat_id", "artifact_id", name="uq_chat_composer_draft_attachment"),
    )
    # The artifact foreign key needs its own index, or every artifact deletion
    # scans this table to find referrers.
    op.create_index(
        op.f("ix_chat_composer_draft_attachments_artifact_id"),
        "chat_composer_draft_attachments",
        ["artifact_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_chat_composer_draft_attachments_artifact_id"),
        table_name="chat_composer_draft_attachments",
    )
    op.drop_table("chat_composer_draft_attachments")
    op.drop_table("chat_composer_drafts")
