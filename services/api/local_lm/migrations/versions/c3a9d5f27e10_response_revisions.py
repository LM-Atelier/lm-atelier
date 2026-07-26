"""add stable response revisions

Revision ID: c3a9d5f27e10
Revises: b8e4f2c17a90
Create Date: 2026-07-25
"""

from __future__ import annotations

import uuid
from collections import defaultdict

import sqlalchemy as sa
from alembic import op

revision: str = "c3a9d5f27e10"
down_revision: str | None = "b8e4f2c17a90"
branch_labels: str | None = None
depends_on: str | None = None


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def upgrade() -> None:
    with op.batch_alter_table("messages") as batch_op:
        batch_op.add_column(
            sa.Column(
                "transcript_visible",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
        batch_op.add_column(
            sa.Column("active_response_revision_id", sa.String(length=40), nullable=True)
        )
        batch_op.create_index(
            "ix_messages_transcript_visible",
            ["transcript_visible"],
            unique=False,
        )

    op.create_table(
        "response_revisions",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("message_id", sa.String(length=40), nullable=False),
        sa.Column("run_id", sa.String(length=40), nullable=True),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "message_id",
            "sequence",
            name="uq_response_revision_sequence",
        ),
        sa.UniqueConstraint("run_id", name="uq_response_revision_run"),
    )
    op.create_index(
        "ix_response_revisions_message_id",
        "response_revisions",
        ["message_id"],
        unique=False,
    )
    op.create_index(
        "ix_response_revisions_run_id",
        "response_revisions",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "ix_response_revisions_status",
        "response_revisions",
        ["status"],
        unique=False,
    )
    op.create_index(
        "uq_response_revision_pending_message",
        "response_revisions",
        ["message_id"],
        unique=True,
        sqlite_where=sa.text("status = 'pending'"),
    )
    op.create_table(
        "response_revision_parts",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("response_revision_id", sa.String(length=40), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("artifact_id", sa.String(length=80), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["response_revision_id"],
            ["response_revisions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "response_revision_id",
            "position",
            name="uq_response_revision_part_position",
        ),
    )
    op.create_index(
        "ix_response_revision_parts_response_revision_id",
        "response_revision_parts",
        ["response_revision_id"],
        unique=False,
    )

    connection = op.get_bind()
    assistant_rows = connection.execute(
        sa.text(
            """
            SELECT messages.id, messages.status, messages.created_at,
                   messages.updated_at, runs.id AS run_id
            FROM messages
            LEFT JOIN runs ON runs.assistant_message_id = messages.id
            WHERE messages.role = 'assistant'
            ORDER BY messages.created_at, messages.id
            """
        )
    ).mappings()
    revision_for_message: dict[str, str] = {}
    for row in assistant_rows:
        revision_id = _id("rev")
        revision_for_message[str(row["id"])] = revision_id
        connection.execute(
            sa.text(
                """
                INSERT INTO response_revisions
                    (id, message_id, run_id, sequence, status, created_at, updated_at)
                VALUES
                    (:id, :message_id, :run_id, 1, :status, :created_at, :updated_at)
                """
            ),
            {
                "id": revision_id,
                "message_id": row["id"],
                "run_id": row["run_id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            },
        )
        connection.execute(
            sa.text(
                """
                UPDATE messages
                SET active_response_revision_id = :revision_id
                WHERE id = :message_id
                """
            ),
            {"revision_id": revision_id, "message_id": row["id"]},
        )

    part_positions: dict[str, int] = defaultdict(int)
    part_rows = connection.execute(
        sa.text(
            """
            SELECT id, message_id, position, type, text, artifact_id, metadata_json,
                   created_at, updated_at
            FROM message_parts
            WHERE message_id IN (
                SELECT id FROM messages WHERE role = 'assistant'
            )
            ORDER BY message_id, position
            """
        )
    ).mappings()
    for row in part_rows:
        message_id = str(row["message_id"])
        backfill_revision_id = revision_for_message.get(message_id)
        if not backfill_revision_id:
            continue
        position = part_positions[backfill_revision_id]
        part_positions[backfill_revision_id] += 1
        connection.execute(
            sa.text(
                """
                INSERT INTO response_revision_parts
                    (id, response_revision_id, position, type, text, artifact_id,
                     metadata_json, created_at, updated_at)
                VALUES
                    (:id, :revision_id, :position, :type, :text, :artifact_id,
                     :metadata_json, :created_at, :updated_at)
                """
            ),
            {
                "id": _id("revpart"),
                "revision_id": backfill_revision_id,
                "position": position,
                "type": row["type"],
                "text": row["text"],
                "artifact_id": row["artifact_id"],
                "metadata_json": row["metadata_json"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            },
        )


def downgrade() -> None:
    op.drop_index(
        "ix_response_revision_parts_response_revision_id",
        table_name="response_revision_parts",
    )
    op.drop_table("response_revision_parts")
    op.drop_index(
        "uq_response_revision_pending_message",
        table_name="response_revisions",
    )
    op.drop_index("ix_response_revisions_status", table_name="response_revisions")
    op.drop_index("ix_response_revisions_run_id", table_name="response_revisions")
    op.drop_index("ix_response_revisions_message_id", table_name="response_revisions")
    op.drop_table("response_revisions")
    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_index("ix_messages_transcript_visible")
        batch_op.drop_column("active_response_revision_id")
        batch_op.drop_column("transcript_visible")
