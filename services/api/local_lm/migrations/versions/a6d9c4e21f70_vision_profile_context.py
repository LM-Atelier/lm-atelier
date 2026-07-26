"""add vision profile context

Revision ID: a6d9c4e21f70
Revises: e5c8a1b7f240
Create Date: 2026-07-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "a6d9c4e21f70"
down_revision: str | None = "e5c8a1b7f240"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    with op.batch_alter_table("chats") as batch_op:
        batch_op.add_column(
            sa.Column(
                "active_vision_profile_id",
                sa.String(length=40),
                server_default="__auto__",
            )
        )
        batch_op.add_column(
            sa.Column(
                "vision_settings_json",
                sa.JSON(),
                nullable=False,
                server_default=sa.text(
                    """'{"max_images": 4, "max_video_frames": 6, """
                    """\"include_prior_visual\": true}'"""
                ),
            )
        )
    with op.batch_alter_table("chats") as batch_op:
        batch_op.alter_column("active_vision_profile_id", server_default=None)
        batch_op.alter_column("vision_settings_json", server_default=None)
    with op.batch_alter_table("runs") as batch_op:
        batch_op.add_column(sa.Column("vision_profile_id", sa.String(length=40)))


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch_op:
        batch_op.drop_column("vision_profile_id")
    with op.batch_alter_table("chats") as batch_op:
        batch_op.drop_column("vision_settings_json")
        batch_op.drop_column("active_vision_profile_id")
