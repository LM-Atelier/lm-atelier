"""Keep the chosen purpose of each picture in an unsent draft.

Revision ID: e5a27b3d9061
Revises: 9c4e7a2d1830
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from local_lm.chat_recovery_schema import CREATE_CHAT_RECOVERY_TRIGGER_SQL

revision: str = "e5a27b3d9061"
down_revision: str | None = "9c4e7a2d1830"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_composer_draft_attachments",
        sa.Column("image_role", sa.String(length=16), nullable=True),
    )


def downgrade() -> None:
    if (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT 1 FROM chat_composer_draft_attachments WHERE image_role IS NOT NULL LIMIT 1"
            )
        )
        .first()
        is not None
    ):
        raise RuntimeError("Picture purposes must be cleared before downgrading.")
    with op.batch_alter_table("chat_composer_draft_attachments") as batch:
        batch.drop_column("image_role")
    for statement in CREATE_CHAT_RECOVERY_TRIGGER_SQL:
        if statement.startswith("CREATE TRIGGER chat_recovery_chat_composer_draft_attachments_"):
            op.execute(statement)
