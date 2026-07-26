"""persist project and chat generation defaults

Revision ID: a4d7c2e91b63
Revises: f7a2c9e51b40
Create Date: 2026-07-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4d7c2e91b63"
down_revision: str | None = "f7a2c9e51b40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for table in ("projects", "chats"):
        op.add_column(
            table,
            sa.Column(
                "generation_settings_json",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            ),
        )
        op.add_column(
            table,
            sa.Column(
                "generation_preset_ids_json",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            ),
        )


def downgrade() -> None:
    for table in ("chats", "projects"):
        op.drop_column(table, "generation_preset_ids_json")
        op.drop_column(table, "generation_settings_json")
