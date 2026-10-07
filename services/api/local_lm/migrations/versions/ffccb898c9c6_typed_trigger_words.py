"""Keep the trigger words a person records for a LoRA apart from measured ones."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "ffccb898c9c6"
down_revision: str | None = "c8a4f72e91bd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_asset_installs",
        sa.Column("typed_trigger_words", sa.JSON(), nullable=False, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("model_asset_installs", "typed_trigger_words")
