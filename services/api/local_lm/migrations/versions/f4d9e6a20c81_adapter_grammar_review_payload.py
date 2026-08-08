"""bind a grammar review to every input that changes emitted text

Revision ID: f4d9e6a20c81
Revises: e2b8c5d10f37
Create Date: 2026-08-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f4d9e6a20c81"
down_revision: str | None = "e2b8c5d10f37"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("adapter_prompt_grammars") as batch:
        # The digest of what a rewriter would act on, overlays included. Without
        # it, widening approval or verification changes the emitted text while
        # nothing review is bound to has moved.
        batch.add_column(
            sa.Column("grammar_sha256", sa.String(length=64), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column("approved_prose_json", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.add_column(
            sa.Column("verified_values_json", sa.JSON(), nullable=False, server_default="{}")
        )
        # What the fit was judged against, so a compiler change expires the
        # evidence instead of overflowing at turn time.
        batch.add_column(
            sa.Column("compiler_version", sa.String(length=64), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column("compiler_ceiling", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("fits", sa.Boolean(), nullable=False, server_default=sa.false()))
        # Superseded by approved_prose_json, which is content-bound rather than a
        # flag. Keeping both would let them disagree about the same question.
        batch.drop_column("examples_reviewed")


def downgrade() -> None:
    with op.batch_alter_table("adapter_prompt_grammars") as batch:
        batch.add_column(
            sa.Column("examples_reviewed", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch.drop_column("fits")
        batch.drop_column("compiler_ceiling")
        batch.drop_column("compiler_version")
        batch.drop_column("verified_values_json")
        batch.drop_column("approved_prose_json")
        batch.drop_column("grammar_sha256")
