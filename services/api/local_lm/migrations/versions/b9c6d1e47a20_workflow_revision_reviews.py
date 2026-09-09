"""Record explicit approval of exact workflow revisions and node identities."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "b9c6d1e47a20"
down_revision: str | None = "a2c7f9d31b60"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_revision_reviews",
        sa.Column("workflow_revision_id", sa.String(40), primary_key=True),
        sa.Column("revision_sha256", sa.String(64), nullable=False),
        sa.Column("subject_sha256", sa.String(64), nullable=False),
        sa.Column("node_bindings_json", sa.JSON(), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["workflow_revision_id"], ["workflow_revisions.id"], ondelete="CASCADE"
        ),
    )


def downgrade() -> None:
    # Older builds cannot revalidate this authority. Dropping its evidence must
    # never leave the cached trust Boolean granting execution on its own.
    op.execute(
        "UPDATE workflow_revisions SET trusted = 0 "
        "WHERE id IN (SELECT workflow_revision_id FROM workflow_revision_reviews)"
    )
    op.drop_table("workflow_revision_reviews")
