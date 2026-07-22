"""pin immutable media workflow revisions to projects

Revision ID: d1e7a4b8c920
Revises: 9c4d2a7e1f30
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d1e7a4b8c920"
down_revision: str | None = "9c4d2a7e1f30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.add_column(sa.Column("image_workflow_revision_id", sa.String(length=40)))
        batch_op.add_column(sa.Column("video_workflow_revision_id", sa.String(length=40)))
        batch_op.create_foreign_key(
            "fk_projects_image_workflow_revision",
            "workflow_revisions",
            ["image_workflow_revision_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_projects_video_workflow_revision",
            "workflow_revisions",
            ["video_workflow_revision_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_constraint("fk_projects_video_workflow_revision", type_="foreignkey")
        batch_op.drop_constraint("fk_projects_image_workflow_revision", type_="foreignkey")
        batch_op.drop_column("video_workflow_revision_id")
        batch_op.drop_column("image_workflow_revision_id")
