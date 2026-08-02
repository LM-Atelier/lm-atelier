"""bind Registry installs to exact wheel environments

Revision ID: e6c42a9b13fd
Revises: f3a8d1c72e60
Create Date: 2026-08-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e6c42a9b13fd"
down_revision: str | None = "f3a8d1c72e60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("comfy_registry_installs") as batch_op:
        batch_op.add_column(sa.Column("wheel_closure_sha256", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("wheel_environment_sha256", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("wheel_environment_path", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("comfy_registry_installs") as batch_op:
        batch_op.drop_column("wheel_environment_path")
        batch_op.drop_column("wheel_environment_sha256")
        batch_op.drop_column("wheel_closure_sha256")
