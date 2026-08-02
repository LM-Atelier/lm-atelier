"""widen generated model identifiers

Revision ID: a4f6d2c91e70
Revises: e6c42a9b13fd
Create Date: 2026-08-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4f6d2c91e70"
down_revision: str | None = "e6c42a9b13fd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("model_component_manifests") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.String(40),
            type_=sa.String(64),
            existing_nullable=False,
        )
    with op.batch_alter_table("model_capability_evidence") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.String(40),
            type_=sa.String(64),
            existing_nullable=False,
        )
    with op.batch_alter_table("workflow_definitions") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.String(40),
            type_=sa.String(64),
            existing_nullable=False,
        )
    with op.batch_alter_table("workflow_revisions") as batch_op:
        batch_op.alter_column(
            "workflow_id",
            existing_type=sa.String(40),
            type_=sa.String(64),
            existing_nullable=False,
        )
    with op.batch_alter_table("comfy_registry_installs") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.String(40),
            type_=sa.String(64),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("comfy_registry_installs") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.String(64),
            type_=sa.String(40),
            existing_nullable=False,
        )
    with op.batch_alter_table("workflow_revisions") as batch_op:
        batch_op.alter_column(
            "workflow_id",
            existing_type=sa.String(64),
            type_=sa.String(40),
            existing_nullable=False,
        )
    with op.batch_alter_table("workflow_definitions") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.String(64),
            type_=sa.String(40),
            existing_nullable=False,
        )
    with op.batch_alter_table("model_capability_evidence") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.String(64),
            type_=sa.String(40),
            existing_nullable=False,
        )
    with op.batch_alter_table("model_component_manifests") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.String(64),
            type_=sa.String(40),
            existing_nullable=False,
        )
