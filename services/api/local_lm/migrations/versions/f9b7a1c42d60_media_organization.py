"""add manual Media Library collections and normalized tags

Revision ID: f9b7a1c42d60
Revises: e8a4c1d73b20
Create Date: 2026-08-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from local_lm.media_organization_schema import (
    CREATE_MEDIA_ORGANIZATION_TRIGGER_SQL,
    DROP_MEDIA_ORGANIZATION_TRIGGER_SQL,
)

revision: str = "f9b7a1c42d60"
down_revision: str | None = "e8a4c1d73b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _begin_write_fence() -> None:
    connection = op.get_bind()
    if connection.dialect.name != "sqlite":
        return
    driver = connection.connection.driver_connection
    if not bool(getattr(driver, "in_transaction", False)):
        connection.exec_driver_sql("BEGIN IMMEDIATE")
    else:
        connection.exec_driver_sql("UPDATE alembic_version SET version_num = version_num")


def upgrade() -> None:
    _begin_write_fence()
    op.create_table(
        "media_collections",
        sa.Column("id", sa.String(length=43), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("kind = 'manual'", name="ck_media_collection_kind"),
        sa.CheckConstraint(
            "length(name) BETWEEN 1 AND 200 AND name = trim(name) AND instr(name, char(0)) = 0",
            name="ck_media_collection_name",
        ),
        sa.CheckConstraint(
            "length(description) <= 2000 AND instr(description, char(0)) = 0",
            name="ck_media_collection_description",
        ),
        sa.CheckConstraint("version > 0", name="ck_media_collection_version_positive"),
    )
    op.create_table(
        "media_tags",
        sa.Column("id", sa.String(length=41), nullable=False),
        sa.Column("normalized_name", sa.String(length=80), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=False),
        sa.Column("color", sa.String(length=7), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_name", name="uq_media_tag_normalized_name"),
        sa.CheckConstraint(
            "length(normalized_name) BETWEEN 1 AND 80 "
            "AND normalized_name NOT GLOB '*[^a-z0-9-]*' "
            "AND normalized_name NOT LIKE '-%' AND normalized_name NOT LIKE '%-' "
            "AND normalized_name NOT LIKE '%--%'",
            name="ck_media_tag_normalized_name",
        ),
        sa.CheckConstraint(
            "length(label) BETWEEN 1 AND 200 AND label = trim(label) AND instr(label, char(0)) = 0",
            name="ck_media_tag_label",
        ),
        sa.CheckConstraint(
            "color IS NULL OR (length(color) = 7 AND substr(color, 1, 1) = '#' "
            "AND substr(color, 2) NOT GLOB '*[^0-9a-f]*')",
            name="ck_media_tag_color",
        ),
        sa.CheckConstraint("version > 0", name="ck_media_tag_version_positive"),
    )
    op.create_table(
        "media_collection_memberships",
        sa.Column("collection_id", sa.String(length=43), nullable=False),
        sa.Column("entry_id", sa.String(length=80), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("note", sa.String(length=1000), nullable=True),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["media_collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["entry_id"], ["artifact_library_entries.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("collection_id", "entry_id"),
        sa.UniqueConstraint(
            "collection_id", "position", name="uq_media_collection_membership_position"
        ),
        sa.CheckConstraint("position >= 0", name="ck_media_collection_membership_position"),
        sa.CheckConstraint(
            "note IS NULL OR (length(note) BETWEEN 1 AND 1000 AND instr(note, char(0)) = 0)",
            name="ck_media_collection_membership_note",
        ),
    )
    op.create_table(
        "media_tag_assignments",
        sa.Column("tag_id", sa.String(length=41), nullable=False),
        sa.Column("entry_id", sa.String(length=80), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tag_id"], ["media_tags.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["entry_id"], ["artifact_library_entries.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("tag_id", "entry_id"),
    )
    for statement in CREATE_MEDIA_ORGANIZATION_TRIGGER_SQL:
        op.execute(statement)


def downgrade() -> None:
    for statement in DROP_MEDIA_ORGANIZATION_TRIGGER_SQL:
        op.execute(statement)
    op.drop_table("media_tag_assignments")
    op.drop_table("media_collection_memberships")
    op.drop_table("media_tags")
    op.drop_table("media_collections")
