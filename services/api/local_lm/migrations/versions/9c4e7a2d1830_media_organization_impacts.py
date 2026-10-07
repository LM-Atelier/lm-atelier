"""Materialize selected organization changes and their committed results.

Revision ID: 9c4e7a2d1830
Revises: 7d39a4c81e20
Create Date: 2026-10-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from local_lm.media_organization_catalog_schema import (
    CATALOG_SEED_SQL,
    CREATE_CATALOG_TRIGGER_SQL,
    DROP_CATALOG_TRIGGER_SQL,
)
from local_lm.media_organization_schema import (
    COLLECTION_UPDATE_TRIGGER,
    COLLECTION_UPDATE_TRIGGER_V1,
    TAG_INSERT_TRIGGER,
    TAG_INSERT_TRIGGER_V1,
    TAG_UPDATE_TRIGGER,
    TAG_UPDATE_TRIGGER_V1,
    normalized_tag_label_sql,
)

revision: str = "9c4e7a2d1830"
down_revision: str | None = "7d39a4c81e20"
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
        "media_organization_creations",
        sa.Column("operation_key", sa.String(32), nullable=False),
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("entity_id", sa.String(43), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
        sa.Column("response_sha256", sa.String(64), nullable=False),
        sa.PrimaryKeyConstraint("operation_key"),
        sa.CheckConstraint("kind IN ('albums', 'tags')", name="ck_media_creation_kind"),
        sa.CheckConstraint(
            "length(operation_key) = 32 AND operation_key NOT GLOB '*[^0-9a-f]*'",
            name="ck_media_creation_key",
        ),
    )
    invalid = op.get_bind().scalar(
        sa.text(
            "SELECT EXISTS(SELECT 1 FROM media_tags WHERE slug != "
            + normalized_tag_label_sql("label")
            + ")"
        )
    )
    if invalid:
        raise RuntimeError(
            "Media tag names must be consistent before organization can be upgraded."
        )
    op.create_table(
        "media_organization_catalog_revisions",
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("kind"),
        sa.CheckConstraint("kind IN ('albums', 'tags')", name="ck_media_catalog_kind"),
        sa.CheckConstraint(
            "typeof(revision) = 'integer' AND revision BETWEEN 1 AND 9007199254740991",
            name="ck_media_catalog_revision",
        ),
    )
    op.execute(CATALOG_SEED_SQL)
    for statement in CREATE_CATALOG_TRIGGER_SQL:
        op.execute(statement)
    op.create_index("ix_media_collection_catalog_order", "media_collections", ["created_at", "id"])
    op.create_index("ix_media_tag_catalog_order", "media_tags", ["created_at", "id"])
    op.create_table(
        "media_organization_impacts",
        sa.Column("id", sa.String(40), nullable=False),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("preview_json", sa.JSON(), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recovery_json", sa.JSON(), nullable=True),
        sa.Column("operation_key", sa.String(128), nullable=True),
        sa.Column("response_json", sa.JSON(), nullable=True),
        sa.Column("response_sha256", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("operation_key"),
    )
    op.create_index(
        "ix_media_organization_impacts_expires_at", "media_organization_impacts", ["expires_at"]
    )
    op.execute("DROP TRIGGER media_collection_update_guard")
    op.execute("DROP TRIGGER media_tag_insert_guard")
    op.execute("DROP TRIGGER media_tag_update_guard")
    op.execute(COLLECTION_UPDATE_TRIGGER)
    op.execute(TAG_INSERT_TRIGGER)
    op.execute(TAG_UPDATE_TRIGGER)


def downgrade() -> None:
    _begin_write_fence()
    op.drop_table("media_organization_creations")
    for statement in DROP_CATALOG_TRIGGER_SQL:
        op.execute(statement)
    op.drop_index("ix_media_collection_catalog_order", table_name="media_collections")
    op.drop_index("ix_media_tag_catalog_order", table_name="media_tags")
    op.drop_table("media_organization_catalog_revisions")
    op.drop_table("media_organization_impacts")
    op.execute("DROP TRIGGER media_collection_update_guard")
    op.execute("DROP TRIGGER media_tag_insert_guard")
    op.execute("DROP TRIGGER media_tag_update_guard")
    op.execute(COLLECTION_UPDATE_TRIGGER_V1)
    op.execute(TAG_INSERT_TRIGGER_V1)
    op.execute(TAG_UPDATE_TRIGGER_V1)
