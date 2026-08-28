"""add user-visible Prompt Template deletion tombstones

Revision ID: d6a8f1c42b90
Revises: c5a8e1d72f40
Create Date: 2026-08-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d6a8f1c42b90"
down_revision: str | None = "c5a8e1d72f40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LIVE_INSERT_TRIGGER = """
CREATE TRIGGER prompt_template_definition_insert_guard
BEFORE INSERT ON prompt_template_definitions
BEGIN
  SELECT CASE WHEN NEW.current_revision_id IS NOT NULL
    THEN RAISE(ABORT, 'prompt template must begin without a current revision') END;
  SELECT CASE WHEN NEW.deleted_at IS NOT NULL
    THEN RAISE(ABORT, 'prompt template cannot begin deleted') END;
  SELECT CASE WHEN substr(NEW.name, 1, 28) = '__deleted_prompt_template__:'
    THEN RAISE(ABORT, 'prompt template name is reserved') END;
END
"""

_LIVE_UPDATE_TRIGGER = """
CREATE TRIGGER prompt_template_definition_update_guard
BEFORE UPDATE ON prompt_template_definitions
BEGIN
  SELECT CASE WHEN NEW.id != OLD.id
    THEN RAISE(ABORT, 'prompt template identity is immutable') END;
  SELECT CASE WHEN OLD.deleted_at IS NOT NULL
    THEN RAISE(ABORT, 'deleted prompt template is immutable') END;
  SELECT CASE WHEN NEW.deleted_at IS NULL
                        AND substr(NEW.name, 1, 28) = '__deleted_prompt_template__:'
    THEN RAISE(ABORT, 'prompt template name is reserved') END;
  SELECT CASE WHEN OLD.deleted_at IS NULL
                        AND NEW.deleted_at IS NOT NULL
                        AND (
                          NEW.name != '__deleted_prompt_template__:' || OLD.id
                          OR NEW.description != ''
                          OR NEW.archived != 1
                          OR NEW.current_revision_id IS NOT OLD.current_revision_id
                        )
    THEN RAISE(ABORT, 'prompt template deletion tombstone is invalid') END;
  SELECT CASE WHEN OLD.current_revision_id IS NOT NULL
                        AND NEW.current_revision_id IS NULL
    THEN RAISE(ABORT, 'prompt template current revision cannot be cleared') END;
  SELECT CASE WHEN NEW.current_revision_id IS NOT NULL AND NOT EXISTS (
    SELECT 1
    FROM prompt_template_revisions AS revision
    WHERE revision.id = NEW.current_revision_id
      AND revision.prompt_template_id = NEW.id
      AND revision.version = (
        SELECT max(candidate.version)
        FROM prompt_template_revisions AS candidate
        WHERE candidate.prompt_template_id = NEW.id
      )
  ) THEN RAISE(ABORT, 'prompt template current revision is invalid') END;
END
"""

_LEGACY_INSERT_TRIGGER = """
CREATE TRIGGER prompt_template_definition_insert_guard
BEFORE INSERT ON prompt_template_definitions
BEGIN
  SELECT CASE WHEN NEW.current_revision_id IS NOT NULL
    THEN RAISE(ABORT, 'prompt template must begin without a current revision') END;
END
"""

_LEGACY_UPDATE_TRIGGER = """
CREATE TRIGGER prompt_template_definition_update_guard
BEFORE UPDATE ON prompt_template_definitions
BEGIN
  SELECT CASE WHEN NEW.id != OLD.id
    THEN RAISE(ABORT, 'prompt template identity is immutable') END;
  SELECT CASE WHEN OLD.current_revision_id IS NOT NULL
                        AND NEW.current_revision_id IS NULL
    THEN RAISE(ABORT, 'prompt template current revision cannot be cleared') END;
  SELECT CASE WHEN NEW.current_revision_id IS NOT NULL AND NOT EXISTS (
    SELECT 1
    FROM prompt_template_revisions AS revision
    WHERE revision.id = NEW.current_revision_id
      AND revision.prompt_template_id = NEW.id
      AND revision.version = (
        SELECT max(candidate.version)
        FROM prompt_template_revisions AS candidate
        WHERE candidate.prompt_template_id = NEW.id
      )
  ) THEN RAISE(ABORT, 'prompt template current revision is invalid') END;
END
"""

_DEFINITION_DELETE_TRIGGER = """
CREATE TRIGGER prompt_template_definition_delete_guard
BEFORE DELETE ON prompt_template_definitions
BEGIN
  SELECT RAISE(ABORT, 'prompt template definitions are archived, not deleted');
END
"""

_IMPORT_WINNER_INSERT_TRIGGER = """
CREATE TRIGGER prompt_template_import_winner_insert_guard
BEFORE INSERT ON prompt_template_import_winners
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1
    FROM prompt_template_definitions AS definition
    JOIN prompt_template_revisions AS revision
      ON revision.id = NEW.prompt_template_revision_id
     AND revision.prompt_template_id = definition.id
    WHERE definition.id = NEW.prompt_template_id
      AND definition.current_revision_id = NEW.prompt_template_revision_id
      AND revision.version = 1
      AND revision.contract_sha256 = NEW.contract_sha256
  ) THEN RAISE(ABORT, 'prompt template import winner is invalid') END;
END
"""


def _replace_definition_triggers(insert_sql: str, update_sql: str) -> None:
    op.execute("DROP TRIGGER prompt_template_definition_update_guard")
    op.execute("DROP TRIGGER prompt_template_definition_insert_guard")
    op.execute(insert_sql)
    op.execute(update_sql)


def upgrade() -> None:
    reserved_name_count = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT count(*) FROM prompt_template_definitions "
                "WHERE substr(name, 1, 28) = '__deleted_prompt_template__:'"
            )
        )
        .scalar_one()
    )
    if reserved_name_count:
        raise RuntimeError(
            "Cannot enable Prompt Template deletion: "
            "rename templates using the reserved name prefix."
        )
    with op.batch_alter_table("prompt_template_definitions") as batch:
        batch.add_column(sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_index(
            "ix_prompt_template_definitions_deleted_at",
            ["deleted_at"],
            unique=False,
        )
    _replace_definition_triggers(_LIVE_INSERT_TRIGGER, _LIVE_UPDATE_TRIGGER)


def downgrade() -> None:
    deleted_count = (
        op.get_bind()
        .execute(
            sa.text("SELECT count(*) FROM prompt_template_definitions WHERE deleted_at IS NOT NULL")
        )
        .scalar_one()
    )
    if deleted_count:
        raise RuntimeError(
            "Cannot downgrade Prompt Template deletion: "
            "deleted template metadata cannot be restored."
        )
    op.execute("DROP TRIGGER prompt_template_definition_update_guard")
    op.execute("DROP TRIGGER prompt_template_definition_insert_guard")
    op.execute("DROP TRIGGER prompt_template_definition_delete_guard")
    op.execute("DROP TRIGGER prompt_template_import_winner_insert_guard")
    with op.batch_alter_table("prompt_template_definitions") as batch:
        batch.drop_index("ix_prompt_template_definitions_deleted_at")
        batch.drop_column("deleted_at")
    op.execute(_LEGACY_INSERT_TRIGGER)
    op.execute(_LEGACY_UPDATE_TRIGGER)
    op.execute(_DEFINITION_DELETE_TRIGGER)
    op.execute(_IMPORT_WINNER_INSERT_TRIGGER)
