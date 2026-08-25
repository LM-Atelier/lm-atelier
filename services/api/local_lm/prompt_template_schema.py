"""SQLite guards for append-only Prompt Library definitions and revisions."""

from __future__ import annotations

DEFINITION_INSERT_TRIGGER = """
CREATE TRIGGER prompt_template_definition_insert_guard
BEFORE INSERT ON prompt_template_definitions
BEGIN
  SELECT CASE WHEN NEW.current_revision_id IS NOT NULL
    THEN RAISE(ABORT, 'prompt template must begin without a current revision') END;
END
"""

DEFINITION_UPDATE_TRIGGER = """
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

DEFINITION_DELETE_TRIGGER = """
CREATE TRIGGER prompt_template_definition_delete_guard
BEFORE DELETE ON prompt_template_definitions
BEGIN
  SELECT RAISE(ABORT, 'prompt template definitions are archived, not deleted');
END
"""

REVISION_INSERT_TRIGGER = """
CREATE TRIGGER prompt_template_revision_insert_guard
BEFORE INSERT ON prompt_template_revisions
BEGIN
  SELECT CASE WHEN NEW.version != COALESCE((
    SELECT max(existing.version) + 1
    FROM prompt_template_revisions AS existing
    WHERE existing.prompt_template_id = NEW.prompt_template_id
  ), 1)
    THEN RAISE(ABORT, 'prompt template revision version is not append-only') END;
  SELECT CASE WHEN NOT json_valid(NEW.contract_json)
                        OR json_type(NEW.contract_json) != 'object'
                        OR json_type(NEW.contract_json, '$.schema_version')
                           != 'integer'
                        OR json_extract(NEW.contract_json, '$.schema_version')
                           IS NOT NEW.schema_version
    THEN RAISE(ABORT, 'prompt template revision contract is invalid') END;
END
"""

REVISION_UPDATE_TRIGGER = """
CREATE TRIGGER prompt_template_revision_update_guard
BEFORE UPDATE ON prompt_template_revisions
BEGIN
  SELECT RAISE(ABORT, 'prompt template revisions are immutable');
END
"""

REVISION_DELETE_TRIGGER = """
CREATE TRIGGER prompt_template_revision_delete_guard
BEFORE DELETE ON prompt_template_revisions
BEGIN
  SELECT RAISE(ABORT, 'prompt template revisions are immutable');
END
"""

IMPORT_WINNER_INSERT_TRIGGER = """
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

IMPORT_WINNER_UPDATE_TRIGGER = """
CREATE TRIGGER prompt_template_import_winner_update_guard
BEFORE UPDATE ON prompt_template_import_winners
BEGIN
  SELECT RAISE(ABORT, 'prompt template import winners are immutable');
END
"""

IMPORT_WINNER_DELETE_TRIGGER = """
CREATE TRIGGER prompt_template_import_winner_delete_guard
BEFORE DELETE ON prompt_template_import_winners
BEGIN
  SELECT RAISE(ABORT, 'prompt template import winners are immutable');
END
"""
CREATE_PROMPT_TEMPLATE_TRIGGER_SQL = (
    DEFINITION_INSERT_TRIGGER,
    DEFINITION_UPDATE_TRIGGER,
    DEFINITION_DELETE_TRIGGER,
    REVISION_INSERT_TRIGGER,
    REVISION_UPDATE_TRIGGER,
    REVISION_DELETE_TRIGGER,
    IMPORT_WINNER_INSERT_TRIGGER,
    IMPORT_WINNER_UPDATE_TRIGGER,
    IMPORT_WINNER_DELETE_TRIGGER,
)

DROP_PROMPT_TEMPLATE_TRIGGER_SQL = (
    "DROP TRIGGER prompt_template_import_winner_delete_guard",
    "DROP TRIGGER prompt_template_import_winner_update_guard",
    "DROP TRIGGER prompt_template_import_winner_insert_guard",
    "DROP TRIGGER prompt_template_revision_delete_guard",
    "DROP TRIGGER prompt_template_revision_update_guard",
    "DROP TRIGGER prompt_template_revision_insert_guard",
    "DROP TRIGGER prompt_template_definition_delete_guard",
    "DROP TRIGGER prompt_template_definition_update_guard",
    "DROP TRIGGER prompt_template_definition_insert_guard",
)
