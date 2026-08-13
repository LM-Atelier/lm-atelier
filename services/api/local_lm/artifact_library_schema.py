"""SQLite trigger truth shared by ORM create-all and the Alembic migration."""

from __future__ import annotations

ENTRY_INSERT_TRIGGER = """
CREATE TRIGGER artifact_library_entry_insert_guard
BEFORE INSERT ON artifact_library_entries
BEGIN
  SELECT CASE WHEN NEW.id != 'libentry:sha256:' || (
    SELECT sha256 FROM artifacts WHERE id = NEW.artifact_id
  )
    THEN RAISE(ABORT, 'artifact library entry identity is invalid') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM artifacts
    WHERE id = NEW.artifact_id AND kind IN ('image', 'video')
  ) THEN RAISE(ABORT, 'artifact library entry requires media') END;
END
"""

ENTRY_UPDATE_TRIGGER = """
CREATE TRIGGER artifact_library_entry_update_guard
BEFORE UPDATE ON artifact_library_entries
BEGIN
  SELECT CASE WHEN NEW.id != OLD.id OR NEW.artifact_id != OLD.artifact_id
                      OR NEW.created_at != OLD.created_at
    THEN RAISE(ABORT, 'artifact library entry identity is immutable') END;
  SELECT CASE WHEN NEW.version != OLD.version + 1
    THEN RAISE(ABORT, 'artifact library entry version is stale') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM artifacts
    WHERE id = NEW.artifact_id AND kind IN ('image', 'video')
  ) THEN RAISE(ABORT, 'artifact library entry requires media') END;
END
"""

ARTIFACT_UPDATE_TRIGGER = """
CREATE TRIGGER artifact_library_artifact_update_guard
BEFORE UPDATE OF id, sha256, kind ON artifacts
WHEN EXISTS (SELECT 1 FROM artifact_library_entries WHERE artifact_id = OLD.id)
BEGIN
  SELECT CASE WHEN NEW.id != OLD.id OR NEW.sha256 != OLD.sha256
                      OR NEW.kind != OLD.kind
    THEN RAISE(ABORT, 'library artifact identity is immutable') END;
END
"""

ENTRY_DELETE_TRIGGER = """
CREATE TRIGGER artifact_library_entry_delete_guard
BEFORE DELETE ON artifact_library_entries
BEGIN
  SELECT RAISE(ABORT, 'artifact library entry deletion is not authorized');
END
"""

CREATE_TRIGGER_SQL = (
    ENTRY_INSERT_TRIGGER,
    ENTRY_UPDATE_TRIGGER,
    ARTIFACT_UPDATE_TRIGGER,
    ENTRY_DELETE_TRIGGER,
)

DROP_TRIGGER_SQL = (
    "DROP TRIGGER IF EXISTS artifact_library_entry_delete_guard",
    "DROP TRIGGER IF EXISTS artifact_library_artifact_update_guard",
    "DROP TRIGGER IF EXISTS artifact_library_entry_update_guard",
    "DROP TRIGGER IF EXISTS artifact_library_entry_insert_guard",
)
