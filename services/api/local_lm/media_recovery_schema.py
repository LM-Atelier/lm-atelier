"""Retire library memberships only through their exact permanent-deletion transition."""

ENTRY_DELETE_TRIGGER = """
CREATE TRIGGER artifact_library_entry_delete_guard
BEFORE DELETE ON artifact_library_entries
WHEN NOT EXISTS (
  SELECT 1 FROM recovery_items
  WHERE kind = 'media_library_entry' AND subject_id = OLD.id
    AND deletion_id = OLD.recovery_id AND state = 'purging' AND OLD.state = 'trashed'
)
BEGIN
  SELECT RAISE(ABORT, 'artifact library entry deletion is not authorized');
END
"""

ENTRY_WRITE_TRIGGER = """
CREATE TRIGGER media_recovery_entry_update_guard
BEFORE UPDATE ON artifact_library_entries
WHEN OLD.state = 'trashed' AND NOT (
  NEW.state = 'visible' AND NEW.recovery_id IS NULL AND NEW.deleted_at IS NULL
  AND NEW.display_name IS OLD.display_name AND NEW.favorite IS OLD.favorite
  AND EXISTS (SELECT 1 FROM recovery_items
    WHERE kind = 'media_library_entry' AND subject_id = OLD.id
      AND deletion_id = OLD.recovery_id AND state = 'restoring')
)
BEGIN SELECT RAISE(ABORT, 'media-recovery-write-refused'); END
"""

FAVORITE_WRITE_TRIGGER = """
CREATE TRIGGER media_recovery_artifact_favorite_guard
BEFORE UPDATE OF favorite ON artifacts
WHEN NEW.favorite IS NOT OLD.favorite AND EXISTS (
  SELECT 1 FROM artifact_library_entries entry
  WHERE entry.artifact_id = OLD.id AND entry.state = 'trashed'
    AND NOT (NEW.favorite = 0 AND EXISTS (SELECT 1 FROM recovery_items
      WHERE kind = 'media_library_entry' AND subject_id = entry.id
        AND deletion_id = entry.recovery_id AND state = 'purging'))
)
BEGIN SELECT RAISE(ABORT, 'media-recovery-write-refused'); END
"""


def _organization_delete(table: str) -> str:
    return f"""
CREATE TRIGGER media_recovery_{table}_delete_guard
BEFORE DELETE ON {table}
WHEN EXISTS (
  SELECT 1 FROM artifact_library_entries entry
  WHERE entry.id = OLD.entry_id AND entry.state = 'trashed'
    AND NOT EXISTS (SELECT 1 FROM recovery_items
      WHERE kind = 'media_library_entry' AND subject_id = entry.id
        AND deletion_id = entry.recovery_id AND state = 'purging')
)
BEGIN SELECT RAISE(ABORT, 'media-recovery-write-refused'); END
"""


CREATE_MEDIA_RECOVERY_TRIGGER_SQL = (
    "DROP TRIGGER artifact_library_entry_delete_guard",
    ENTRY_DELETE_TRIGGER,
    ENTRY_WRITE_TRIGGER,
    FAVORITE_WRITE_TRIGGER,
    _organization_delete("media_collection_memberships"),
    _organization_delete("media_tag_assignments"),
)
DROP_MEDIA_RECOVERY_TRIGGER_SQL = tuple(
    f"DROP TRIGGER IF EXISTS {statement.split()[2]}"
    for statement in CREATE_MEDIA_RECOVERY_TRIGGER_SQL[1:]
)
