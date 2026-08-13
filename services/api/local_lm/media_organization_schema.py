"""SQLite trigger truth for manual Media Library organization records."""

from __future__ import annotations

_CONTROL_CODES = (*range(32), 127)


def _control_invalid(expression: str) -> str:
    return " OR ".join(f"instr({expression}, char({code})) != 0" for code in _CONTROL_CODES)


def _guard_end(condition: str, message: str) -> str:
    return f"  SELECT CASE WHEN {condition}\n    THEN RAISE(ABORT, '{message}') END;\nEND\n"


COLLECTION_INSERT_TRIGGER = """
CREATE TRIGGER media_collection_insert_guard
BEFORE INSERT ON media_collections
BEGIN
  SELECT CASE WHEN length(NEW.id) != 43
                    OR substr(NEW.id, 1, 11) != 'collection_'
                    OR substr(NEW.id, 12) GLOB '*[^0-9a-f]*'
    THEN RAISE(ABORT, 'media collection identity is invalid') END;
END
""".replace(
    "END\n",
    _guard_end(
        f"{_control_invalid('NEW.name')} OR {_control_invalid('NEW.description')}",
        "media collection text is invalid",
    ),
)

COLLECTION_UPDATE_TRIGGER = """
CREATE TRIGGER media_collection_update_guard
BEFORE UPDATE ON media_collections
BEGIN
  SELECT CASE WHEN NEW.id != OLD.id OR NEW.kind != OLD.kind
                      OR NEW.created_at != OLD.created_at
    THEN RAISE(ABORT, 'media collection identity is immutable') END;
  SELECT CASE WHEN NEW.version != OLD.version + 1
    THEN RAISE(ABORT, 'media collection version is stale') END;
END
""".replace(
    "END\n",
    _guard_end(
        f"{_control_invalid('NEW.name')} OR {_control_invalid('NEW.description')}",
        "media collection text is invalid",
    ),
)

MEMBERSHIP_INSERT_TRIGGER = """
CREATE TRIGGER media_collection_membership_insert_guard
BEFORE INSERT ON media_collection_memberships
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM media_collections
    WHERE id = NEW.collection_id AND kind = 'manual'
  ) THEN RAISE(ABORT, 'manual media collection is required') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM artifact_library_entries WHERE id = NEW.entry_id
  ) THEN RAISE(ABORT, 'media library entry is required') END;
END
"""

MEMBERSHIP_UPDATE_TRIGGER = """
CREATE TRIGGER media_collection_membership_update_guard
BEFORE UPDATE ON media_collection_memberships
BEGIN
  SELECT RAISE(ABORT, 'media collection membership is immutable');
END
"""

MEMBERSHIP_INSERT_VERSION_TRIGGER = """
CREATE TRIGGER media_collection_membership_insert_version
AFTER INSERT ON media_collection_memberships
BEGIN
  UPDATE media_collections
  SET version = version + 1, updated_at = CURRENT_TIMESTAMP
  WHERE id = NEW.collection_id;
END
"""

MEMBERSHIP_DELETE_VERSION_TRIGGER = """
CREATE TRIGGER media_collection_membership_delete_version
AFTER DELETE ON media_collection_memberships
BEGIN
  UPDATE media_collections
  SET version = version + 1, updated_at = CURRENT_TIMESTAMP
  WHERE id = OLD.collection_id;
END
"""

TAG_INSERT_TRIGGER = """
CREATE TRIGGER media_tag_insert_guard
BEFORE INSERT ON media_tags
BEGIN
  SELECT CASE WHEN length(NEW.id) != 41
                    OR substr(NEW.id, 1, 9) != 'mediatag_'
                    OR substr(NEW.id, 10) GLOB '*[^0-9a-f]*'
    THEN RAISE(ABORT, 'media tag identity is invalid') END;
END
""".replace("END\n", _guard_end(_control_invalid("NEW.label"), "media tag text is invalid"))

TAG_UPDATE_TRIGGER = """
CREATE TRIGGER media_tag_update_guard
BEFORE UPDATE ON media_tags
BEGIN
  SELECT CASE WHEN NEW.id != OLD.id OR NEW.normalized_name != OLD.normalized_name
                      OR NEW.created_at != OLD.created_at
    THEN RAISE(ABORT, 'media tag identity is immutable') END;
  SELECT CASE WHEN NEW.version != OLD.version + 1
    THEN RAISE(ABORT, 'media tag version is stale') END;
END
""".replace("END\n", _guard_end(_control_invalid("NEW.label"), "media tag text is invalid"))

TAG_ASSIGNMENT_INSERT_TRIGGER = """
CREATE TRIGGER media_tag_assignment_insert_guard
BEFORE INSERT ON media_tag_assignments
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM media_tags WHERE id = NEW.tag_id
  ) THEN RAISE(ABORT, 'media tag is required') END;
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM artifact_library_entries WHERE id = NEW.entry_id
  ) THEN RAISE(ABORT, 'media library entry is required') END;
END
"""

TAG_ASSIGNMENT_UPDATE_TRIGGER = """
CREATE TRIGGER media_tag_assignment_update_guard
BEFORE UPDATE ON media_tag_assignments
BEGIN
  SELECT RAISE(ABORT, 'media tag assignment is immutable');
END
"""

TAG_ASSIGNMENT_INSERT_VERSION_TRIGGER = """
CREATE TRIGGER media_tag_assignment_insert_version
AFTER INSERT ON media_tag_assignments
BEGIN
  UPDATE media_tags
  SET version = version + 1, updated_at = CURRENT_TIMESTAMP
  WHERE id = NEW.tag_id;
END
"""

TAG_ASSIGNMENT_DELETE_VERSION_TRIGGER = """
CREATE TRIGGER media_tag_assignment_delete_version
AFTER DELETE ON media_tag_assignments
BEGIN
  UPDATE media_tags
  SET version = version + 1, updated_at = CURRENT_TIMESTAMP
  WHERE id = OLD.tag_id;
END
"""

CREATE_MEDIA_ORGANIZATION_TRIGGER_SQL = (
    COLLECTION_INSERT_TRIGGER,
    COLLECTION_UPDATE_TRIGGER,
    MEMBERSHIP_INSERT_TRIGGER,
    MEMBERSHIP_UPDATE_TRIGGER,
    MEMBERSHIP_INSERT_VERSION_TRIGGER,
    MEMBERSHIP_DELETE_VERSION_TRIGGER,
    TAG_INSERT_TRIGGER,
    TAG_UPDATE_TRIGGER,
    TAG_ASSIGNMENT_INSERT_TRIGGER,
    TAG_ASSIGNMENT_UPDATE_TRIGGER,
    TAG_ASSIGNMENT_INSERT_VERSION_TRIGGER,
    TAG_ASSIGNMENT_DELETE_VERSION_TRIGGER,
)

DROP_MEDIA_ORGANIZATION_TRIGGER_SQL = tuple(
    f"DROP TRIGGER IF EXISTS {name}"
    for name in (
        "media_tag_assignment_delete_version",
        "media_tag_assignment_insert_version",
        "media_tag_assignment_update_guard",
        "media_tag_assignment_insert_guard",
        "media_tag_update_guard",
        "media_tag_insert_guard",
        "media_collection_membership_delete_version",
        "media_collection_membership_insert_version",
        "media_collection_membership_update_guard",
        "media_collection_membership_insert_guard",
        "media_collection_update_guard",
        "media_collection_insert_guard",
    )
)
