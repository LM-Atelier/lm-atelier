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

_SCALAR_KEYS = (
    "artifact_id",
    "source_artifact_id",
    "result_artifact_id",
    "input_artifact_id",
    "poster_artifact_id",
    "browser_proxy_artifact_id",
)
_LIST_KEYS = ("artifact_ids", "input_artifact_ids", "resolved_dependency_artifact_ids")


def _quoted(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _missing_reference(column: str, *, root_list: bool = False) -> str:
    selector = (
        "child.type = 'text'"
        if root_list
        else (
            f"child.type = 'text' AND (child.key IN ({_quoted(_SCALAR_KEYS)}) "
            f"OR parent.key IN ({_quoted(_LIST_KEYS)}))"
        )
    )
    return f"""EXISTS (
      SELECT 1
      FROM json_tree(NEW.{column}) AS child
      LEFT JOIN json_tree(NEW.{column}) AS parent ON parent.id = child.parent
      WHERE {selector}
        AND NOT EXISTS (SELECT 1 FROM artifacts WHERE id = child.value)
    )"""


def _contains_deleted(column: str, *, root_list: bool = False) -> str:
    table = column.split(".", 1)[0]
    selector = (
        "child.type = 'text'"
        if root_list
        else (
            f"child.type = 'text' AND (child.key IN ({_quoted(_SCALAR_KEYS)}) "
            f"OR parent.key IN ({_quoted(_LIST_KEYS)}))"
        )
    )
    return f"""EXISTS (
      SELECT 1
      FROM {table}, json_tree({column}) AS child
      LEFT JOIN json_tree({column}) AS parent ON parent.id = child.parent
      WHERE {selector} AND child.value = OLD.id
    )"""


def _json_write_triggers(
    table: str,
    columns: tuple[str, ...],
    *,
    when: str | None = None,
    root_list: bool = False,
) -> tuple[str, str]:
    condition = " OR ".join(_missing_reference(column, root_list=root_list) for column in columns)
    when_sql = f"WHEN {when}\n" if when else ""
    body = (
        f"{when_sql}BEGIN\n"
        f"  SELECT CASE WHEN {condition}\n"
        "    THEN RAISE(ABORT, 'artifact JSON reference is invalid') END;\n"
        "END"
    )
    return (
        f"CREATE TRIGGER {table}_artifact_reference_insert_guard\nBEFORE INSERT ON {table}\n{body}",
        f"CREATE TRIGGER {table}_artifact_reference_update_guard\n"
        f"BEFORE UPDATE OF {', '.join(columns)} ON {table}\n{body}",
    )


JSON_WRITE_TRIGGER_SQL = (
    *_json_write_triggers("jobs", ("payload_json", "result_json")),
    *_json_write_triggers("runs", ("settings_json", "provenance_json")),
    *_json_write_triggers("work_steps", ("settings_json", "input_bindings_json")),
    *_json_write_triggers("message_references", ("artifact_ids_json",), root_list=True),
    *_json_write_triggers("chats", ("origin_json",), when="NEW.scope = 'studio'"),
    *_json_write_triggers("artifacts", ("metadata_json",)),
)

ARTIFACT_JSON_DELETE_TRIGGER = f"""
CREATE TRIGGER artifact_json_reference_delete_guard
BEFORE DELETE ON artifacts
BEGIN
  SELECT CASE WHEN
    {_contains_deleted("jobs.payload_json")} OR
    {_contains_deleted("jobs.result_json")} OR
    {_contains_deleted("runs.settings_json")} OR
    {_contains_deleted("runs.provenance_json")} OR
    {_contains_deleted("work_steps.settings_json")} OR
    {_contains_deleted("work_steps.input_bindings_json")} OR
    {_contains_deleted("message_references.artifact_ids_json", root_list=True)} OR
    {_contains_deleted("chats.origin_json")} OR
    {_contains_deleted("artifacts.metadata_json")}
    THEN RAISE(ABORT, 'artifact is retained by JSON reference') END;
END
"""

CREATE_TRIGGER_SQL = (
    ENTRY_INSERT_TRIGGER,
    ENTRY_UPDATE_TRIGGER,
    ARTIFACT_UPDATE_TRIGGER,
    ENTRY_DELETE_TRIGGER,
    *JSON_WRITE_TRIGGER_SQL,
    ARTIFACT_JSON_DELETE_TRIGGER,
)

DROP_TRIGGER_SQL = (
    "DROP TRIGGER IF EXISTS artifact_json_reference_delete_guard",
    *(
        f"DROP TRIGGER IF EXISTS {table}_artifact_reference_{action}_guard"
        for table in ("artifacts", "chats", "message_references", "work_steps", "runs", "jobs")
        for action in ("update", "insert")
    ),
    "DROP TRIGGER IF EXISTS artifact_library_entry_delete_guard",
    "DROP TRIGGER IF EXISTS artifact_library_artifact_update_guard",
    "DROP TRIGGER IF EXISTS artifact_library_entry_update_guard",
    "DROP TRIGGER IF EXISTS artifact_library_entry_insert_guard",
)
