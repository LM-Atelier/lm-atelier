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

_JOB_SCALAR_KEYS = (
    "artifact_id",
    "source_artifact_id",
    "result_artifact_id",
    "input_artifact_id",
    "poster_artifact_id",
    "browser_proxy_artifact_id",
)
_JOB_LIST_KEYS = ("artifact_ids", "input_artifact_ids")

#: Metadata keys by which one artifact retains another. Every site asking
#: "which artifacts does this one name" must read the same keys: the write
#: validation and delete-trigger SQL generated below, the reference walk and
#: its pending-write counterpart, library deletion's linked set, and export
#: bundling. Separate copies can drift apart while still comparing equal, so
#: these sites share this object rather than repeating its literals.
ARTIFACT_METADATA_REFERENCE_KEYS = ("poster_artifact_id", "browser_proxy_artifact_id")
MAX_JSON_BYTES = 1_048_576
MAX_JSON_NODES = 100_000
MAX_JSON_DEPTH = 16
MAX_JSON_MEMBERS = 4_096
MAX_JSON_TEXT_BYTES = 1_000_000

_TABLE_COLUMNS = {
    "jobs": (("payload_json", "object"), ("result_json", "object")),
    "runs": (("settings_json", "object"), ("provenance_json", "object")),
    "work_steps": (("settings_json", "object"), ("input_bindings_json", "array")),
    "message_references": (("artifact_ids_json", "array"),),
    "chats": (("origin_json", "object"),),
    "artifacts": (("metadata_json", "object"),),
}


def _quoted(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _common_invalid(expression: str, root_type: str) -> str:
    return f"""CASE
      WHEN NOT json_valid({expression}) THEN 1
      WHEN length(CAST({expression} AS BLOB)) > {MAX_JSON_BYTES} THEN 1
      WHEN json_type({expression}) != '{root_type}' THEN 1
      WHEN (SELECT count(*) FROM json_tree({expression})) > {MAX_JSON_NODES} THEN 1
      WHEN (SELECT count(*) FROM json_each({expression})) > {MAX_JSON_MEMBERS} THEN 1
      WHEN EXISTS (
        SELECT 1 FROM json_tree({expression}) AS container
        WHERE container.type IN ('array', 'object')
          AND (SELECT count(*) FROM json_each(container.value)) > {MAX_JSON_MEMBERS}
      ) THEN 1
      WHEN (
        WITH RECURSIVE depth(id, level) AS (
          SELECT id, 0 FROM json_tree({expression}) WHERE parent IS NULL
          UNION ALL
          SELECT child.id, depth.level + 1
          FROM json_tree({expression}) AS child
          JOIN depth ON child.parent = depth.id
        )
        SELECT COALESCE(max(level), 0) FROM depth
      ) > {MAX_JSON_DEPTH} THEN 1
      WHEN (SELECT COALESCE(sum(length(CAST(value AS BLOB))), 0)
            FROM json_tree({expression}) WHERE type = 'text') > {MAX_JSON_TEXT_BYTES} THEN 1
      ELSE 0
    END"""


def _optional_id_invalid(expression: str, path: str) -> str:
    return f"""(
      json_type({expression}, '{path}') IS NOT NULL AND
      (json_type({expression}, '{path}') NOT IN ('null', 'text') OR
       (json_type({expression}, '{path}') = 'text' AND
        length(json_extract({expression}, '{path}')) NOT BETWEEN 1 AND 80))
    )"""


def _list_invalid(expression: str, path: str) -> str:
    return f"""CASE
      WHEN json_type({expression}, '{path}') IS NULL THEN 0
      WHEN json_type({expression}, '{path}') != 'array' THEN 1
      WHEN EXISTS (
        SELECT 1 FROM json_each({expression}, '{path}') AS item
        WHERE item.type != 'text' OR length(item.value) NOT BETWEEN 1 AND 80
      ) THEN 1
      ELSE 0 END"""


def _job_invalid(expression: str) -> str:
    return f"""EXISTS (
      SELECT 1
      FROM json_tree({expression}) AS node
      LEFT JOIN json_tree({expression}) AS parent ON parent.id = node.parent
      WHERE (node.key IN ({_quoted(_JOB_SCALAR_KEYS)}) AND
             (node.type NOT IN ('null', 'text') OR
              (node.type = 'text' AND length(node.value) NOT BETWEEN 1 AND 80)))
         OR (node.key IN ({_quoted(_JOB_LIST_KEYS)}) AND node.type != 'array')
         OR (parent.key IN ({_quoted(_JOB_LIST_KEYS)}) AND
             (node.type != 'text' OR length(node.value) NOT BETWEEN 1 AND 80))
    )"""


def _mask_invalid(expression: str) -> str:
    return f"""(
      (json_type({expression}, '$.mask') IS NOT NULL AND
       json_type({expression}, '$.mask') NOT IN ('null', 'object')) OR
      (json_type({expression}, '$.mask') = 'object'
       AND (json_type({expression}, '$.mask.artifact_id') IS NULL OR
            {_optional_id_invalid(expression, "$.mask.artifact_id")}))
    )"""


def _array_items_not_objects(expression: str, path: str = "$") -> str:
    return f"""EXISTS (
      SELECT 1 FROM json_each({expression}, '{path}') AS item
      WHERE item.type != 'object'
    )"""


def _run_invalid(settings: str, provenance: str) -> str:
    output_ids = " OR ".join(
        _optional_id_invalid("output.value", f"$.{key}")
        for key in ("artifact_id", "poster_artifact_id", "browser_proxy_artifact_id")
    )
    return " OR ".join(
        (
            _mask_invalid(settings),
            _list_invalid(provenance, "$.input_artifact_ids"),
            _list_invalid(provenance, "$.resolved_dependency_artifact_ids"),
            f"""CASE
              WHEN json_type({provenance}, '$.outputs') IS NULL THEN 0
              WHEN json_type({provenance}, '$.outputs') != 'array' THEN 1
              WHEN {_array_items_not_objects(provenance, "$.outputs")} THEN 1
              WHEN EXISTS (
                 SELECT 1 FROM json_each({provenance}, '$.outputs') AS output
                 WHERE {output_ids}
               ) THEN 1
              ELSE 0 END""",
        )
    )


def _work_step_invalid(settings: str, bindings: str) -> str:
    return " OR ".join(
        (
            _mask_invalid(settings),
            _array_items_not_objects(bindings),
            f"""EXISTS (
              SELECT 1 FROM json_each({bindings}) AS binding
              WHERE {_optional_id_invalid("binding.value", "$.artifact_id")}
            )""",
        )
    )


def _root_id_list_invalid(expression: str) -> str:
    return f"""EXISTS (
      SELECT 1 FROM json_each({expression}) AS item
      WHERE item.type != 'text' OR length(item.value) NOT BETWEEN 1 AND 80
    )"""


def _table_invalid(table: str, prefix: str) -> tuple[str, ...]:
    if table == "jobs":
        return (_job_invalid(f"{prefix}.payload_json"), _job_invalid(f"{prefix}.result_json"))
    if table == "runs":
        return (_run_invalid(f"{prefix}.settings_json", f"{prefix}.provenance_json"),)
    if table == "work_steps":
        return (_work_step_invalid(f"{prefix}.settings_json", f"{prefix}.input_bindings_json"),)
    if table == "message_references":
        return (_root_id_list_invalid(f"{prefix}.artifact_ids_json"),)
    if table == "chats":
        return (_optional_id_invalid(f"{prefix}.origin_json", "$.source_artifact_id"),)
    if table == "artifacts":
        return tuple(
            _optional_id_invalid(f"{prefix}.metadata_json", f"$.{key}")
            for key in ARTIFACT_METADATA_REFERENCE_KEYS
        )
    raise AssertionError(table)


def _reference_values(table: str, prefix: str) -> str:
    if table == "jobs":
        selects = []
        for column in ("payload_json", "result_json"):
            expression = f"{prefix}.{column}"
            selects.append(
                f"SELECT node.value AS artifact_id FROM json_tree({expression}) AS node "
                f"LEFT JOIN json_tree({expression}) AS parent ON parent.id = node.parent "
                f"WHERE node.type = 'text' AND (node.key IN ({_quoted(_JOB_SCALAR_KEYS)}) "
                f"OR parent.key IN ({_quoted(_JOB_LIST_KEYS)}))"
            )
        return " UNION ALL ".join(selects)
    if table == "runs":
        selects = [
            f"SELECT json_extract({prefix}.settings_json, '$.mask.artifact_id') AS artifact_id "
            f"WHERE json_type({prefix}.settings_json, '$.mask.artifact_id') = 'text'",
            f"SELECT item.value FROM json_each({prefix}.provenance_json, "
            "'$.input_artifact_ids') AS item",
            f"SELECT item.value FROM json_each({prefix}.provenance_json, "
            "'$.resolved_dependency_artifact_ids') AS item",
        ]
        selects.extend(
            f"SELECT json_extract(output.value, '$.{key}') "
            f"FROM json_each({prefix}.provenance_json, '$.outputs') AS output "
            f"WHERE json_type(output.value, '$.{key}') = 'text'"
            for key in ("artifact_id", "poster_artifact_id", "browser_proxy_artifact_id")
        )
        return " UNION ALL ".join(selects)
    if table == "work_steps":
        return " UNION ALL ".join(
            (
                f"SELECT json_extract({prefix}.settings_json, '$.mask.artifact_id') AS artifact_id "
                f"WHERE json_type({prefix}.settings_json, '$.mask.artifact_id') = 'text'",
                f"SELECT json_extract(binding.value, '$.artifact_id') "
                f"FROM json_each({prefix}.input_bindings_json) AS binding "
                "WHERE json_type(binding.value, '$.artifact_id') = 'text'",
            )
        )
    if table == "message_references":
        return (
            f"SELECT item.value AS artifact_id FROM json_each({prefix}.artifact_ids_json) AS item"
        )
    if table == "chats":
        return (
            f"SELECT json_extract({prefix}.origin_json, '$.source_artifact_id') AS artifact_id "
            f"WHERE json_type({prefix}.origin_json, '$.source_artifact_id') = 'text'"
        )
    if table == "artifacts":
        return " UNION ALL ".join(
            f"SELECT json_extract({prefix}.metadata_json, '$.{key}') AS artifact_id "
            f"WHERE json_type({prefix}.metadata_json, '$.{key}') = 'text'"
            for key in ARTIFACT_METADATA_REFERENCE_KEYS
        )
    raise AssertionError(table)


def _json_write_triggers(
    table: str,
    columns: tuple[tuple[str, str], ...],
    *,
    when: str | None = None,
) -> tuple[str, str]:
    structural_invalid = [
        _common_invalid(f"NEW.{column}", root_type) for column, root_type in columns
    ]
    semantic_invalid = list(_table_invalid(table, "NEW"))
    invalid = (
        f"CASE WHEN {' OR '.join(structural_invalid)} THEN 1 "
        f"WHEN {' OR '.join(semantic_invalid)} THEN 1 ELSE 0 END"
    )
    references = _reference_values(table, "NEW")
    column_names = tuple(column for column, _ in columns)
    watched_columns = (*column_names, "scope") if table == "chats" else column_names
    when_sql = f"WHEN {when}\n" if when else ""
    body = (
        f"{when_sql}BEGIN\n"
        f"  SELECT CASE WHEN {invalid}\n"
        "    THEN RAISE(ABORT, 'artifact JSON reference is invalid') END;\n"
        "  SELECT CASE WHEN EXISTS (\n"
        f"    SELECT 1 FROM ({references}) AS reference\n"
        "    WHERE NOT EXISTS (SELECT 1 FROM artifacts WHERE id = reference.artifact_id)\n"
        "  ) THEN RAISE(ABORT, 'artifact JSON reference is invalid') END;\n"
        "END"
    )
    return (
        f"CREATE TRIGGER {table}_artifact_reference_insert_guard\nBEFORE INSERT ON {table}\n{body}",
        f"CREATE TRIGGER {table}_artifact_reference_update_guard\n"
        f"BEFORE UPDATE OF {', '.join(watched_columns)} ON {table}\n{body}",
    )


JSON_WRITE_TRIGGER_SQL = (
    *(
        trigger
        for table, columns in _TABLE_COLUMNS.items()
        for trigger in _json_write_triggers(
            table,
            columns,
            when="NEW.scope = 'studio'" if table == "chats" else None,
        )
    ),
)


def _stored_json_invalid() -> str:
    checks = []
    for table, columns in _TABLE_COLUMNS.items():
        structural_invalid = [
            _common_invalid(f"{table}.{column}", root_type) for column, root_type in columns
        ]
        semantic_invalid = list(_table_invalid(table, table))
        invalid = (
            f"CASE WHEN {' OR '.join(structural_invalid)} THEN 1 "
            f"WHEN {' OR '.join(semantic_invalid)} THEN 1 ELSE 0 END"
        )
        scope = "scope = 'studio' AND " if table == "chats" else ""
        checks.append(f"EXISTS (SELECT 1 FROM {table} WHERE {scope}({invalid}))")
    return " OR\n    ".join(checks)


def _stored_reference_missing() -> str:
    checks = []
    for table in _TABLE_COLUMNS:
        scope = "scope = 'studio' AND " if table == "chats" else ""
        checks.append(
            f"EXISTS (SELECT 1 FROM {table} WHERE {scope}EXISTS ("
            f"SELECT 1 FROM ({_reference_values(table, table)}) AS reference "
            "WHERE NOT EXISTS (SELECT 1 FROM artifacts "
            "WHERE id = reference.artifact_id)))"
        )
    return " OR\n    ".join(checks)


PREMIGRATION_INVALID_SQL = f"""SELECT CASE
  WHEN {_stored_json_invalid()} THEN 1
  WHEN {_stored_reference_missing()} THEN 1
  WHEN EXISTS (
    SELECT 1 FROM artifacts
    WHERE kind IN ('image', 'video') AND original_name IS NOT NULL
      AND (length(trim(original_name)) > 500 OR instr(original_name, char(0)) != 0)
  ) THEN 1
  ELSE 0 END
"""


def _contains_deleted(table: str) -> str:
    scope = "scope = 'studio' AND " if table == "chats" else ""
    return f"""EXISTS (
      SELECT 1 FROM {table}
      WHERE {scope}EXISTS (
        SELECT 1 FROM ({_reference_values(table, table)}) AS reference
        WHERE reference.artifact_id = OLD.id
      )
    )"""


ARTIFACT_JSON_DELETE_TRIGGER = f"""
CREATE TRIGGER artifact_json_reference_delete_guard
BEFORE DELETE ON artifacts
BEGIN
  SELECT CASE WHEN
    {_stored_json_invalid()}
    THEN RAISE(ABORT, 'artifact JSON reference is invalid') END;
  SELECT CASE WHEN
    {" OR ".join(_contains_deleted(table) for table in _TABLE_COLUMNS)}
    THEN RAISE(ABORT, 'artifact is retained by JSON reference') END;
END
"""

AUDIT_TRIGGER_SQL = tuple(
    f"UPDATE {table} SET {', '.join(f'{column} = {column}' for column, _ in columns)}"
    + (" WHERE scope = 'studio'" if table == "chats" else "")
    for table, columns in _TABLE_COLUMNS.items()
)

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
