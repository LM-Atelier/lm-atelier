"""SQLite text for the artifact reference index, shared by ORM create-all and the migration.

The index records, for every stored document that can name an artifact, which
artifacts it names. Triggers on the six source tables record only WHICH row
changed, in the same statement as the change, so every writer is covered:
the ORM, Core statements, raw SQL, foreign-key cascades and other connections.
Python then re-reads each recorded row with the reference walk's own readers
and replaces its entries; see `artifact_reference_index`.

Only a connection that registered the writer function may change the index or
clear a recorded change, and only while it has opened the writer window, so a
stray statement cannot make an artifact look unreferenced. Recording a change
needs no function, so any connection can still write the source tables.
"""

from __future__ import annotations

from typing import Final

from .artifact_library_schema import (
    _TABLE_COLUMNS,
    ARTIFACT_JSON_DELETE_TRIGGER,
    ARTIFACT_REFERENCE_INDEX_WRITER_SQL,
    JSON_WRITE_TRIGGER_SQL,
    _common_invalid,
    _member_depth,
    _table_invalid,
)
from .artifact_reference_extractors import SOURCE_CHATS, SOURCE_COLUMNS

#: Where a statement below takes its list of keys: one placeholder per key,
#: filled by `with_keys`, so every key is bound and matched exactly as stored.
KEYS: Final = "{keys}"

EDGES_TABLE: Final = "artifact_reference_edges"
CHANGES_TABLE: Final = "artifact_reference_changes"
STAMP_TABLE: Final = "artifact_reference_index"

#: Exactly as SQLAlchemy writes these tables for the ORM models, so a store
#: created either way holds the same definitions and the migration can tell
#: its own objects from anything else. A change is recorded once for each
#: audit flag, so a record asking for the stored-document check can be added
#: beside an ordinary one by any connection, and no connection can replace it.
CREATE_TABLE_SQL: Final = (
    f"CREATE TABLE {EDGES_TABLE} (\n"
    "\tartifact_id VARCHAR(80) NOT NULL, \n"
    "\tsource INTEGER NOT NULL, \n"
    "\tsource_key VARCHAR(80) NOT NULL, \n"
    "\tPRIMARY KEY (artifact_id, source, source_key)\n"
    ")\n WITHOUT ROWID",
    f"CREATE TABLE {CHANGES_TABLE} (\n"
    "\tsource INTEGER NOT NULL, \n"
    "\tsource_key VARCHAR(80) NOT NULL, \n"
    "\taudit INTEGER NOT NULL, \n"
    "\tPRIMARY KEY (source, source_key, audit)\n"
    ")\n WITHOUT ROWID",
    f"CREATE TABLE {STAMP_TABLE} (\n"
    "\tid INTEGER NOT NULL, \n"
    "\tformat_version INTEGER, \n"
    "\tsqlite_version VARCHAR(32), \n"
    "\tschema_version INTEGER, \n"
    "\talembic_revision VARCHAR(32), \n"
    "\tPRIMARY KEY (id)\n"
    ")",
)

CREATE_INDEX_SQL: Final = (
    f"CREATE INDEX ix_artifact_reference_edges_source ON {EDGES_TABLE} (source, source_key)",
    f"CREATE INDEX ix_artifact_reference_changes_audit ON {CHANGES_TABLE} (audit)",
)

#: The stamp has exactly one row. It is inserted before the authority triggers
#: exist, and from then on only the writer window may touch it.
SEED_STAMP_SQL: Final = (
    f"INSERT INTO {STAMP_TABLE} (id) SELECT 1 "
    f"WHERE NOT EXISTS (SELECT 1 FROM {STAMP_TABLE} WHERE id = 1)"
)


def _record(source: int, key: str, audit: str = "0", when: str = "") -> str:
    """Record one key unless that very record is already there.

    There is no conflict clause and so no conflict: an outer statement's own
    clause, which SQLite applies inside the trigger instead, has nothing here
    to act on.
    """

    condition = f"{when} AND " if when else ""
    return (
        f"INSERT INTO {CHANGES_TABLE} (source, source_key, audit)\n"
        f"    SELECT {source}, {key}, {audit} WHERE {condition}NOT EXISTS (\n"
        f"      SELECT 1 FROM {CHANGES_TABLE}\n"
        f"      WHERE source = {source} AND source_key = {key} AND audit = {audit});"
    )


def _change_triggers(source: int) -> tuple[str, str, str]:
    table, _columns = SOURCE_COLUMNS[source]
    documents = tuple(column for column, _root in _TABLE_COLUMNS[table])
    studio = source == SOURCE_CHATS
    watched = ("id", *documents, "scope") if studio else ("id", *documents)
    insert_when = "WHEN NEW.scope = 'studio'\n" if studio else ""
    update_when = "WHEN OLD.scope = 'studio' OR NEW.scope = 'studio'\n" if studio else ""
    delete_when = "WHEN OLD.scope = 'studio'\n" if studio else ""
    name = f"{table}_artifact_reference_change"
    # The write guards do not watch `id`, so a row whose id changes arrives at
    # its new key unchecked: that key is also recorded as due its check.
    moved = "OLD.id IS NOT NEW.id"
    return (
        f"CREATE TRIGGER {name}_insert\nAFTER INSERT ON {table}\n{insert_when}"
        f"BEGIN\n  {_record(source, 'NEW.id')}\nEND",
        f"CREATE TRIGGER {name}_update\nAFTER UPDATE OF {', '.join(watched)} ON {table}\n"
        f"{update_when}BEGIN\n  {_record(source, 'NEW.id')}\n"
        f"  {_record(source, 'NEW.id', '1', moved)}\n"
        f"  {_record(source, 'OLD.id', '0', moved)}\nEND",
        f"CREATE TRIGGER {name}_delete\nAFTER DELETE ON {table}\n{delete_when}"
        f"BEGIN\n  {_record(source, 'OLD.id')}\nEND",
    )


CHANGE_TRIGGER_SQL: Final = tuple(
    trigger for source in SOURCE_COLUMNS for trigger in _change_triggers(source)
)


def _authority_trigger(table: str, action: str) -> str:
    return (
        f"CREATE TRIGGER {table}_{action.lower()}_authority\n"
        f"BEFORE {action} ON {table}\n"
        f"WHEN {ARTIFACT_REFERENCE_INDEX_WRITER_SQL}() IS NOT 1\n"
        "BEGIN\n"
        "  SELECT RAISE(ABORT, 'artifact reference index write is not authorized');\n"
        "END"
    )


AUTHORITY_TRIGGER_SQL: Final = (
    *(_authority_trigger(EDGES_TABLE, action) for action in ("INSERT", "UPDATE", "DELETE")),
    # Recording a change stays open to every connection: the source tables'
    # triggers do it, and a record that did not need making only costs a
    # second reading of its row.
    *(_authority_trigger(CHANGES_TABLE, action) for action in ("UPDATE", "DELETE")),
    *(_authority_trigger(STAMP_TABLE, action) for action in ("INSERT", "UPDATE", "DELETE")),
)

CREATE_TRIGGER_SQL: Final = (*CHANGE_TRIGGER_SQL, *AUTHORITY_TRIGGER_SQL)

#: The trigger whose re-creation moves the schema cookie when it must move.
#: Any canonical trigger would do; this one guards nothing a reader relies on
#: between its drop and its creation, because both run in one transaction.
COOKIE_TRIGGER: Final = AUTHORITY_TRIGGER_SQL[-2]


def trigger_name(statement: str) -> str:
    return statement.split(None, 3)[2]


def with_keys(statement: str, count: int) -> str:
    """The statement with room for `count` bound keys where it takes its list."""

    if count < 1:
        raise ValueError("a key list needs at least one key")
    return statement.replace(KEYS, ", ".join("?" * count))


DROP_TRIGGER_SQL: Final = tuple(
    f"DROP TRIGGER IF EXISTS {trigger_name(statement)}" for statement in CREATE_TRIGGER_SQL
)

#: Every trigger whose exact text the index relies on. The write guards keep
#: every stored document valid, which is what lets a deletion skip checking
#: them all again; the change and authority triggers keep the index exact.
REQUIRED_TRIGGER_SQL: Final = (
    *JSON_WRITE_TRIGGER_SQL,
    ARTIFACT_JSON_DELETE_TRIGGER,
    *CREATE_TRIGGER_SQL,
)


def _row_invalid_sql(source: int) -> str:
    table, _columns = SOURCE_COLUMNS[source]
    structural = [
        _common_invalid(f"{table}.{column}", root_type, depth=_member_depth)
        for column, root_type in _TABLE_COLUMNS[table]
    ]
    semantic = list(_table_invalid(table, table))
    invalid = (
        f"CASE WHEN {' OR '.join(structural)} THEN 1 WHEN {' OR '.join(semantic)} THEN 1 ELSE 0 END"
    )
    scope = "scope = 'studio' AND " if source == SOURCE_CHATS else ""
    return f"SELECT id FROM {table} WHERE id IN ({KEYS}) AND {scope}({invalid})"


#: The stored-document check a deletion runs over the whole store, run here for
#: the named rows of one source only. A row it refuses is recorded as
#: underivable, so the index refuses exactly where that check does.
ROW_INVALID_SQL: Final = {source: _row_invalid_sql(source) for source in SOURCE_COLUMNS}


def _rows_sql(source: int) -> str:
    table, columns = SOURCE_COLUMNS[source]
    return f"SELECT id, {', '.join(columns)} FROM {table} WHERE id IN ({KEYS})"


#: Each source's named rows with the columns `row_edges` reads, as stored text.
ROWS_SQL: Final = {source: _rows_sql(source) for source in SOURCE_COLUMNS}


def _seed_select(source: int) -> str:
    table, _columns = SOURCE_COLUMNS[source]
    scope = " WHERE scope = 'studio'" if source == SOURCE_CHATS else ""
    return f"SELECT {source}, id, 1 FROM {table}{scope}"


DELETE_EDGES_SQL: Final = f"DELETE FROM {EDGES_TABLE} WHERE source = ? AND source_key IN ({KEYS})"
DELETE_CHANGES_SQL: Final = (
    f"DELETE FROM {CHANGES_TABLE} WHERE source = ? AND source_key IN ({KEYS})"
)
EDGES_OF_SQL: Final = (
    f"SELECT artifact_id, source_key FROM {EDGES_TABLE} WHERE source = ? AND source_key IN ({KEYS})"
)
#: Which of the named keys already carry the unreadable marker.
MARKED_SQL: Final = (
    f"SELECT source_key FROM {EDGES_TABLE} "
    f"WHERE artifact_id = '' AND source = ? AND source_key IN ({KEYS})"
)
INSERT_EDGE_SQL: Final = (
    f"INSERT INTO {EDGES_TABLE} (artifact_id, source, source_key) VALUES (?, ?, ?)"
)
RECORD_CHANGE_SQL: Final = (
    f"INSERT OR IGNORE INTO {CHANGES_TABLE} (source, source_key, audit) VALUES (?, ?, ?)"
)

#: Every source row, recorded as changed and due its stored-document check.
SEED_CHANGES_SQL: Final = (
    f"INSERT INTO {CHANGES_TABLE} (source, source_key, audit) "
    + " UNION ALL ".join(_seed_select(source) for source in SOURCE_COLUMNS)
)
