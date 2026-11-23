"""Keep the artifact reference index exact, and say when it can be trusted.

The index holds, for every stored row that can name an artifact, the ids it
names. The source tables' triggers record each changed row in the same
statement as the change; draining reads each recorded row again with the
reference walk's own readers and replaces its entries. So for every row with
no recorded change, its entries are exactly what the walk would read from it.

That holds only while nothing rewrote the schema since the index was last
rebuilt: a dropped trigger, a migration with foreign keys off, a restored copy
or another SQLite library could each change rows unseen. The stamp records the
schema cookie, the Alembic revision, the SQLite version and the reader format
the index was rebuilt under, and the index is current only while all four
still match. A rebuild records every row as changed and due the same
stored-document check a deletion runs over the whole store, then drains them,
so a row refused before is refused still, one row at a time.

A deletion is proven from the index: `form_reference_snapshot` drains what is
recorded under the writer and answers whether an artifact is retained from the
rows that name it, and the database checks the proven delete again against the
same index. Plain reads come from it too, without writing. The reference walk
that reads every document still answers a plain read while the index is not
current, and the test suite checks every answer here against it.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from contextlib import suppress
from dataclasses import dataclass
from itertools import batched
from typing import Final

from sqlalchemy import Connection
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from .artifact_deletion_authority import (
    ArtifactReferenceDataError,
    ArtifactReferenceIndexStale,
    ArtifactReferenceRefusal,
    ArtifactReferenceSnapshot,
    open_reference_index_writer,
)
from .artifact_reference_extractors import (
    REFERENCE_CORRUPT,
    SOURCE_ARTIFACTS,
    SOURCE_CHATS,
    SOURCE_COLUMNS,
    SOURCE_MESSAGE_REFERENCES,
    UNDERIVABLE,
    row_edges,
)
from .artifact_reference_index_schema import (
    CHANGES_TABLE,
    COOKIE_TRIGGER,
    DELETE_CHANGES_SQL,
    DELETE_EDGES_SQL,
    EDGES_OF_SQL,
    EDGES_TABLE,
    INSERT_EDGE_SQL,
    MARKED_SQL,
    NAMERS_SQL,
    NAMINGS_SQL,
    RECORD_CHANGE_SQL,
    REQUIRED_TRIGGER_SQL,
    RETAINING_COLUMNS,
    REVISION_PARTS_SQL,
    ROOT_EDGES_SQL,
    ROOTED_SQL,
    ROW_INVALID_SQL,
    ROWS_SQL,
    SEED_CHANGES_SQL,
    SEED_STAMP_SQL,
    STAMP_TABLE,
    trigger_name,
    with_keys,
)
from .db import database_is_contended

logger = logging.getLogger(__name__)

#: The version of the readers the index was built with. Changing how any
#: reader in `artifact_reference_extractors` reads a document changes what the
#: index should hold, so it changes this number and every store rebuilds.
REFERENCE_INDEX_FORMAT: Final = 1
#: A store with at most this many source rows is rebuilt where it is found
#: stale, in that transaction; a larger one is rebuilt in the background.
MAX_INLINE_REBUILD_ROWS: Final = 2_000
#: How many recorded changes one transaction drains, and how many keys one
#: statement names. A chunk of the largest rows holds the writer for a
#: fraction of a second.
DRAIN_CHUNK: Final = 500
#: How many recorded changes a reader reads again rather than finding the
#: index behind; past it, the index is not used until they are drained.
MAX_PENDING_OVERLAY: Final = 2_000
#: How many rows of each source one audit reads again.
AUDIT_SAMPLE: Final = 500
#: How many naming artifacts one membership question follows before the whole
#: retained set is read instead.
MAX_REVERSE_SEARCH: Final = 1_000

#: How many complete walks were compared with the index in this process,
#: and how many of them disagreed with it.
shadow_comparisons = 0
shadow_disagreements = 0
#: How many times upkeep failed unexpectedly in this process.
upkeep_failures = 0

Key = object
Edge = tuple[str, int, Key]


class ReferenceIndexUncertified(ArtifactReferenceRefusal):
    """A trigger the index relies on is missing or does not read as written.

    The index stays stale rather than certified over a guard that might let a
    document through unchecked. Only the trigger's name is reported. A
    deletion that meets it is declined as for any other reference refusal.
    """


@dataclass(frozen=True)
class StampState:
    """What the stamp recorded, and what the store is now."""

    recorded: tuple[object, ...] | None
    current: tuple[int, str, int, str | None]

    @property
    def schema_version(self) -> int:
        return self.current[2]

    @property
    def matches(self) -> bool:
        # A copy made through SQLite's backup interface starts at cookie 1,
        # so no stamp ever certifies one; see `prepare_reference_index`.
        return self.schema_version > 1 and self.recorded == self.current


def _alembic_revision(connection: Connection) -> str | None:
    exists = connection.exec_driver_sql(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'alembic_version'"
    ).first()
    if exists is None:
        return None
    revisions = sorted(
        str(value)
        for value in connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalars()
    )
    return ",".join(revisions) if revisions else None


def stamp_state(connection: Connection) -> StampState:
    row = connection.exec_driver_sql(
        "SELECT format_version, sqlite_version, schema_version, alembic_revision "
        f"FROM {STAMP_TABLE} WHERE id = 1"
    ).first()
    schema_version = int(connection.exec_driver_sql("PRAGMA schema_version").scalar_one())
    current = (
        REFERENCE_INDEX_FORMAT,
        sqlite3.sqlite_version,
        schema_version,
        _alembic_revision(connection),
    )
    return StampState(recorded=None if row is None else tuple(row), current=current)


def _known_source(source: object) -> int | None:
    if type(source) is int and source in SOURCE_COLUMNS:
        return source
    return None


def derived_edges(
    connection: Connection,
    source: int,
    keys: list[Key],
    *,
    audited: Iterable[Key] = (),
) -> dict[Key, set[str]]:
    """The ids each named row names now, read as the walk reads it.

    A key with no row, or a chat outside the studio, names nothing. A row any
    reader refuses names only the underivable marker; so does a row in
    `audited` that the stored-document check refuses, whatever else it names.
    """

    found: dict[Key, set[str]] = {}
    audit = set(audited)
    for chunk in batched(keys, DRAIN_CHUNK):
        rows = {
            row[0]: tuple(row[1:])
            for row in connection.exec_driver_sql(with_keys(ROWS_SQL[source], len(chunk)), chunk)
        }
        checked = tuple(key for key in chunk if key in audit and key in rows)
        invalid: set[Key] = set()
        if checked:
            invalid = set(
                connection.exec_driver_sql(
                    with_keys(ROW_INVALID_SQL[source], len(checked)), checked
                ).scalars()
            )
        for key in chunk:
            columns = rows.get(key)
            if columns is None:
                found[key] = set()
                continue
            names = row_edges(source, columns)
            if key in invalid:
                names.add(UNDERIVABLE)
            found[key] = names
    return found


def _marked_keys(connection: Connection, source: int, keys: list[Key]) -> set[Key]:
    """Which of these keys already carry the unreadable marker.

    Such a row is checked again however it came to be recorded, so no record
    lifts a refusal that its stored text still earns.
    """

    marked: set[Key] = set()
    for chunk in batched(keys, DRAIN_CHUNK):
        marked.update(
            connection.exec_driver_sql(
                with_keys(MARKED_SQL, len(chunk)), (source, *chunk)
            ).scalars()
        )
    return marked


def drain_changes(connection: Connection, *, limit: int | None) -> int:
    """Bring the index up to date for recorded changes, at most `limit` of them, or all.

    Runs in the caller's transaction, which must hold the writer: a change
    another connection recorded between reading a row and clearing its record
    would otherwise be cleared unread. Returns how many it drained.
    """

    # Any record that is not exactly 0 asks for the check, whatever its type:
    # anyone may record, and SQLite ranks text and blobs above every number.
    pending = connection.exec_driver_sql(
        f"SELECT source, source_key, max(audit IS NOT 0) FROM {CHANGES_TABLE} "
        "GROUP BY source, source_key ORDER BY source, source_key LIMIT ?",
        (-1 if limit is None else limit,),
    ).all()
    if not pending:
        return 0
    by_source: dict[object, list[tuple[Key, bool]]] = {}
    for source, key, audit in pending:
        by_source.setdefault(source, []).append((key, bool(audit)))
    with open_reference_index_writer(connection):
        for source, entries in by_source.items():
            keys = [key for key, _audit in entries]
            known = _known_source(source)
            if known is not None:
                marked = _marked_keys(connection, known, keys)
                edges = derived_edges(
                    connection,
                    known,
                    keys,
                    audited=[key for key, audit in entries if audit or key in marked],
                )
                for chunk in batched(keys, DRAIN_CHUNK):
                    connection.exec_driver_sql(
                        with_keys(DELETE_EDGES_SQL, len(chunk)), (known, *chunk)
                    )
                rows = [(name, known, key) for key in keys for name in sorted(edges[key])]
                if rows:
                    connection.exec_driver_sql(INSERT_EDGE_SQL, rows)
            # A change recorded under a source the index does not know names
            # nothing the index holds; it is cleared like any other.
            for chunk in batched(keys, DRAIN_CHUNK):
                connection.exec_driver_sql(
                    with_keys(DELETE_CHANGES_SQL, len(chunk)), (source, *chunk)
                )
    return len(pending)


def pending_changes(connection: Connection) -> int:
    return int(connection.exec_driver_sql(f"SELECT count(*) FROM {CHANGES_TABLE}").scalar_one())


def record_changes(connection: Connection, keys: Mapping[int, Iterable[Key]]) -> None:
    """Record these rows as changed and due their check, so they are read again."""

    rows = [(source, key, 1) for source, source_keys in keys.items() for key in source_keys]
    if rows:
        connection.exec_driver_sql(RECORD_CHANGE_SQL, rows)


def _require_canonical_triggers(connection: Connection) -> None:
    installed = {
        str(name): sql
        for name, sql in connection.exec_driver_sql(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    for statement in REQUIRED_TRIGGER_SQL:
        name = trigger_name(statement)
        sql = installed.get(name)
        if not isinstance(sql, str) or sql.strip() != statement.strip():
            raise ReferenceIndexUncertified(
                f"SQLite trigger {name} does not match the application schema"
            )


def certify_and_seed(connection: Connection) -> None:
    """Start a rebuild: every source row recorded as changed and due its check.

    Runs in the caller's write transaction. The stamp is written together with
    the records, so a crash before the drain finishes leaves rows recorded
    rather than an index that only looks complete.
    """

    _require_canonical_triggers(connection)
    state = stamp_state(connection)
    if state.schema_version <= 1:
        raise ReferenceIndexUncertified("the schema cookie has not moved since this copy was made")
    with open_reference_index_writer(connection):
        connection.exec_driver_sql(f"DELETE FROM {EDGES_TABLE}")
        connection.exec_driver_sql(f"DELETE FROM {CHANGES_TABLE}")
        connection.exec_driver_sql(SEED_CHANGES_SQL)
        connection.exec_driver_sql(SEED_STAMP_SQL)
        connection.exec_driver_sql(
            f"UPDATE {STAMP_TABLE} SET format_version = ?, sqlite_version = ?, "
            "schema_version = ?, alembic_revision = ? WHERE id = 1",
            state.current,
        )


def clear_stamp(connection: Connection) -> None:
    with open_reference_index_writer(connection):
        connection.exec_driver_sql(
            f"UPDATE {STAMP_TABLE} SET format_version = NULL, sqlite_version = NULL, "
            "schema_version = NULL, alembic_revision = NULL WHERE id = 1"
        )


def _move_schema_cookie(connection: Connection) -> None:
    # DDL is the only safe way to move the cookie. Re-creating one trigger with
    # its own canonical text, in one transaction, changes nothing else.
    connection.exec_driver_sql(f"DROP TRIGGER IF EXISTS {trigger_name(COOKIE_TRIGGER)}")
    connection.exec_driver_sql(COOKIE_TRIGGER)


def source_rows(connection: Connection, limit: int) -> int:
    """How many source rows the store holds, counting no further than past `limit`."""

    total = 0
    for table, _columns in SOURCE_COLUMNS.values():
        scope = " WHERE scope = 'studio'" if table == "chats" else ""
        total += int(
            connection.exec_driver_sql(
                f"SELECT count(*) FROM (SELECT 1 FROM {table}{scope} LIMIT ?)",
                (limit - total + 1,),
            ).scalar_one()
        )
        if total > limit:
            break
    return total


def certify_if_empty(connection: Connection) -> bool:
    """Certify a store that has never held a row, as a freshly created schema is.

    Nothing to read means nothing to drain, so such a store is current from
    the moment it exists. A store whose triggers do not read as written is
    left stale. Says whether it certified.
    """

    if stamp_state(connection).recorded != (None, None, None, None):
        return False
    if source_rows(connection, 0) or pending_changes(connection):
        return False
    if connection.exec_driver_sql(f"SELECT 1 FROM {EDGES_TABLE} LIMIT 1").first() is not None:
        return False
    try:
        certify_and_seed(connection)
    except ReferenceIndexUncertified:
        return False
    return True


def rebuild_inline(connection: Connection) -> None:
    """Rebuild a small store completely, in the caller's write transaction."""

    certify_and_seed(connection)
    drain_changes(connection, limit=None)


def _begin_writer(session: Session) -> Connection:
    connection = session.connection()
    driver = connection.connection.driver_connection
    if not bool(getattr(driver, "in_transaction", False)):
        connection.exec_driver_sql("BEGIN IMMEDIATE")
    return connection


def _reset_copy(session: Session) -> Connection:
    """Take the writer; first clear a backup copy's stamp and move its cookie.

    A copy made through SQLite's backup interface starts at schema cookie 1,
    whatever its stamp says. Clearing the stamp and moving the cookie, in one
    committed transaction, has the copy rebuilt rather than trusted.
    """

    connection = _begin_writer(session)
    if stamp_state(connection).schema_version <= 1:
        clear_stamp(connection)
        _move_schema_cookie(connection)
        session.commit()
        connection = _begin_writer(session)
    return connection


def prepare_reference_index(session: Session) -> bool:
    """Make the index current at startup where that is quick, and say whether it is.

    A copy made through SQLite's backup interface starts at schema cookie 1,
    whatever its stamp says, so its stamp is cleared and the cookie moved
    first: a copy is rebuilt, never trusted. A small store is then rebuilt
    here; a large one is left to `maintain_reference_index`, which runs once
    the port is open. Commits what it changes.
    """

    connection = _reset_copy(session)
    current = stamp_state(connection).matches
    if not current and source_rows(connection, MAX_INLINE_REBUILD_ROWS) <= MAX_INLINE_REBUILD_ROWS:
        rebuild_inline(connection)
        current = True
    session.commit()
    return current


@dataclass(frozen=True)
class MaintenanceStep:
    drained: int
    remaining: int
    rebuilt: bool


def maintenance_step(session: Session) -> MaintenanceStep:
    """One write transaction of upkeep: start a rebuild if stale, then drain a chunk.

    A current store with nothing recorded is read and left, without taking the
    writer, so idle upkeep never makes another writer wait.
    """

    connection = session.connection()
    if stamp_state(connection).matches and not pending_changes(connection):
        session.rollback()
        return MaintenanceStep(drained=0, remaining=0, rebuilt=False)
    session.rollback()
    # A copy restored after startup, or one startup could not reset, is reset
    # here, so upkeep never stops on it.
    connection = _reset_copy(session)
    rebuilt = False
    if not stamp_state(connection).matches:
        certify_and_seed(connection)
        rebuilt = True
    drained = drain_changes(connection, limit=DRAIN_CHUNK)
    remaining = pending_changes(connection)
    session.commit()
    return MaintenanceStep(drained=drained, remaining=remaining, rebuilt=rebuilt)


SessionFactory = Callable[[], Session]


def start_reference_index(factory: SessionFactory) -> bool:
    """`prepare_reference_index` for application startup, which it never stops.

    Whatever goes wrong is logged and left to `maintain_reference_index`: the
    index decides nothing yet, so a start must not wait on it or fail with it.
    """

    global upkeep_failures
    try:
        with factory() as session:
            try:
                return prepare_reference_index(session)
            except BaseException:
                session.rollback()
                raise
    except ReferenceIndexUncertified as exc:
        logger.error("Artifact reference index cannot be certified: %s", exc)
    except OperationalError as exc:
        if not database_is_contended(exc):
            upkeep_failures += 1
            logger.exception("Artifact reference index could not be prepared at startup")
        else:
            logger.warning("Artifact reference index waited on the database writer (%s)", exc)
    except Exception:
        upkeep_failures += 1
        logger.exception("Artifact reference index could not be prepared at startup")
    return False


def _run_step(factory: SessionFactory) -> MaintenanceStep:
    with factory() as session:
        try:
            return maintenance_step(session)
        except BaseException:
            session.rollback()
            raise


async def maintain_reference_index(
    factory: SessionFactory,
    *,
    ready: asyncio.Event | None = None,
    poll_seconds: float = 5.0,
    pause_seconds: float = 0.02,
    audit_every: int = 120,
) -> None:
    """Keep the index current for as long as the application runs.

    Rebuilds a stale store a chunk at a time, then drains what writers record.
    Each chunk is its own short write transaction, so another writer waits at
    most one chunk. Sets `ready` once nothing is left to drain. Every
    `audit_every` idle polls it reads a sample of rows again and compares
    them with the index; a row that disagrees is recorded to be read again.
    """

    global upkeep_failures
    loop = asyncio.get_running_loop()
    polls = 0
    cursor: dict[int, Key] = {}
    rebuild_started: float | None = None
    while True:
        # A step in flight cannot be stopped inside SQLite. On cancellation it
        # is waited for, so no upkeep writes to the store after shutdown.
        operation = asyncio.ensure_future(asyncio.to_thread(_run_step, factory))
        try:
            step = await asyncio.shield(operation)
        except asyncio.CancelledError:
            with suppress(Exception):
                await operation
            raise
        except ReferenceIndexUncertified as exc:
            logger.error("Artifact reference index cannot be certified: %s", exc)
            return
        except OperationalError as exc:
            if not database_is_contended(exc):
                upkeep_failures += 1
                logger.exception("Artifact reference index upkeep failed; retrying")
            else:
                logger.warning("Artifact reference index waited on the database writer (%s)", exc)
            await asyncio.sleep(max(poll_seconds, 1.0))
            continue
        except Exception:
            upkeep_failures += 1
            logger.exception("Artifact reference index upkeep failed; retrying")
            await asyncio.sleep(max(poll_seconds, 1.0))
            continue
        if step.rebuilt:
            rebuild_started = loop.time()
            logger.info("Artifact reference index rebuild started: %s row(s)", step.remaining)
        if step.remaining:
            await asyncio.sleep(pause_seconds)
            continue
        if rebuild_started is not None:
            logger.info("Artifact reference index rebuilt in %.3fs", loop.time() - rebuild_started)
            rebuild_started = None
        if ready is not None:
            ready.set()
        polls += 1
        if audit_every and polls % audit_every == 0:
            audit = asyncio.ensure_future(asyncio.to_thread(_run_audit, factory, cursor))
            try:
                await asyncio.shield(audit)
            except asyncio.CancelledError:
                with suppress(Exception):
                    await audit
                raise
            except OperationalError as exc:
                logger.warning("Artifact reference index audit waited on the writer (%s)", exc)
            except Exception:
                upkeep_failures += 1
                logger.exception("Artifact reference index audit failed")
        await asyncio.sleep(poll_seconds)


@dataclass(frozen=True)
class AuditResult:
    examined: int
    missing: int
    extra: int


def _sample_keys(connection: Connection, source: int, after: Key | None, limit: int) -> list[Key]:
    table, _columns = SOURCE_COLUMNS[source]
    scope = "scope = 'studio'" if source == SOURCE_CHATS else "1"
    if after is None:
        statement = f"SELECT id FROM {table} WHERE {scope} ORDER BY id LIMIT ?"
        return list(connection.exec_driver_sql(statement, (limit,)).scalars())
    statement = f"SELECT id FROM {table} WHERE {scope} AND id > ? ORDER BY id LIMIT ?"
    return list(connection.exec_driver_sql(statement, (after, limit)).scalars())


def audit_sample(
    connection: Connection,
    cursor: dict[int, Key],
    *,
    per_source: int = AUDIT_SAMPLE,
) -> tuple[AuditResult, dict[int, list[Key]]]:
    """Read a rotating sample of rows again and compare them with the index.

    Read-only, in the caller's transaction. Rows with a recorded change are
    skipped, since the index does not claim them yet. Returns the counts and,
    by source, the keys that disagreed, for the caller to record as changed.
    """

    examined = missing = extra = 0
    disagreed: dict[int, list[Key]] = {}
    if not stamp_state(connection).matches:
        return AuditResult(0, 0, 0), disagreed
    pending = set(
        connection.exec_driver_sql(f"SELECT source, source_key FROM {CHANGES_TABLE}").tuples()
    )
    for source in SOURCE_COLUMNS:
        sampled = _sample_keys(connection, source, cursor.get(source), per_source)
        if len(sampled) == per_source:
            cursor[source] = sampled[-1]
        else:
            cursor.pop(source, None)
        keys = [key for key in sampled if (source, key) not in pending]
        if not keys:
            continue
        stored: dict[Key, set[str]] = {key: set() for key in keys}
        for chunk in batched(keys, DRAIN_CHUNK):
            for name, key in connection.exec_driver_sql(
                with_keys(EDGES_OF_SQL, len(chunk)), (source, *chunk)
            ):
                stored[key].add(str(name))
        # A row the stored-document check refused carries the marker; read it
        # with that check again, or the marker would look like an extra name.
        audited = [key for key, names in stored.items() if UNDERIVABLE in names]
        derived = derived_edges(connection, source, keys, audited=audited)
        for key in keys:
            examined += 1
            lost = derived[key] - stored[key]
            added = stored[key] - derived[key]
            if lost or added:
                missing += len(lost)
                extra += len(added)
                disagreed.setdefault(source, []).append(key)
    return AuditResult(examined, missing, extra), disagreed


def _run_audit(factory: SessionFactory, cursor: dict[int, Key]) -> AuditResult:
    with factory() as session:
        connection = session.connection()
        connection.exec_driver_sql("BEGIN")
        result, disagreed = audit_sample(connection, cursor)
        session.rollback()
        if disagreed:
            logger.error(
                "Artifact reference index disagreed with %s of %s row(s) read again: "
                "%s name(s) missing, %s extra; they are recorded to be read again",
                sum(len(keys) for keys in disagreed.values()),
                result.examined,
                result.missing,
                result.extra,
            )
            record_changes(_begin_writer(session), disagreed)
            session.commit()
        return result


@dataclass(frozen=True)
class IndexedReferences:
    """The index's answer to what a complete walk computes."""

    retained: frozenset[str]
    namers: Mapping[str, frozenset[str]]


def _retaining_rows(
    connection: Connection,
    *,
    exclude_message: str | None,
    exclude_library: AbstractSet[str],
) -> Iterator[str]:
    """The artifacts rows other than documents retain, with the walk's own exclusions.

    A previewed message's parts and revision parts are left out with `!=`, as
    the walk leaves them out; a released library membership is left out by id.
    """

    for table, column in RETAINING_COLUMNS:
        statement = f"SELECT {column} FROM {table}"
        parameters: tuple[object, ...] = ()
        if table == "message_parts" and exclude_message is not None:
            statement += " WHERE message_id != ?"
            parameters = (exclude_message,)
        for value in connection.exec_driver_sql(statement, parameters).scalars():
            if value and not (table == "artifact_library_entries" and value in exclude_library):
                yield str(value)
    statement = REVISION_PARTS_SQL
    parameters = ()
    if exclude_message is not None:
        statement += " WHERE revision.message_id != ?"
        parameters = (exclude_message,)
    for value in connection.exec_driver_sql(statement, parameters).scalars():
        if value:
            yield str(value)


def _current_edges(connection: Connection) -> list[Edge] | None:
    """Every entry, with rows that have a recorded change read again; None if behind."""

    pending = connection.exec_driver_sql(
        f"SELECT source, source_key, audit IS NOT 0 FROM {CHANGES_TABLE} LIMIT ?",
        (MAX_PENDING_OVERLAY + 1,),
    ).all()
    if len(pending) > MAX_PENDING_OVERLAY or any(audit for _source, _key, audit in pending):
        return None
    overlay: dict[tuple[object, Key], set[str]] = {}
    by_source: dict[object, list[Key]] = {}
    for source, key, _audit in pending:
        by_source.setdefault(source, []).append(key)
    for source, keys in by_source.items():
        known = _known_source(source)
        derived: dict[Key, set[str]] = {}
        if known is not None:
            # Read as a drain would read it, so a waiting record no more lifts
            # a marked row's refusal here than it does there.
            derived = derived_edges(
                connection, known, keys, audited=_marked_keys(connection, known, keys)
            )
        for key in keys:
            overlay[(source, key)] = derived.get(key, set())
    edges: list[Edge] = [
        (str(name), int(source), key)
        for name, source, key in connection.exec_driver_sql(
            f"SELECT artifact_id, source, source_key FROM {EDGES_TABLE}"
        )
        if (source, key) not in overlay
    ]
    for (source, key), names in overlay.items():
        known = _known_source(source)
        if known is not None:
            edges.extend((name, known, key) for name in names)
    return edges


def _closure(roots: Iterable[str], names_by_namer: Mapping[str, Iterable[str]]) -> set[str]:
    retained = set(roots)
    frontier = list(retained)
    while frontier:
        reached: list[str] = []
        for artifact_id in frontier:
            for linked in names_by_namer.get(artifact_id, ()):
                if linked not in retained:
                    retained.add(linked)
                    reached.append(linked)
        frontier = reached
    return retained


def indexed_references(
    connection: Connection,
    *,
    complete: bool = True,
    exclude_message_payload_for: str | None = None,
    exclude_library_membership_for: AbstractSet[str] = frozenset(),
) -> IndexedReferences | None:
    """What the reference walk would retain, from the index, without writing.

    `complete` asks what a deletion's walk asks, which refuses when any row
    cannot be read. Otherwise this answers the plain walk, which refuses only
    for a row it would follow: any document but an artifact's, and an
    artifact's own only when that artifact is retained. The exclusions are
    the walk's own, for previewing a removal.

    None when the index is not current, a rebuild's checks are still due, or
    more changes are recorded than `MAX_PENDING_OVERLAY`. Rows with a recorded
    change are read again rather than trusted, and nothing is written, so this
    never moves the change counter a deletion proof is bound to.
    """

    if complete and (exclude_message_payload_for is not None or exclude_library_membership_for):
        raise ValueError("a selective reference preview cannot answer for a deletion")
    if not stamp_state(connection).matches:
        return None
    edges = _current_edges(connection)
    if edges is None:
        return None
    excluded_references: set[str] = set()
    if exclude_message_payload_for is not None:
        excluded_references = {
            str(key)
            for key in connection.exec_driver_sql(
                "SELECT id FROM message_references WHERE message_id = ?",
                (exclude_message_payload_for,),
            ).scalars()
        }
    roots: set[str] = set(
        _retaining_rows(
            connection,
            exclude_message=exclude_message_payload_for,
            exclude_library=exclude_library_membership_for,
        )
    )
    names_by_namer: dict[str, set[str]] = {}
    namers: dict[str, set[str]] = {}
    unreadable_artifacts: set[str] = set()
    for name, source, key in edges:
        if source == SOURCE_ARTIFACTS:
            if name == UNDERIVABLE:
                if complete:
                    raise ArtifactReferenceDataError(REFERENCE_CORRUPT)
                unreadable_artifacts.add(str(key))
                continue
            names_by_namer.setdefault(str(key), set()).add(name)
            namers.setdefault(name, set()).add(str(key))
            continue
        if source == SOURCE_MESSAGE_REFERENCES and str(key) in excluded_references:
            continue
        if name == UNDERIVABLE:
            raise ArtifactReferenceDataError(REFERENCE_CORRUPT)
        roots.add(name)
    retained = _closure(roots, names_by_namer)
    if unreadable_artifacts & retained:
        raise ArtifactReferenceDataError(REFERENCE_CORRUPT)
    return IndexedReferences(
        retained=frozenset(retained),
        namers={name: frozenset(keys) for name, keys in namers.items()},
    )


class IndexedReferenceSnapshot(ArtifactReferenceSnapshot):
    """A deletion's reference graph answered from the index, a question at a time.

    Formed by `form_reference_snapshot` once the index is current and drained
    under the writer, so the index is exactly what the walk would read. Asking
    whether one artifact is retained follows only the artifacts that name it,
    a few indexed reads, instead of reading the whole store. Answers are kept
    for the snapshot's life: a recorded deletion removes only an unretained
    artifact and the names it gave, which changes no other answer.
    """

    def __init__(
        self, *, session: Session, connection: Connection, driver: sqlite3.Connection
    ) -> None:
        super().__init__(session=session, connection=connection, driver=driver)
        self._answers: dict[str, bool] = {}
        self._retained: frozenset[str] | None = None

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, str):
            return False
        if self._retained is not None:
            return value in self._retained
        answer = self._answers.get(value)
        if answer is None:
            answer = self._search(value)
        return answer

    def __iter__(self) -> Iterator[str]:
        return iter(self.retained())

    def __len__(self) -> int:
        return len(self.retained())

    def _rooted(self, artifact_id: str) -> bool:
        return bool(self._connection.exec_driver_sql(ROOTED_SQL, (artifact_id,)).scalar_one())

    def namers(self, artifact_id: str) -> frozenset[str]:
        return frozenset(
            str(namer)
            for namer in self._connection.exec_driver_sql(NAMERS_SQL, (artifact_id,)).scalars()
        )

    def _search(self, artifact_id: str) -> bool:
        """Retained when a row other than an artifact names it, or a retained artifact does."""

        visited = {artifact_id}
        frontier = [artifact_id]
        while frontier:
            reached: list[str] = []
            for candidate in frontier:
                if self._answers.get(candidate) is True or self._rooted(candidate):
                    self._answers[artifact_id] = True
                    return True
                for namer in self.namers(candidate) - self._removed:
                    if namer not in visited and self._answers.get(namer) is not False:
                        visited.add(namer)
                        reached.append(namer)
            if len(visited) > MAX_REVERSE_SEARCH:
                return artifact_id in self.retained()
            frontier = reached
        # Nothing that names it is retained, so nothing that names those is either.
        for candidate in visited:
            self._answers[candidate] = False
        return False

    def retained(self) -> frozenset[str]:
        """Every retained artifact, read once for a caller that asks about them all."""

        if self._retained is None:
            self._retained = self._read_retained()
        return self._retained

    def _read_retained(self) -> frozenset[str]:
        roots = {
            str(name) for name, _source, _key in self._connection.exec_driver_sql(ROOT_EDGES_SQL)
        }
        roots.update(
            _retaining_rows(self._connection, exclude_message=None, exclude_library=frozenset())
        )
        return frozenset(_closure(roots, self.namings()))

    def namings(self) -> dict[str, set[str]]:
        """Which surviving artifact names which, by the naming artifact."""

        names_by_namer: dict[str, set[str]] = {}
        for name, namer in self._connection.exec_driver_sql(NAMINGS_SQL):
            if str(namer) not in self._removed:
                names_by_namer.setdefault(str(namer), set()).add(str(name))
        return names_by_namer


def form_reference_snapshot(session: Session) -> IndexedReferenceSnapshot | None:
    """This transaction's complete reference graph for a deletion, from the index.

    Takes the writer with a write that changes nothing, so a read snapshot
    another writer has since moved past fails here instead of being proven
    against. A small store found stale is rebuilt in this transaction; a large
    one raises `ArtifactReferenceIndexStale` until the background rebuild
    finishes. Everything recorded is drained here, however much: this
    transaction's own writes are among it, and nothing else can drain those.
    A row that cannot be read refuses every deletion, as the walk's
    whole-store check did. None outside a transaction, where nothing can be
    proven.
    """

    session.flush()
    connection = session.connection()
    driver = connection.connection.driver_connection
    if not isinstance(driver, sqlite3.Connection) or not driver.in_transaction:
        return None
    connection.exec_driver_sql(f"UPDATE {STAMP_TABLE} SET id = id WHERE 0")
    stale = not stamp_state(connection).matches or (
        connection.exec_driver_sql(
            f"SELECT 1 FROM {CHANGES_TABLE} WHERE audit IS NOT 0 LIMIT 1"
        ).first()
        is not None
    )
    if stale:
        if source_rows(connection, MAX_INLINE_REBUILD_ROWS) > MAX_INLINE_REBUILD_ROWS:
            raise ArtifactReferenceIndexStale()
        rebuild_inline(connection)
    drain_changes(connection, limit=None)
    if connection.exec_driver_sql(
        f"SELECT 1 FROM {EDGES_TABLE} WHERE artifact_id = ? LIMIT 1", (UNDERIVABLE,)
    ).first():
        raise ArtifactReferenceRefusal(REFERENCE_CORRUPT)
    return IndexedReferenceSnapshot(session=session, connection=connection, driver=driver)


def drain_after_deletion(session: Session) -> None:
    """Clear what a proven deletion's own flush recorded, before it is recorded as done.

    The database checks every proven deletion against an index with nothing
    left to drain, so the next deletion in the same pass needs this one's
    entries gone first.
    """

    drain_changes(session.connection(), limit=None)


#: Whether every reference answer from the index is also checked against the
#: walk. Set for the test suite, where a disagreement fails the test; the walk
#: costs what the index exists to avoid, so the application leaves it off.
CHECK_AGAINST_WALK = False


#: What the walk answers: the retained ids and, for a deletion's walk, which
#: surviving artifacts name each id.
WalkAnswer = tuple[frozenset[str], Mapping[str, AbstractSet[str]] | None]


def _count_disagreement(message: str) -> None:
    global shadow_disagreements
    shadow_disagreements += 1
    logger.error("Artifact reference index disagreed with the reference walk: %s", message)


def _walk_to_compare(walk: Callable[[], WalkAnswer]) -> WalkAnswer | None:
    """The walk's answer, counted as one comparison; None once its refusal is counted."""

    global shadow_comparisons
    shadow_comparisons += 1
    try:
        return walk()
    except ArtifactReferenceDataError:
        _count_disagreement("the walk refused a store the index answered for")
        return None


def compare_answers(retained: AbstractSet[str], walk: Callable[[], WalkAnswer]) -> None:
    """Check a plain read from the index against the walk's answer to the same question.

    Counts and logs a disagreement and never raises.
    """

    answer = _walk_to_compare(walk)
    if answer is None:
        return
    answered = frozenset(retained)
    if answered != answer[0]:
        _count_disagreement(f"{len(answered ^ answer[0])} retained id(s) differ")


def compare_snapshot(snapshot: IndexedReferenceSnapshot, walk: Callable[[], WalkAnswer]) -> None:
    """Check a deletion's snapshot against the walk, asking it what a deletion asks.

    Every stored artifact, and every id the walk retains or the index names,
    is asked about through a second snapshot of the same transaction, so the
    answers checked are the reverse search's own, and the snapshot handed back
    keeps nothing the check learned. The whole retained set and each id's
    namers are checked too. Counts and logs a disagreement and never raises.
    """

    answer = _walk_to_compare(walk)
    if answer is None:
        return
    walked, referrers = answer
    connection = snapshot._connection
    probe = IndexedReferenceSnapshot(
        session=snapshot._session, connection=connection, driver=snapshot._driver
    )
    retained = probe._read_retained()
    if retained != walked:
        _count_disagreement(f"{len(retained ^ walked)} retained id(s) differ")
        return
    walked_namers = {name: frozenset(keys) for name, keys in (referrers or {}).items() if keys}
    questions = sorted(
        walked
        | set(walked_namers)
        | {str(name) for name, _namer in connection.exec_driver_sql(NAMINGS_SQL)}
        | {str(key) for key in connection.exec_driver_sql("SELECT id FROM artifacts").scalars()}
    )
    searched = [question for question in questions if (question in probe) != (question in walked)]
    if searched:
        _count_disagreement(f"the search answered {len(searched)} id(s) wrongly")
        return
    if referrers is None:
        return
    differing = [
        question
        for question in questions
        if probe.namers(question) != walked_namers.get(question, frozenset())
    ]
    if differing:
        _count_disagreement(f"the namers of {len(differing)} id(s) differ")


def compare_refusal(walk: Callable[[], WalkAnswer]) -> None:
    """The index refused a stored row; check that the walk refuses the same store."""

    global shadow_comparisons
    try:
        walk()
    except ArtifactReferenceDataError:
        shadow_comparisons += 1
        return
    shadow_comparisons += 1
    _count_disagreement("the index refused a store the walk read")
