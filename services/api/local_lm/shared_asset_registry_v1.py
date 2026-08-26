"""Neutral claims registry for the Shared Asset Library (item 58, Phase 0).

The registry records which opaque consumer holds a claim on which immutable
package. Claims are the deletion authority: bytes may be collected only when
no claim (and no active lease, a later slice) remains. Callers pass an
explicit registry database path; this module never discovers the desktop
library, never touches object bytes, and exposes one consumer's rows only to
that consumer. Short WAL transactions; no transaction is held across any
byte-moving work.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import re
import secrets
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final, NoReturn
from urllib.parse import quote

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryKind,
    create_entry,
    discard_entry,
    link_entry,
    list_entries,
    sync_directory,
)

SCHEMA_ID: Final = "lm-atelier-shared-asset-registry-v1"
REGISTRY_LEAF: Final = "index.sqlite3"
INVALID_REGISTRY: Final = "shared asset registry is invalid"
PROVISIONAL: Final = "provisional"
FINAL: Final = "final"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONSUMER = re.compile(r"^[0-9a-f]{32,64}$")
_CLAIM_ID = re.compile(r"^[0-9a-f]{32}$")
_BUSY_TIMEOUT_MS: Final = 5000


def _lowercase_hex(column: str, *, low: int, high: int) -> str:
    """SQL that is true only for lowercase hex of a bounded length.

    SQLite has no regular expressions, so the test is built by stripping every
    hex digit and requiring nothing to be left. `lower(x) = x` is what refuses
    uppercase, which strip-and-compare alone would not: `replace` is
    case-sensitive, so an uppercase digit survives the stripping and is caught
    by the length test only by accident.

    This is the same shape the Prompt import winners migration uses, and the
    nine cases behind it are pinned by a test rather than trusted.
    """

    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    bounds = (
        f"length({column}) = {low}" if low == high else f"length({column}) BETWEEN {low} AND {high}"
    )
    return f"{bounds} AND lower({column}) = {column} AND {remainder} = ''"


# The exact CREATE statements, byte-for-byte: sqlite_master stores the
# literal text, so validation compares against these same strings.
#
# The claim formats are DATABASE constraints and not only Python ones. The
# predecessor constrained nullability, state and per-consumer uniqueness and
# left the three identifier formats to the entry points, which meant an
# otherwise exact registry could hold a row with a valid consumer and state
# and a claim id no public call could ever have written. claims_for_consumer
# handed that row back, while finalize_claim and release_claim correctly
# refused its format - an accepted claim that can never be addressed, and a
# claim is the deletion authority, so on a well-formed digest it would block
# collection of those bytes for good.
_REGISTRY_META_SQL: Final = (
    "CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT"
)
_PACKAGE_CLAIMS_SQL: Final = (
    "CREATE TABLE package_claims (claim_id TEXT PRIMARY KEY "
    f"CHECK ({_lowercase_hex('claim_id', low=32, high=32)}), "
    "consumer_id TEXT NOT NULL "
    f"CHECK ({_lowercase_hex('consumer_id', low=32, high=64)}), "
    "package_digest TEXT NOT NULL "
    f"CHECK ({_lowercase_hex('package_digest', low=64, high=64)}), "
    "state TEXT NOT NULL CHECK (state IN ('provisional', 'final')), "
    "UNIQUE (consumer_id, package_digest)) STRICT"
)

#: One query rather than a row-by-row read: a registry with many claims should
#: not be pulled into memory to be checked, and COUNT is byte-preserving.
#:
#: COALESCE is load-bearing, not decoration. SQL is three-valued: if any column
#: here is NULL the comparison is NULL, `NOT NULL` is NULL, and a NULL WHERE
#: clause does not match - so the offending row would be counted as fine and
#: the registry would validate. NOT NULL on the columns is not enough to rule
#: that out, because a database can arrive with rows that were written under a
#: different schema. With `PRAGMA writable_schema=ON` the table text is
#: swapped for one without NOT NULL, a row of NULLs is inserted, and the
#: exact CREATE text is put back.
#: The row survives, `PRAGMA integrity_check` then says
#: "NULL value in package_claims.consumer_id", and the stored text this module
#: compares against is byte-identical to ours.
_MALFORMED_CLAIMS_SQL: Final = (
    "SELECT COUNT(*) FROM package_claims WHERE NOT COALESCE("
    f"{_lowercase_hex('claim_id', low=32, high=32)}"
    f" AND {_lowercase_hex('consumer_id', low=32, high=64)}"
    f" AND {_lowercase_hex('package_digest', low=64, high=64)}"
    " AND state IN ('provisional', 'final')"
    ", 0)"
)


_EXPECTED_TABLES: Final = {
    "registry_meta": _REGISTRY_META_SQL,
    "package_claims": _PACKAGE_CLAIMS_SQL,
}
# Stamped at creation, verified on every open: schema markers can be
# counterfeited, so the application id and version are part of identity.
_APPLICATION_ID: Final = 0x4C4D4153
_USER_VERSION: Final = 1


class SharedAssetRegistryError(ValueError):
    """Fixed non-echoing refusal for an unusable registry operation."""


def _invalid() -> NoReturn:
    raise SharedAssetRegistryError(INVALID_REGISTRY)


@dataclasses.dataclass(frozen=True)
class PackageClaim:
    """One consumer's durable hold on one immutable package."""

    claim_id: str
    consumer_id: str
    package_digest: str
    state: str


def _require_database_location(database: Path) -> tuple[Path, str]:
    """Split the caller's path into a directory to hold and a leaf name.

    These checks concern the ARGUMENT and nothing else: that it is a path,
    absolute, free of NUL, and not a UNC share. They are deliberately not
    described as containment. The predecessor validated exactly this much and
    then USED the path, which let a junctioned parent redirect creation
    without needing a race at all: reserve_claim on a junction's
    index.sqlite3 returned with no exception and the foreign directory
    gained the database. Containment is the
    anchor's job below; syntax was never going to do it.
    """

    if not isinstance(database, Path):
        _invalid()
    try:
        chosen = database.expanduser()
    except (OSError, RuntimeError, ValueError):
        _invalid()
    text = str(chosen)
    if not text or "\x00" in text or not chosen.is_absolute():
        _invalid()
    if text.startswith("\\\\") or chosen.as_posix().startswith("//"):
        _invalid()
    parent = chosen.parent
    leaf = chosen.name
    if not leaf or parent == chosen:
        _invalid()
    return parent, leaf


def _require_consumer(consumer_id: str) -> str:
    if not isinstance(consumer_id, str) or not _CONSUMER.fullmatch(consumer_id):
        _invalid()
    return consumer_id


def _require_digest(digest: str) -> str:
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        _invalid()
    return digest


def _build_registry_bytes() -> bytes:
    """The complete schema, as the bytes a database file would hold.

    Built in memory on purpose. The predecessor created the database at a
    sibling path and linked it into place, which meant SQLite itself opened,
    wrote and fsynced a file inside a directory nothing had verified. Here
    the only thing that ever touches the shared directory is one exclusive
    create through the anchor, and what it writes is already complete.
    """

    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version={_USER_VERSION}")
        connection.execute(_REGISTRY_META_SQL)
        connection.execute(_PACKAGE_CLAIMS_SQL)
        connection.execute(
            "INSERT INTO registry_meta (key, value) VALUES ('schema', ?)",
            (SCHEMA_ID,),
        )
        connection.commit()
        return connection.serialize()
    except sqlite3.Error:
        _invalid()
    finally:
        connection.close()


def _create_new_registry(anchor: AnchoredDirectory, leaf: str) -> None:
    """Stage and publish the registry through the held directory itself.

    No reader observes a half-created registry: the finished bytes are
    written to an exclusively created staging entry, flushed, and only then
    linked under the real name. Publication is create-only, so losing the
    race to another creator is convergence rather than failure - whoever won
    wrote the same schema, and validation adopts it.

    Staging uses an unpredictable name: a fixed sibling can be planted in
    advance, and the plant would then be what gets published as the registry.
    """

    payload = _build_registry_bytes()
    staging = f".{leaf}.creating-{secrets.token_hex(8)}"
    try:
        descriptor = create_entry(anchor, staging)
    except AnchoredDirectoryError:
        _invalid()
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        link_entry(anchor, staging, leaf)
        sync_directory(anchor)
    except (AnchoredDirectoryError, OSError):
        _invalid()
    finally:
        with contextlib.suppress(AnchoredDirectoryError):
            discard_entry(anchor, staging)


def _validate_registry(path: Path) -> None:
    """Prove the database is OURS before any writable open.

    Read-only by construction (a mode=ro URI connection): a foreign,
    counterfeit, or corrupt database is refused byte-preserved. The marker
    alone proves nothing - application id, schema version, the exact stored
    CREATE text of every table, and the complete object set must all match.
    """

    # The path is DATA inside a URI, so it is escaped rather than dropped in.
    # Unescaped, a "#" in the caller's directory name starts a URI fragment and
    # the filename silently ends early - measured: the registry is created
    # through the anchor, and then every call refuses validation on a database
    # that is right there on disk. "?" does the same and is a legal directory
    # name on Linux, which is the always-on CI leg. The drive colon and the
    # separators are the only characters that are structure here.
    uri = "file:" + quote(path.as_posix(), safe="/:") + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=_BUSY_TIMEOUT_MS / 1000)
    except sqlite3.Error:
        _invalid()
    try:
        application_id = connection.execute("PRAGMA application_id").fetchone()
        user_version = connection.execute("PRAGMA user_version").fetchone()
        if application_id is None or application_id[0] != _APPLICATION_ID:
            _invalid()
        if user_version is None or user_version[0] != _USER_VERSION:
            _invalid()
        rows = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY name"
        ).fetchall()
        tables = {name: sql for kind, name, sql in rows if kind == "table"}
        if tables != _EXPECTED_TABLES:
            _invalid()
        for kind, name, _sql in rows:
            if kind == "table":
                continue
            if kind == "index" and name.startswith("sqlite_autoindex_"):
                continue
            _invalid()
        marker = connection.execute(
            "SELECT value FROM registry_meta WHERE key = 'schema'"
        ).fetchone()
        if marker is None or marker[0] != SCHEMA_ID:
            _invalid()
        # The SCHEMA being exact does not make the DATA well formed. A CHECK
        # constraint is enforced when a row is written, and a writer can turn
        # that off - `PRAGMA ignore_check_constraints=ON` inserts straight past
        # every constraint above, and a table built by hand need never have had
        # them. So the rows are checked here too, under this same read-only
        # connection, before anything opens the database for writing.
        #
        # Not belt and braces: the constraints stop US and any ordinary writer
        # from creating such a row, and this stops us from ADOPTING one.
        malformed = connection.execute(_MALFORMED_CLAIMS_SQL).fetchone()
        if malformed is None or malformed[0] != 0:
            _invalid()
    except sqlite3.Error:
        _invalid()
    finally:
        connection.close()


@contextlib.contextmanager
def _registry(database: Path) -> Iterator[sqlite3.Connection]:
    """Hold the registry's directory itself for the whole operation.

    Every public entry point goes through here, and the anchor is what makes
    the caller's path mean something. `AnchoredDirectory` walks the ancestry
    component by component and refuses a reparse point at any depth - on
    POSIX because every component is opened O_NOFOLLOW, on Windows because
    each opened handle is checked. So a junctioned parent is refused before
    a single byte is written, rather than after the file exists.

    One honest limit, stated rather than implied. SQLite cannot be handed a
    descriptor, so the connection below is opened by path while the anchor is
    held. On Windows that is a real guarantee: the chain is held without
    FILE_SHARE_DELETE, so no ancestor can be renamed or deleted, and the
    directory is non-empty by then, so it cannot be converted to a junction
    in place either. On POSIX, holding a descriptor does not prevent an
    ancestor from being renamed, so that last open is not proof against a
    concurrent rename of a directory that was verified a moment earlier.
    What both platforms do get is that creation and publication never touch a
    directory the anchor did not verify.
    """

    parent, leaf = _require_database_location(database)
    try:
        anchor = AnchoredDirectory(parent)
    except AnchoredDirectoryError:
        _invalid()
    with anchor:
        # What the registry name IS, from the directory's own enumeration
        # record rather than from a second lookup by path.
        #
        # open_entry is not enough here, and the difference is measurable. On
        # Windows every anchored open carries FILE_OPEN_REPARSE_POINT, so a
        # symlink under the registry name opens as itself - and fstat calls it
        # a regular file, because it is not a directory. The descriptor comes
        # back, the entry looks present and ordinary, and the connection below
        # then follows the link by path into somebody else's database. The
        # listing is the only place the LINK kind survives.
        try:
            listed = {entry.name: entry for entry in list_entries(anchor)}
        except AnchoredDirectoryError:
            _invalid()
        present = listed.get(leaf)
        if present is None:
            _create_new_registry(anchor, leaf)
        elif present.kind is not AnchoredEntryKind.FILE:
            # A link, a directory, or a kind the filesystem would not name.
            # Refusing beats adopting whatever it points at.
            _invalid()
        path = parent / leaf
        _validate_registry(path)
        try:
            connection = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_MS / 1000)
        except sqlite3.Error:
            _invalid()
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            # The database is published as finished bytes, which carry the
            # rollback journal mode a serialized in-memory database has, so
            # every connection asks for WAL and the first one to get the lock
            # makes it stick - it is persistent thereafter.
            #
            # Best-effort ON PURPOSE. Switching journal mode needs a lock no
            # other connection is holding, so asking for it unconditionally
            # turns an ordinary concurrent open into a refusal: the existing
            # concurrency test failed exactly this way, with two reservers of
            # one package and "database is locked" surfacing as an invalid
            # registry. The same is true of a registry on read-only media.
            # Failing to switch costs concurrency, never correctness - the
            # rollback journal is still a correct journal, and busy_timeout
            # above covers the contention.
            with contextlib.suppress(sqlite3.Error):
                connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            connection.close()
            _invalid()
        try:
            yield connection
        finally:
            connection.close()


def reserve_claim(*, database: Path, consumer_id: str, package_digest: str) -> PackageClaim:
    """Reserve an idempotent provisional claim before local binding commits.

    Reserving over an existing claim returns that claim unchanged - a crashed
    consumer retrying its publication converges instead of duplicating, and a
    finalized claim is never demoted back to provisional.
    """

    consumer = _require_consumer(consumer_id)
    digest = _require_digest(package_digest)
    with _registry(database) as connection:
        try:
            with connection:
                # Insert-or-ignore then read makes concurrent reservers converge
                # on one row instead of colliding on the uniqueness constraint.
                connection.execute(
                    "INSERT INTO package_claims"
                    " (claim_id, consumer_id, package_digest, state)"
                    " VALUES (?, ?, ?, ?)"
                    " ON CONFLICT(consumer_id, package_digest) DO NOTHING",
                    (uuid.uuid4().hex, consumer, digest, PROVISIONAL),
                )
                row = connection.execute(
                    "SELECT claim_id, state FROM package_claims"
                    " WHERE consumer_id = ? AND package_digest = ?",
                    (consumer, digest),
                ).fetchone()
                if row is None:
                    _invalid()
                return PackageClaim(
                    claim_id=row[0],
                    consumer_id=consumer,
                    package_digest=digest,
                    state=row[1],
                )
        except sqlite3.Error:
            _invalid()


def finalize_claim(*, database: Path, consumer_id: str, claim_id: str) -> PackageClaim:
    """Promote one consumer's provisional claim after its binding is durable."""

    consumer = _require_consumer(consumer_id)
    if not isinstance(claim_id, str) or not _CONSUMER.fullmatch(claim_id):
        _invalid()
    with _registry(database) as connection:
        try:
            with connection:
                row = connection.execute(
                    "SELECT package_digest, state FROM package_claims"
                    " WHERE claim_id = ? AND consumer_id = ?",
                    (claim_id, consumer),
                ).fetchone()
                if row is None:
                    _invalid()
                if row[1] != FINAL:
                    updated = connection.execute(
                        "UPDATE package_claims SET state = ?"
                        " WHERE claim_id = ? AND consumer_id = ? AND state = ?",
                        (FINAL, claim_id, consumer, PROVISIONAL),
                    )
                    if updated.rowcount != 1:
                        _invalid()
                return PackageClaim(
                    claim_id=claim_id,
                    consumer_id=consumer,
                    package_digest=row[0],
                    state=FINAL,
                )
        except sqlite3.Error:
            _invalid()


def release_claim(*, database: Path, consumer_id: str, claim_id: str) -> None:
    """Release one consumer's claim. Bytes are never touched here.

    Releasing an absent claim refuses: a caller that believes it holds a
    claim it does not hold has a bookkeeping defect worth surfacing, not
    papering over.
    """

    consumer = _require_consumer(consumer_id)
    if not isinstance(claim_id, str) or not _CONSUMER.fullmatch(claim_id):
        _invalid()
    with _registry(database) as connection:
        try:
            with connection:
                deleted = connection.execute(
                    "DELETE FROM package_claims WHERE claim_id = ? AND consumer_id = ?",
                    (claim_id, consumer),
                )
                if deleted.rowcount != 1:
                    _invalid()
        except sqlite3.Error:
            _invalid()


def claims_for_consumer(*, database: Path, consumer_id: str) -> list[PackageClaim]:
    """List exactly one consumer's claims; no cross-consumer enumeration."""

    consumer = _require_consumer(consumer_id)
    with _registry(database) as connection:
        try:
            rows = connection.execute(
                "SELECT claim_id, package_digest, state FROM package_claims"
                " WHERE consumer_id = ? ORDER BY package_digest",
                (consumer,),
            ).fetchall()
            return [
                PackageClaim(
                    claim_id=row[0],
                    consumer_id=consumer,
                    package_digest=row[1],
                    state=row[2],
                )
                for row in rows
            ]
        except sqlite3.Error:
            _invalid()


def package_is_claimed(*, database: Path, package_digest: str) -> bool:
    """True while ANY consumer claims the package - the deletion gate.

    This is deliberately the only cross-consumer view, and it exposes one
    bit: whether collection is forbidden. Which consumer holds the claim
    stays private.
    """

    digest = _require_digest(package_digest)
    with _registry(database) as connection:
        try:
            row = connection.execute(
                "SELECT 1 FROM package_claims WHERE package_digest = ? LIMIT 1",
                (digest,),
            ).fetchone()
            return row is not None
        except sqlite3.Error:
            _invalid()
