"""Opaque per-profile claims on Shared Asset Library objects (item 58).

Callers pass an explicit root and consumer id. This module never discovers the
desktop library and never returns another consumer's rows. Bytes stay until a
later collection slice. No API mount, Settings, migrate, or leases.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn

from .shared_asset_store_v1 import SharedAssetStoreError, object_path

SCHEMA_ID: Final = "lm-atelier-shared-asset-claim-v1"
SCHEMA_VERSION: Final = 1
INVALID_CLAIM: Final = "shared asset claim is invalid"
INDEX_NAME: Final = "index.sqlite3"
_TOKEN = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1024 * 1024
_TIMEOUT = 5.0


class SharedAssetClaimError(ValueError):
    """Fixed non-echoing refusal for an unusable claim operation."""


@dataclass(frozen=True)
class SharedAssetClaim:
    claim_id: str
    digest: str


def _invalid() -> NoReturn:
    raise SharedAssetClaimError(INVALID_CLAIM)


def _is_unc(path: Path) -> bool:
    text = str(path)
    return text.startswith("\\\\") or text.startswith("//") or path.as_posix().startswith("//")


def _is_link(path: Path) -> bool:
    try:
        return path.is_symlink() or os.path.islink(path)
    except OSError:
        _invalid()


def _require_token(value: object) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        _invalid()
    return value


def _require_digest(value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _invalid()
    return value


def _require_store_root(root: Path) -> Path:
    if not isinstance(root, Path):
        _invalid()
    try:
        chosen = root.expanduser()
    except (OSError, RuntimeError, ValueError):
        _invalid()
    if not str(chosen) or _is_unc(chosen) or not chosen.is_absolute():
        _invalid()
    try:
        if not chosen.is_dir() or _is_link(chosen):
            _invalid()
    except OSError:
        _invalid()
    return chosen


def _require_published_object(*, root: Path, digest: str) -> Path:
    try:
        destination = object_path(root=root, digest=digest)
    except SharedAssetStoreError:
        _invalid()
    try:
        if not destination.is_file() or _is_link(destination):
            _invalid()
        hasher = hashlib.sha256()
        with destination.open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                hasher.update(chunk)
    except OSError:
        _invalid()
    if hasher.hexdigest() != digest:
        _invalid()
    return destination


def _connect(root: Path, *, create: bool) -> sqlite3.Connection | None:
    index = root / INDEX_NAME
    if _is_unc(index) or _is_link(index):
        _invalid()
    try:
        if index.exists() and not index.is_file():
            _invalid()
        if not index.exists() and not create:
            return None
        connection = sqlite3.connect(str(index), isolation_level=None, timeout=_TIMEOUT)
    except sqlite3.Error:
        _invalid()
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                consumer_id TEXT NOT NULL,
                digest TEXT NOT NULL,
                UNIQUE (consumer_id, digest)
            )
            """
        )
    except sqlite3.Error:
        connection.close()
        _invalid()
    return connection


def _rollback(connection: sqlite3.Connection) -> None:
    with contextlib.suppress(sqlite3.Error):
        connection.execute("ROLLBACK")


def claim_object(*, root: Path, consumer_id: str, digest: str) -> str:
    """Reserve or return this consumer's claim on a published object."""

    store = _require_store_root(root)
    consumer = _require_token(consumer_id)
    chosen = _require_digest(digest)
    _require_published_object(root=store, digest=chosen)
    connection = _connect(store, create=True)
    if connection is None:
        _invalid()
    try:
        existing = connection.execute(
            "SELECT claim_id FROM claims WHERE consumer_id = ? AND digest = ?",
            (consumer, chosen),
        ).fetchone()
        if existing is not None:
            connection.execute("COMMIT")
            return str(existing[0])
        claim_id = secrets.token_hex(16)
        try:
            connection.execute(
                "INSERT INTO claims (claim_id, consumer_id, digest) VALUES (?, ?, ?)",
                (claim_id, consumer, chosen),
            )
        except sqlite3.IntegrityError:
            raced = connection.execute(
                "SELECT claim_id FROM claims WHERE consumer_id = ? AND digest = ?",
                (consumer, chosen),
            ).fetchone()
            if raced is None:
                _rollback(connection)
                _invalid()
            connection.execute("COMMIT")
            return str(raced[0])
        connection.execute("COMMIT")
        return claim_id
    except sqlite3.Error:
        _rollback(connection)
        _invalid()
    finally:
        connection.close()


def release_claim(*, root: Path, consumer_id: str, claim_id: str) -> None:
    """Drop only this consumer's claim. Physical bytes are left in place."""

    store = _require_store_root(root)
    consumer = _require_token(consumer_id)
    chosen = _require_token(claim_id)
    connection = _connect(store, create=False)
    if connection is None:
        _invalid()
    try:
        cursor = connection.execute(
            "DELETE FROM claims WHERE claim_id = ? AND consumer_id = ?",
            (chosen, consumer),
        )
        if cursor.rowcount != 1:
            _rollback(connection)
            _invalid()
        connection.execute("COMMIT")
    except sqlite3.Error:
        _rollback(connection)
        _invalid()
    finally:
        connection.close()


def claims_for_consumer(*, root: Path, consumer_id: str) -> tuple[SharedAssetClaim, ...]:
    """Return this consumer's claims only."""

    store = _require_store_root(root)
    consumer = _require_token(consumer_id)
    connection = _connect(store, create=False)
    if connection is None:
        return ()
    try:
        rows = connection.execute(
            """
            SELECT claim_id, digest
            FROM claims
            WHERE consumer_id = ?
            ORDER BY claim_id
            """,
            (consumer,),
        ).fetchall()
        connection.execute("COMMIT")
    except sqlite3.Error:
        _rollback(connection)
        _invalid()
    finally:
        connection.close()
    return tuple(
        SharedAssetClaim(claim_id=str(claim_id), digest=str(digest)) for claim_id, digest in rows
    )


def claim_count_for_digest(*, root: Path, digest: str) -> int:
    """Return how many claims hold a digest, without naming consumers."""

    store = _require_store_root(root)
    chosen = _require_digest(digest)
    connection = _connect(store, create=False)
    if connection is None:
        return 0
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM claims WHERE digest = ?",
            (chosen,),
        ).fetchone()
        connection.execute("COMMIT")
    except sqlite3.Error:
        _rollback(connection)
        _invalid()
    finally:
        connection.close()
    if row is None:
        _invalid()
    return int(row[0])
