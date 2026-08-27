"""Renewable, consumer-scoped read leases for shared asset packages.

The registry row is durable bookkeeping. The held operating-system lock is
the authority that the holder still exists; pid and executable are diagnosis
only. Callers pass an explicit registry path and receive no other consumer's
identity or rows.
"""

from __future__ import annotations

import contextlib
import dataclasses
import secrets
import sqlite3
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Final, NoReturn, cast

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    open_child_directory,
)
from .shared_asset_lock_v1 import (
    EntryIdentity,
    SharedAssetLockError,
    current_process_identity,
    hold,
)
from .shared_asset_registry_v1 import (
    FINAL,
    SharedAssetRegistryError,
    _registry,
    _require_consumer,
    _require_digest,
)

INVALID_LEASE: Final = "shared asset lease is invalid"
DEFAULT_TTL_SECONDS: Final = 3600
LOCKS_LEAF: Final = "locks"
_MAX_SQLITE_INTEGER: Final = (1 << 63) - 1


class SharedAssetLeaseError(ValueError):
    """Fixed non-echoing refusal for an unusable lease operation."""


def _invalid() -> NoReturn:
    raise SharedAssetLeaseError(INVALID_LEASE) from None


@dataclasses.dataclass(frozen=True)
class SharedAssetReadLease:
    """One consumer's durable read lease, without lock or path details."""

    lease_id: str
    consumer_id: str
    package_digest: str
    holder_pid: int
    expires_at: int


@dataclasses.dataclass
class _HeldLeaseLock:
    locks: AnchoredDirectory
    manager: AbstractContextManager[tuple[int, EntryIdentity]]
    identity: EntryIdentity

    def close(self) -> None:
        try:
            self.manager.__exit__(None, None, None)
        finally:
            self.locks.close()


_HELD: dict[tuple[str, str], _HeldLeaseLock] = {}
_HELD_GUARD = threading.RLock()


def _require_lease_id(value: str) -> str:
    if not isinstance(value, str) or len(value) != 32:
        _invalid()
    if value.lower() != value or any(character not in "0123456789abcdef" for character in value):
        _invalid()
    return value


def _clock(now: int | None, ttl: int) -> tuple[int, int]:
    moment = int(time.time()) if now is None else now
    if type(moment) is not int or type(ttl) is not int:
        _invalid()
    if moment < 0 or ttl <= 0:
        _invalid()
    expires_at = moment + ttl
    if expires_at > _MAX_SQLITE_INTEGER:
        _invalid()
    return moment, expires_at


def _key(database: Path, lease_id: str) -> tuple[str, str]:
    return str(database), lease_id


def _lock_name(lease_id: str) -> str:
    return f"lease-{lease_id}.lock"


def _row_identity(row: sqlite3.Row | tuple[object, ...], offset: int) -> EntryIdentity:
    try:
        device = row[offset]
        inode = row[offset + 1]
        token = row[offset + 2]
        if (
            not isinstance(device, str)
            or not isinstance(inode, str)
            or not isinstance(token, bytes)
        ):
            _invalid()
        if not 1 <= len(device) <= 64 or not 1 <= len(inode) <= 64:
            _invalid()
        if device.lower() != device or inode.lower() != inode:
            _invalid()
        if any(character not in "0123456789abcdef" for character in device + inode):
            _invalid()
        identity = (int(device, 16), int(inode, 16), token)
    except (IndexError, TypeError, ValueError):
        _invalid()
    if identity[0] < 0 or identity[1] < 0 or len(identity[2]) != 16:
        _invalid()
    return identity


def _take_lock(
    database: Path, lease_id: str, *, expect: EntryIdentity | None = None
) -> _HeldLeaseLock:
    locks: AnchoredDirectory | None = None
    try:
        with AnchoredDirectory(database.parent) as root:
            locks = open_child_directory(root, LOCKS_LEAF, create=True)
        manager = hold(locks, _lock_name(lease_id), expect=expect)
        _descriptor, identity = manager.__enter__()
        return _HeldLeaseLock(locks=locks, manager=manager, identity=identity)
    except (AnchoredDirectoryError, SharedAssetLockError, OSError):
        if locks is not None:
            locks.close()
        _invalid()


def _owned_final_claim(
    connection: sqlite3.Connection, consumer_id: str, package_digest: str
) -> bool:
    row = connection.execute(
        "SELECT 1 FROM package_claims WHERE consumer_id = ? AND package_digest = ? AND state = ?",
        (consumer_id, package_digest, FINAL),
    ).fetchone()
    return row is not None


def _lease_from_row(consumer_id: str, row: tuple[object, ...]) -> SharedAssetReadLease:
    try:
        return SharedAssetReadLease(
            lease_id=cast(str, row[0]),
            consumer_id=consumer_id,
            package_digest=cast(str, row[1]),
            holder_pid=cast(int, row[2]),
            expires_at=cast(int, row[3]),
        )
    except (IndexError, TypeError, ValueError):
        _invalid()


def acquire_read_lease(
    *,
    database: Path,
    consumer_id: str,
    package_digest: str,
    now: int | None = None,
    ttl: int = DEFAULT_TTL_SECONDS,
) -> SharedAssetReadLease:
    """Acquire or reacquire this consumer's lease on a finalized claim."""

    consumer = _require_consumer(consumer_id)
    digest = _require_digest(package_digest)
    _moment, expires_at = _clock(now, ttl)
    holder_pid, holder_executable = current_process_identity()
    with _HELD_GUARD:
        acquired: _HeldLeaseLock | None = None
        lease_id = ""
        try:
            with _registry(database, require_leases=True) as connection, connection:
                if not _owned_final_claim(connection, consumer, digest):
                    _invalid()
                row = connection.execute(
                    "SELECT lease_id, lock_device, lock_inode, lock_token"
                    " FROM package_leases"
                    " WHERE consumer_id = ? AND package_digest = ?",
                    (consumer, digest),
                ).fetchone()
                if row is None:
                    lease_id = secrets.token_hex(16)
                    acquired = _take_lock(database, lease_id)
                    identity = acquired.identity
                    connection.execute(
                        "INSERT INTO package_leases"
                        " (lease_id, consumer_id, package_digest, lock_device,"
                        " lock_inode, lock_token, holder_pid, holder_executable, expires_at)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            lease_id,
                            consumer,
                            digest,
                            format(identity[0], "x"),
                            format(identity[1], "x"),
                            sqlite3.Binary(identity[2]),
                            holder_pid,
                            holder_executable,
                            expires_at,
                        ),
                    )
                else:
                    lease_id = _require_lease_id(str(row[0]))
                    expected = _row_identity(row, 1)
                    existing = _HELD.get(_key(database, lease_id))
                    if existing is not None:
                        if existing.identity != expected:
                            _invalid()
                        acquired = existing
                    else:
                        acquired = _take_lock(database, lease_id, expect=expected)
                    connection.execute(
                        "UPDATE package_leases"
                        " SET holder_pid = ?, holder_executable = ?, expires_at = ?"
                        " WHERE lease_id = ? AND consumer_id = ?",
                        (holder_pid, holder_executable, expires_at, lease_id, consumer),
                    )
                result = SharedAssetReadLease(
                    lease_id=lease_id,
                    consumer_id=consumer,
                    package_digest=digest,
                    holder_pid=holder_pid,
                    expires_at=expires_at,
                )
            assert acquired is not None
            _HELD[_key(database, lease_id)] = acquired
            return result
        except (sqlite3.Error, SharedAssetRegistryError, SharedAssetLeaseError):
            if acquired is not None and _HELD.get(_key(database, lease_id)) is not acquired:
                acquired.close()
            _invalid()


def renew_read_lease(
    *,
    database: Path,
    consumer_id: str,
    lease_id: str,
    now: int | None = None,
    ttl: int = DEFAULT_TTL_SECONDS,
) -> SharedAssetReadLease:
    """Extend a lease held by this process and return its new expiry."""

    consumer = _require_consumer(consumer_id)
    chosen = _require_lease_id(lease_id)
    _moment, expires_at = _clock(now, ttl)
    holder_pid, holder_executable = current_process_identity()
    with _HELD_GUARD:
        try:
            with _registry(database, require_leases=True) as connection, connection:
                row = connection.execute(
                    "SELECT package_digest, holder_pid, lock_device, lock_inode, lock_token"
                    " FROM package_leases WHERE lease_id = ? AND consumer_id = ?",
                    (chosen, consumer),
                ).fetchone()
                held = _HELD.get(_key(database, chosen))
                if row is None or held is None or held.identity != _row_identity(row, 2):
                    _invalid()
                updated = connection.execute(
                    "UPDATE package_leases"
                    " SET holder_pid = ?, holder_executable = ?, expires_at = ?"
                    " WHERE lease_id = ? AND consumer_id = ?",
                    (holder_pid, holder_executable, expires_at, chosen, consumer),
                )
                if updated.rowcount != 1:
                    _invalid()
                return SharedAssetReadLease(
                    lease_id=chosen,
                    consumer_id=consumer,
                    package_digest=str(row[0]),
                    holder_pid=holder_pid,
                    expires_at=expires_at,
                )
        except (sqlite3.Error, SharedAssetRegistryError, SharedAssetLeaseError):
            _invalid()


def release_read_lease(*, database: Path, consumer_id: str, lease_id: str) -> None:
    """Release only the caller's locally held lease; package bytes stay."""

    consumer = _require_consumer(consumer_id)
    chosen = _require_lease_id(lease_id)
    key = _key(database, chosen)
    with _HELD_GUARD:
        held = _HELD.get(key)
        try:
            with _registry(database, require_leases=True) as connection, connection:
                row = connection.execute(
                    "SELECT lock_device, lock_inode, lock_token FROM package_leases"
                    " WHERE lease_id = ? AND consumer_id = ?",
                    (chosen, consumer),
                ).fetchone()
                if row is None or held is None or held.identity != _row_identity(row, 0):
                    _invalid()
                deleted = connection.execute(
                    "DELETE FROM package_leases WHERE lease_id = ? AND consumer_id = ?",
                    (chosen, consumer),
                )
                if deleted.rowcount != 1:
                    _invalid()
        except (sqlite3.Error, SharedAssetRegistryError, SharedAssetLeaseError):
            _invalid()
        _HELD.pop(key, None)
        held.close()


def steal_expired_read_lease(
    *,
    database: Path,
    consumer_id: str,
    package_digest: str,
    now: int | None = None,
    ttl: int = DEFAULT_TTL_SECONDS,
) -> SharedAssetReadLease:
    """Reassign one expired row only while holding its exact lock object."""

    consumer = _require_consumer(consumer_id)
    digest = _require_digest(package_digest)
    moment, expires_at = _clock(now, ttl)
    holder_pid, holder_executable = current_process_identity()
    with _HELD_GUARD:
        acquired: _HeldLeaseLock | None = None
        lease_id = ""
        try:
            with _registry(database, require_leases=True) as connection, connection:
                if not _owned_final_claim(connection, consumer, digest):
                    _invalid()
                mine = connection.execute(
                    "SELECT 1 FROM package_leases WHERE consumer_id = ? AND package_digest = ?",
                    (consumer, digest),
                ).fetchone()
                if mine is not None:
                    _invalid()
                row = connection.execute(
                    "SELECT lease_id, lock_device, lock_inode, lock_token"
                    " FROM package_leases"
                    " WHERE package_digest = ? AND consumer_id != ? AND expires_at <= ?"
                    " ORDER BY lease_id LIMIT 1",
                    (digest, consumer, moment),
                ).fetchone()
                if row is None:
                    _invalid()
                lease_id = _require_lease_id(str(row[0]))
                expected = _row_identity(row, 1)
                acquired = _take_lock(database, lease_id, expect=expected)
                updated = connection.execute(
                    "UPDATE package_leases"
                    " SET consumer_id = ?, holder_pid = ?, holder_executable = ?,"
                    " expires_at = ?"
                    " WHERE lease_id = ? AND package_digest = ? AND expires_at <= ?",
                    (
                        consumer,
                        holder_pid,
                        holder_executable,
                        expires_at,
                        lease_id,
                        digest,
                        moment,
                    ),
                )
                if updated.rowcount != 1:
                    _invalid()
                result = SharedAssetReadLease(
                    lease_id=lease_id,
                    consumer_id=consumer,
                    package_digest=digest,
                    holder_pid=holder_pid,
                    expires_at=expires_at,
                )
            assert acquired is not None
            _HELD[_key(database, lease_id)] = acquired
            return result
        except (sqlite3.Error, SharedAssetRegistryError, SharedAssetLeaseError):
            if acquired is not None:
                acquired.close()
            _invalid()


def leases_for_consumer(*, database: Path, consumer_id: str) -> tuple[SharedAssetReadLease, ...]:
    """Return exactly one consumer's leases; no cross-consumer enumeration."""

    consumer = _require_consumer(consumer_id)
    try:
        with _registry(database, require_leases=True) as connection:
            rows = connection.execute(
                "SELECT lease_id, package_digest, holder_pid, expires_at"
                " FROM package_leases WHERE consumer_id = ? ORDER BY package_digest",
                (consumer,),
            ).fetchall()
            return tuple(_lease_from_row(consumer, row) for row in rows)
    except (sqlite3.Error, SharedAssetRegistryError):
        _invalid()


def package_has_read_lease(*, database: Path, package_digest: str) -> bool:
    """Expose the one cross-consumer bit future collection needs."""

    digest = _require_digest(package_digest)
    try:
        with _registry(database, require_leases=True) as connection:
            row = connection.execute(
                "SELECT 1 FROM package_leases WHERE package_digest = ? LIMIT 1",
                (digest,),
            ).fetchone()
            return row is not None
    except (sqlite3.Error, SharedAssetRegistryError):
        _invalid()


def _release_all_for_testing() -> None:
    """Release process-local descriptors; tests only, never registry rows."""

    with _HELD_GUARD:
        held = list(_HELD.values())
        _HELD.clear()
        for item in held:
            with contextlib.suppress(Exception):
                item.close()
