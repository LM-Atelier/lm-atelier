"""Fail-closed collection of one unreferenced shared-asset object.

Callers pass one explicit store root, its exact registry database, and one
digest. Collection holds the registry write gate while it proves that no
claim and no read lease protects the digest. Object access stays relative to
held directory identities; no profile identity or cross-consumer row leaves
the module.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
import sqlite3
import stat
from pathlib import Path
from typing import Final, NoReturn

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryKind,
    list_entries,
    open_child_directory,
    open_entry,
    remove_entry,
    rename_entry,
    sync_directory,
)
from .shared_asset_contract_v1 import SharedAssetContractError, _require_absolute_root
from .shared_asset_registry_v1 import (
    REGISTRY_LEAF,
    SharedAssetRegistryError,
    _registry,
    _require_digest,
)

INVALID_COLLECTION: Final = "shared asset collection is invalid"
_CHUNK: Final = 1024 * 1024


class SharedAssetCollectionError(ValueError):
    """Fixed non-echoing refusal for an unsafe collection attempt."""


def _invalid() -> NoReturn:
    raise SharedAssetCollectionError(INVALID_COLLECTION) from None


def _digest_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, _CHUNK):
            digest.update(chunk)
    except OSError:
        _invalid()
    return digest.hexdigest()


def _entries(anchor: AnchoredDirectory) -> dict[str, AnchoredEntryKind]:
    return {entry.name: entry.kind for entry in list_entries(anchor)}


def _restore_staged(anchor: AnchoredDirectory, *, staged: str, destination: str) -> None:
    """Best-effort restoration that never replaces a newer destination."""

    with contextlib.suppress(AnchoredDirectoryError, OSError):
        rename_entry(anchor, staged, destination, replace=False)
        sync_directory(anchor)


def _collect_anchored(*, root: Path, digest: str) -> None:
    staged = f"collect-{secrets.token_hex(16)}"
    renamed = False
    removed = False
    first: AnchoredDirectory | None = None
    second: AnchoredDirectory | None = None
    descriptor: int | None = None
    moved_descriptor: int | None = None
    try:
        with AnchoredDirectory(root) as store:
            first = open_child_directory(store, digest[:2])
            second = open_child_directory(first, digest[2:4])
            if _entries(second).get(digest) is not AnchoredEntryKind.FILE:
                _invalid()
            descriptor = open_entry(second, digest)
            if descriptor is None:
                _invalid()
            measured = os.fstat(descriptor)
            if not stat.S_ISREG(measured.st_mode) or _digest_descriptor(descriptor) != digest:
                _invalid()

            # Windows cannot rename an entry while this read descriptor is
            # open. Keep its stable device/file identity as evidence, close
            # it, then require the staged entry to reopen as that same object.
            os.close(descriptor)
            descriptor = None

            rename_entry(second, digest, staged, replace=False)
            renamed = True
            if _entries(second).get(staged) is not AnchoredEntryKind.FILE:
                _invalid()
            moved_descriptor = open_entry(second, staged)
            if moved_descriptor is None:
                _invalid()
            moved = os.fstat(moved_descriptor)
            if (moved.st_dev, moved.st_ino) != (measured.st_dev, measured.st_ino):
                _invalid()

            os.close(moved_descriptor)
            moved_descriptor = None
            remove_entry(second, staged)
            removed = True
            sync_directory(second)
    except (AnchoredDirectoryError, OSError, SharedAssetCollectionError):
        if moved_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(moved_descriptor)
            moved_descriptor = None
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
            descriptor = None
        if renamed and not removed and second is not None:
            _restore_staged(second, staged=staged, destination=digest)
        _invalid()
    finally:
        if moved_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(moved_descriptor)
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if second is not None:
            second.close()
        if first is not None:
            first.close()


def collect_unreferenced_object(*, root: Path, database: Path, package_digest: str) -> None:
    """Remove one verified object only while no claim or lease protects it."""

    try:
        store = _require_absolute_root(root)
        digest = _require_digest(package_digest)
    except (SharedAssetContractError, SharedAssetRegistryError):
        _invalid()
    if not isinstance(database, Path) or database != store / REGISTRY_LEAF:
        _invalid()

    connection: sqlite3.Connection | None = None
    try:
        with _registry(database, require_membership=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            claimed = connection.execute(
                "SELECT 1 FROM package_claims WHERE package_digest = ? LIMIT 1",
                (digest,),
            ).fetchone()
            leased = connection.execute(
                "SELECT 1 FROM package_leases WHERE package_digest = ? LIMIT 1",
                (digest,),
            ).fetchone()
            if claimed is not None or leased is not None:
                _invalid()
            # A pre-index claim might name a package rather than a raw object.
            # Never interpret missing membership as proof of no dependencies.
            # Idempotent verified package publication fills its exact edges.
            unknown = connection.execute(
                "SELECT 1 FROM (SELECT package_digest FROM package_claims"
                " UNION SELECT package_digest FROM package_leases) AS protected"
                " WHERE NOT EXISTS (SELECT 1 FROM package_members AS members"
                " WHERE members.package_digest = protected.package_digest) LIMIT 1"
            ).fetchone()
            if unknown is not None:
                _invalid()
            protected_member = connection.execute(
                "WITH RECURSIVE parents(digest) AS ("
                " SELECT package_digest FROM package_members WHERE member_digest = ?"
                " UNION SELECT members.package_digest FROM package_members AS members"
                " JOIN parents ON members.member_digest = parents.digest)"
                " SELECT 1 FROM parents WHERE"
                " EXISTS (SELECT 1 FROM package_claims WHERE package_digest = parents.digest)"
                " OR EXISTS (SELECT 1 FROM package_leases WHERE package_digest = parents.digest)"
                " LIMIT 1",
                (digest,),
            ).fetchone()
            if protected_member is not None:
                _invalid()
            _collect_anchored(root=store, digest=digest)
            connection.execute("DELETE FROM package_members WHERE package_digest = ?", (digest,))
            connection.execute("COMMIT")
    except (
        OSError,
        sqlite3.Error,
        SharedAssetCollectionError,
        SharedAssetRegistryError,
    ):
        if connection is not None:
            with contextlib.suppress(sqlite3.Error):
                connection.execute("ROLLBACK")
        _invalid()
