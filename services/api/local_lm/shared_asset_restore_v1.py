"""Restore one verified staged object to an empty content-addressed slot.

Collection can leave an opaque ``collect-<token>`` entry beside the canonical
digest name when its best-effort rollback cannot finish.  Restoration accepts
one explicit store root and digest, keeps both shard directories held, and
moves exactly one matching staged file without replacing any entry.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn

from . import filesystem_links as _filesystem_links
from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryExists,
    AnchoredEntryKind,
    list_entries,
    open_child_directory,
    open_entry,
    remove_entry,
    sync_directory,
)
from .shared_asset_contract_v1 import SharedAssetContractError, _require_absolute_root
from .shared_asset_registry_v1 import SharedAssetRegistryError, _require_digest

_rename_entry = _filesystem_links.rename_entry

INVALID_RESTORE: Final = "shared asset restoration is invalid"
_COLLECT_ENTRY: Final = re.compile(r"^collect-[0-9a-f]{32}$")
_CHUNK: Final = 1024 * 1024


class SharedAssetRestoreError(ValueError):
    """Fixed non-echoing refusal for an unsafe restoration attempt."""


def _invalid() -> NoReturn:
    raise SharedAssetRestoreError(INVALID_RESTORE) from None


@dataclass(frozen=True, slots=True)
class _ObjectState:
    identity: tuple[int, int]
    digest: str


def _digest_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, _CHUNK):
            digest.update(chunk)
    except OSError:
        _invalid()
    return digest.hexdigest()


def _object_state(anchor: AnchoredDirectory, name: str) -> _ObjectState | None:
    descriptor = open_entry(anchor, name)
    if descriptor is None:
        return None
    try:
        measured = os.fstat(descriptor)
        return _ObjectState(
            (measured.st_dev, measured.st_ino),
            _digest_descriptor(descriptor),
        )
    except OSError:
        _invalid()
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def _matching_staged(shard: AnchoredDirectory, digest: str) -> tuple[str, _ObjectState]:
    matches: list[tuple[str, _ObjectState]] = []
    for entry in list_entries(shard):
        if entry.name == digest:
            _invalid()
        if not entry.name.startswith("collect-"):
            continue
        if not _COLLECT_ENTRY.fullmatch(entry.name):
            _invalid()
        if entry.kind is not AnchoredEntryKind.FILE:
            _invalid()
        state = _object_state(shard, entry.name)
        if state is None:
            _invalid()
        if state.digest == digest:
            matches.append((entry.name, state))
    if len(matches) != 1:
        _invalid()
    return matches[0]


def _best_effort_preserve_staged(
    shard: AnchoredDirectory,
    *,
    staged: str,
    destination: str,
    expected: _ObjectState,
) -> None:
    """Undo a refused move without ever replacing or deleting the staged name."""

    try:
        source = _object_state(shard, staged)
        target = _object_state(shard, destination)
        if source is None and target is not None and target.identity == expected.identity:
            _rename_entry(shard, destination, staged, replace=False)
            sync_directory(shard)
        elif source is not None and target is not None and source.identity == target.identity:
            # A POSIX non-replacing move links the destination before removing
            # the source.  If the second step refuses, remove only the new
            # canonical link and retain the staged name that still exists.
            remove_entry(shard, destination)
            sync_directory(shard)
    except (AnchoredDirectoryError, OSError, SharedAssetRestoreError):
        return


def _restore_anchored(*, root: Path, digest: str) -> None:
    first: AnchoredDirectory | None = None
    second: AnchoredDirectory | None = None
    staged = ""
    expected: _ObjectState | None = None
    cleanup_after_refusal = True
    try:
        with AnchoredDirectory(root) as store:
            first = open_child_directory(store, digest[:2])
            second = open_child_directory(first, digest[2:4])
            staged, expected = _matching_staged(second, digest)

            try:
                _rename_entry(second, staged, digest, replace=False)
            except AnchoredEntryExists:
                # The destination existed before this call. Even when it is
                # another name for the staged object, it is not ours to remove.
                cleanup_after_refusal = False
                _invalid()
            except (AnchoredDirectoryError, OSError):
                _best_effort_preserve_staged(
                    second,
                    staged=staged,
                    destination=digest,
                    expected=expected,
                )
                _invalid()

            restored = _object_state(second, digest)
            source = _object_state(second, staged)
            if (
                restored is None
                or source is not None
                or restored.identity != expected.identity
                or restored.digest != digest
            ):
                _best_effort_preserve_staged(
                    second,
                    staged=staged,
                    destination=digest,
                    expected=expected,
                )
                _invalid()
            sync_directory(second)
    except (AnchoredDirectoryError, OSError, SharedAssetRestoreError):
        if cleanup_after_refusal and staged and expected is not None and second is not None:
            _best_effort_preserve_staged(
                second,
                staged=staged,
                destination=digest,
                expected=expected,
            )
        _invalid()
    finally:
        if second is not None:
            second.close()
        if first is not None:
            first.close()


def restore_quarantined_object(*, root: Path, package_digest: str) -> None:
    """Restore one matching staged file only when its canonical slot is empty."""

    try:
        store = _require_absolute_root(root)
        digest = _require_digest(package_digest)
    except (SharedAssetContractError, SharedAssetRegistryError):
        _invalid()
    _restore_anchored(root=store, digest=digest)
