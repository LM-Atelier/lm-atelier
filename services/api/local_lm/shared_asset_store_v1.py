"""Content-addressed object storage for the Shared Asset Library (item 58).

Callers pass an explicit root. This module never discovers the desktop
library. Bytes are staged on the destination filesystem, hashed, and
published by sha256. No profile claims, API mount, migrate, or Settings.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Final, NoReturn

SCHEMA_ID: Final = "lm-atelier-shared-asset-object-v1"
SCHEMA_VERSION: Final = 1
INVALID_OBJECT: Final = "shared asset object is invalid"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1024 * 1024


class SharedAssetStoreError(ValueError):
    """Fixed non-echoing refusal for an unusable store operation."""


def _invalid() -> NoReturn:
    raise SharedAssetStoreError(INVALID_OBJECT)


def _is_unc(path: Path) -> bool:
    text = str(path)
    return text.startswith("\\\\") or text.startswith("//") or path.as_posix().startswith("//")


def _is_link(path: Path) -> bool:
    try:
        return path.is_symlink() or os.path.islink(path)
    except OSError:
        _invalid()


def _require_absolute_dir_root(root: Path) -> Path:
    if not isinstance(root, Path):
        _invalid()
    try:
        chosen = root.expanduser()
    except (OSError, RuntimeError, ValueError):
        _invalid()
    if not str(chosen) or _is_unc(chosen) or not chosen.is_absolute():
        _invalid()
    if chosen.exists() and (not chosen.is_dir() or _is_link(chosen)):
        _invalid()
    return chosen


def object_path(*, root: Path, digest: str) -> Path:
    store = _require_absolute_dir_root(root)
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        _invalid()
    return store / digest[:2] / digest[2:4] / digest


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(_CHUNK):
                digest.update(chunk)
    except OSError:
        _invalid()
    return digest.hexdigest()


def _refuse_links_to_root(root: Path, path: Path) -> None:
    cursor = path
    while True:
        if _is_link(cursor):
            _invalid()
        if cursor == root:
            return
        if root not in cursor.parents:
            _invalid()
        cursor = cursor.parent


def publish_file(*, root: Path, source: Path) -> str:
    """Publish one file into the store. Return the lowercase hex digest."""

    store = _require_absolute_dir_root(root)
    if not isinstance(source, Path):
        _invalid()
    try:
        source_path = source.expanduser()
    except (OSError, RuntimeError, ValueError):
        _invalid()
    if not str(source_path) or _is_unc(source_path) or not source_path.is_absolute():
        _invalid()
    try:
        source_path = source_path.resolve(strict=True)
    except OSError:
        _invalid()
    if not source_path.is_file() or _is_link(source_path):
        _invalid()

    digest = _digest_file(source_path)
    destination = object_path(root=store, digest=digest)
    _refuse_links_to_root(store, destination.parent)
    if destination.exists():
        if not destination.is_file() or _is_link(destination):
            _invalid()
        if _digest_file(destination) != digest:
            _invalid()
        return digest

    store.mkdir(parents=True, exist_ok=True)
    staging_dir = store / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="publish-", dir=str(staging_dir))
    try:
        with os.fdopen(fd, "wb") as handle, source_path.open("rb") as src:
            while chunk := src.read(_CHUNK):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        staged = Path(temporary_name)
        if _digest_file(staged) != digest:
            _invalid()
        destination.parent.mkdir(parents=True, exist_ok=True)
        _refuse_links_to_root(store, destination.parent)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    if _digest_file(destination) != digest:
        _invalid()
    return digest
