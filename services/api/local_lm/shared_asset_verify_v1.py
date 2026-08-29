"""Fail-closed integrity checks for Shared Asset Library objects.

Callers pass an explicit root and digest. The object and both digest shards are
opened through held directory identities, and one descriptor is hashed and
counted. This module never discovers the desktop library and never writes.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from pathlib import Path
from typing import Final, NoReturn

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    open_child_directory,
    open_entry,
)
from .shared_asset_contract_v1 import SharedAssetContractError, _require_absolute_root

SCHEMA_ID: Final = "lm-atelier-shared-asset-verify-v1"
SCHEMA_VERSION: Final = 1
INVALID_VERIFY: Final = "shared asset object failed verification"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK: Final = 1024 * 1024


class SharedAssetVerifyError(ValueError):
    """Fixed non-echoing refusal for a failed integrity check."""


def _invalid() -> NoReturn:
    raise SharedAssetVerifyError(INVALID_VERIFY) from None


def _require_digest(value: object) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        _invalid()
    return value


def _digest_and_size(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, _CHUNK):
            digest.update(chunk)
            size += len(chunk)
    except OSError:
        _invalid()
    return digest.hexdigest(), size


def _verify_published_object(*, root: Path, digest: str) -> int:
    try:
        with (
            AnchoredDirectory(root) as store,
            open_child_directory(store, digest[:2]) as first,
            open_child_directory(first, digest[2:4]) as second,
        ):
            descriptor = open_entry(second, digest)
            if descriptor is None:
                _invalid()
            try:
                hashed, size = _digest_and_size(descriptor)
                if hashed != digest:
                    _invalid()
                return size
            finally:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
    except (AnchoredDirectoryError, OSError, SharedAssetVerifyError):
        _invalid()


def verify_published_object(*, root: Path, digest: str) -> int:
    """Re-hash one published object and return its verified size in bytes."""

    try:
        chosen_root = _require_absolute_root(root)
        chosen_digest = _require_digest(digest)
        return _verify_published_object(root=chosen_root, digest=chosen_digest)
    except (
        AnchoredDirectoryError,
        OSError,
        SharedAssetContractError,
        SharedAssetVerifyError,
    ):
        pass
    _invalid()
