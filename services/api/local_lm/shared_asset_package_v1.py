"""Immutable package membership for Shared Asset Library objects.

A package maps closed runtime roles to already-published object digests and is
itself published as a digest-addressed object. Callers pass an explicit root;
this module performs no discovery, API, Settings, migration, or profile work.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Final, NoReturn

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryExists,
    create_entry,
    discard_entry,
    open_child_directory,
    open_entry,
    remove_entry,
    rename_entry,
    sync_directory,
)
from .shared_asset_contract_v1 import SharedAssetContractError, _require_absolute_root

SCHEMA_ID: Final = "lm-atelier-shared-asset-package-v1"
SCHEMA_VERSION: Final = 1
INVALID_PACKAGE: Final = "shared asset package is invalid"
PACKAGE_ROLES: Final = frozenset(
    {
        "checkpoint",
        "clip_vision",
        "controlnet",
        "diffusion_model",
        "embedding",
        "gguf_model",
        "ip_adapter",
        "lora",
        "text_encoder",
        "unet",
        "upscaler",
        "vae",
    }
)
MAX_PACKAGE_BYTES: Final = 16 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1024 * 1024


class SharedAssetPackageError(ValueError):
    """Fixed non-echoing refusal for an unusable package operation."""


def _invalid() -> NoReturn:
    raise SharedAssetPackageError(INVALID_PACKAGE) from None


def _require_role(value: object) -> str:
    if type(value) is not str or value not in PACKAGE_ROLES:
        _invalid()
    return value


def _require_digest(value: object) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        _invalid()
    return value


def _digest_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, _CHUNK):
            digest.update(chunk)
    except OSError:
        _invalid()
    return digest.hexdigest()


def _read_descriptor(descriptor: int) -> bytes:
    try:
        measured = os.fstat(descriptor)
        if measured.st_size < 0 or measured.st_size > MAX_PACKAGE_BYTES:
            _invalid()
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = bytearray()
        while chunk := os.read(descriptor, min(_CHUNK, MAX_PACKAGE_BYTES + 1 - len(raw))):
            raw.extend(chunk)
            if len(raw) > MAX_PACKAGE_BYTES:
                _invalid()
    except OSError:
        _invalid()
    return bytes(raw)


@contextlib.contextmanager
def _published_descriptor(store: AnchoredDirectory, digest: str) -> Iterator[int]:
    first: AnchoredDirectory | None = None
    second: AnchoredDirectory | None = None
    descriptor: int | None = None
    try:
        first = open_child_directory(store, digest[:2])
        second = open_child_directory(first, digest[2:4])
        descriptor = open_entry(second, digest)
        if descriptor is None:
            _invalid()
        yield descriptor
    except (AnchoredDirectoryError, OSError):
        _invalid()
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if second is not None:
            second.close()
        if first is not None:
            first.close()


def _require_published(store: AnchoredDirectory, digest: str) -> None:
    with _published_descriptor(store, digest) as descriptor:
        if _digest_descriptor(descriptor) != digest:
            _invalid()


def _canonical_members(members: object, store: AnchoredDirectory) -> dict[str, str]:
    if type(members) is not dict or not members or len(members) > len(PACKAGE_ROLES):
        _invalid()
    ordered: dict[str, str] = {}
    for key, digest in members.items():
        role = _require_role(key)
        chosen = _require_digest(digest)
        _require_published(store, chosen)
        ordered[role] = chosen
    return dict(sorted(ordered.items()))


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    try:
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _invalid()
            view = view[written:]
        os.fsync(descriptor)
    except OSError:
        _invalid()


def _exact_payload_exists(directory: AnchoredDirectory, name: str, payload: bytes) -> bool:
    descriptor = open_entry(directory, name)
    if descriptor is None:
        return False
    try:
        if _read_descriptor(descriptor) != payload:
            _invalid()
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)
    return True


def _publish_payload(store: AnchoredDirectory, payload: bytes, digest: str) -> None:
    first: AnchoredDirectory | None = None
    second: AnchoredDirectory | None = None
    descriptor: int | None = None
    staged = f"package-{secrets.token_hex(16)}"
    staged_exists = False
    try:
        first = open_child_directory(store, digest[:2], create=True)
        second = open_child_directory(first, digest[2:4], create=True)
        if _exact_payload_exists(second, digest, payload):
            return

        descriptor = create_entry(second, staged)
        staged_exists = True
        _write_all(descriptor, payload)
        os.close(descriptor)
        descriptor = None
        try:
            rename_entry(second, staged, digest, replace=False)
            staged_exists = False
        except AnchoredEntryExists:
            remove_entry(second, staged)
            staged_exists = False
            if not _exact_payload_exists(second, digest, payload):
                _invalid()
            return
        sync_directory(second)
        if not _exact_payload_exists(second, digest, payload):
            _invalid()
    except (AnchoredDirectoryError, OSError):
        _invalid()
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if staged_exists and second is not None:
            discard_entry(second, staged)
        if second is not None:
            second.close()
        if first is not None:
            first.close()


def _publish_package(*, root: Path, members: dict[str, str]) -> str:
    chosen_root = _require_absolute_root(root)
    with AnchoredDirectory(chosen_root) as store:
        payload = {
            "members": _canonical_members(members, store),
            "schema": SCHEMA_ID,
            "version": SCHEMA_VERSION,
        }
        encoded = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
        if len(encoded) > MAX_PACKAGE_BYTES:
            _invalid()
        digest = hashlib.sha256(encoded).hexdigest()
        _publish_payload(store, encoded, digest)
        return digest


def publish_package(*, root: Path, members: dict[str, str]) -> str:
    """Publish a canonical membership document and return its digest."""

    try:
        return _publish_package(root=root, members=members)
    except (
        AnchoredDirectoryError,
        OSError,
        SharedAssetContractError,
        SharedAssetPackageError,
    ):
        pass
    _invalid()


def _load_package(*, root: Path, digest: str) -> tuple[tuple[str, str], ...]:
    chosen = _require_digest(digest)
    chosen_root = _require_absolute_root(root)
    with AnchoredDirectory(chosen_root) as store:
        with _published_descriptor(store, chosen) as descriptor:
            raw = _read_descriptor(descriptor)
            if hashlib.sha256(raw).hexdigest() != chosen:
                _invalid()
        try:
            value = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            _invalid()
        if type(value) is not dict or set(value) != {"members", "schema", "version"}:
            _invalid()
        if value.get("schema") != SCHEMA_ID or type(value.get("version")) is not int:
            _invalid()
        if value["version"] != SCHEMA_VERSION:
            _invalid()
        loaded = _canonical_members(value.get("members"), store)
        return tuple(loaded.items())


def load_package(*, root: Path, digest: str) -> tuple[tuple[str, str], ...]:
    """Return sorted ``(role, object-digest)`` pairs for one package."""

    try:
        return _load_package(root=root, digest=digest)
    except (
        AnchoredDirectoryError,
        OSError,
        SharedAssetContractError,
        SharedAssetPackageError,
    ):
        pass
    _invalid()
