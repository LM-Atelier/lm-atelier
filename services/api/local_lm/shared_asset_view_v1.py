"""Read-only role views over immutable Shared Asset Library packages.

A view stores only an opaque id and one package digest. Role lookup reopens and
revalidates that immutable package through the same held store root, then
returns its digest-addressed object path. Views never copy or link objects, and
roles never become path components. Callers pass an explicit root; there is no
desktop discovery, API, Settings, or migration behavior here.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
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
    sync_directory,
)
from .shared_asset_contract_v1 import SharedAssetContractError, _require_absolute_root
from .shared_asset_package_v1 import (
    PACKAGE_ROLES,
    SharedAssetPackageError,
    _load_package_from_store,
)

SCHEMA_ID: Final = "lm-atelier-shared-asset-view-v1"
SCHEMA_VERSION: Final = 1
INVALID_VIEW: Final = "shared asset view is invalid"
VIEWS_LEAF: Final = ".views"
MAX_VIEW_BYTES: Final = 4096
_TOKEN = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK: Final = 4096
_CREATE_ATTEMPTS: Final = 32


class SharedAssetViewError(ValueError):
    """Fixed non-echoing refusal for an unusable view operation."""


def _invalid() -> NoReturn:
    raise SharedAssetViewError(INVALID_VIEW) from None


def _require_token(value: object) -> str:
    if type(value) is not str or not _TOKEN.fullmatch(value):
        _invalid()
    return value


def _require_digest(value: object) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        _invalid()
    return value


def _require_role(value: object) -> str:
    if type(value) is not str or value not in PACKAGE_ROLES:
        _invalid()
    return value


def _record_name(view_id: str) -> str:
    return f"{view_id}.json"


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


def _read_bounded(descriptor: int) -> bytes:
    try:
        measured = os.fstat(descriptor)
        if measured.st_size < 0 or measured.st_size > MAX_VIEW_BYTES:
            _invalid()
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = bytearray()
        while chunk := os.read(descriptor, min(_CHUNK, MAX_VIEW_BYTES + 1 - len(raw))):
            raw.extend(chunk)
            if len(raw) > MAX_VIEW_BYTES:
                _invalid()
        return bytes(raw)
    except OSError:
        _invalid()


def _parse_record(raw: bytes) -> str:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _invalid()
    if type(value) is not dict or set(value) != {"package", "schema", "version"}:
        _invalid()
    if value.get("schema") != SCHEMA_ID:
        _invalid()
    if type(value.get("version")) is not int or value["version"] != SCHEMA_VERSION:
        _invalid()
    return _require_digest(value.get("package"))


def _read_record(views: AnchoredDirectory, view_id: str) -> str:
    descriptor = open_entry(views, _record_name(view_id))
    if descriptor is None:
        _invalid()
    try:
        return _parse_record(_read_bounded(descriptor))
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def _open_package_view(*, root: Path, digest: str) -> str:
    chosen_root = _require_absolute_root(root)
    package = _require_digest(digest)
    with AnchoredDirectory(chosen_root) as store:
        _load_package_from_store(store=store, digest=package)
        with open_child_directory(store, VIEWS_LEAF, create=True) as views:
            payload = json.dumps(
                {"package": package, "schema": SCHEMA_ID, "version": SCHEMA_VERSION},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            if len(payload) > MAX_VIEW_BYTES:
                _invalid()
            for _attempt in range(_CREATE_ATTEMPTS):
                view_id = secrets.token_hex(16)
                name = _record_name(view_id)
                try:
                    descriptor = create_entry(views, name)
                except AnchoredEntryExists:
                    continue
                published = False
                try:
                    _write_all(descriptor, payload)
                    os.close(descriptor)
                    descriptor = -1
                    sync_directory(views)
                    published = True
                    return view_id
                finally:
                    if descriptor >= 0:
                        with contextlib.suppress(OSError):
                            os.close(descriptor)
                    if not published:
                        discard_entry(views, name)
    _invalid()


def open_package_view(*, root: Path, digest: str) -> str:
    """Open one package view and return its opaque 32-hex id."""

    try:
        return _open_package_view(root=root, digest=digest)
    except (
        AnchoredDirectoryError,
        OSError,
        SharedAssetContractError,
        SharedAssetPackageError,
        SharedAssetViewError,
    ):
        pass
    _invalid()


def _view_member_path(*, root: Path, view_id: str, role: str) -> Path:
    chosen_root = _require_absolute_root(root)
    chosen_id = _require_token(view_id)
    chosen_role = _require_role(role)
    with AnchoredDirectory(chosen_root) as store:
        with open_child_directory(store, VIEWS_LEAF) as views:
            package = _read_record(views, chosen_id)
        members = dict(_load_package_from_store(store=store, digest=package))
        member = members.get(chosen_role)
        if member is None:
            _invalid()
        return chosen_root / member[:2] / member[2:4] / member


def view_member_path(*, root: Path, view_id: str, role: str) -> Path:
    """Return the digest object path authorized for ``role`` by this view."""

    try:
        return _view_member_path(root=root, view_id=view_id, role=role)
    except (
        AnchoredDirectoryError,
        OSError,
        SharedAssetContractError,
        SharedAssetPackageError,
        SharedAssetViewError,
    ):
        pass
    _invalid()


def _close_package_view(*, root: Path, view_id: str) -> None:
    chosen_root = _require_absolute_root(root)
    chosen_id = _require_token(view_id)
    with (
        AnchoredDirectory(chosen_root) as store,
        open_child_directory(store, VIEWS_LEAF) as views,
    ):
        descriptor = open_entry(views, _record_name(chosen_id))
        if descriptor is None:
            _invalid()
        with contextlib.suppress(OSError):
            os.close(descriptor)
        remove_entry(views, _record_name(chosen_id))
        sync_directory(views)


def close_package_view(*, root: Path, view_id: str) -> None:
    """Remove only this view's metadata; package and member bytes remain."""

    try:
        _close_package_view(root=root, view_id=view_id)
        return
    except (
        AnchoredDirectoryError,
        OSError,
        SharedAssetContractError,
        SharedAssetViewError,
    ):
        pass
    _invalid()
