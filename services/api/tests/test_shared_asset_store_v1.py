from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from local_lm.shared_asset_root_v1 import default_shared_asset_root
from local_lm.shared_asset_store_v1 import (
    INVALID_OBJECT,
    SharedAssetStoreError,
    object_path,
    publish_file,
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_publish_file_is_idempotent_and_content_addressed(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    source = tmp_path / "model.bin"
    payload = b"verified-bytes"
    source.write_bytes(payload)
    first = publish_file(root=root, source=source)
    second = publish_file(root=root, source=source)
    expected = _digest(payload)
    assert first == second == expected
    stored = object_path(root=root, digest=expected)
    assert stored.is_file()
    assert stored.read_bytes() == payload
    assert stored.parent.parent.parent == root


def test_publish_file_does_not_write_the_desktop_library(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    root = tmp_path / "packages"
    source = tmp_path / "lora.bin"
    source.write_bytes(b"keep-out-of-desktop")
    digest = publish_file(root=root, source=source)
    desktop = default_shared_asset_root()
    assert object_path(root=root, digest=digest).is_file()
    assert not (desktop / digest[:2] / digest[2:4] / digest).exists()


def test_publish_file_refuses_existing_drift(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    payload = b"canonical"
    digest = _digest(payload)
    destination = object_path(root=root, digest=digest)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"corrupted")
    source = tmp_path / "ok.bin"
    source.write_bytes(payload)
    with pytest.raises(SharedAssetStoreError, match=INVALID_OBJECT):
        publish_file(root=root, source=source)


def test_publish_file_refuses_unc_relative_and_missing_sources(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    source = tmp_path / "ok.bin"
    source.write_bytes(b"ok")
    with pytest.raises(SharedAssetStoreError, match=INVALID_OBJECT):
        publish_file(root=Path(r"\\server\share\packages"), source=source)
    with pytest.raises(SharedAssetStoreError, match=INVALID_OBJECT):
        publish_file(root=Path("relative-packages"), source=source)
    with pytest.raises(SharedAssetStoreError, match=INVALID_OBJECT):
        publish_file(root=root, source=Path("relative.bin"))
    with pytest.raises(SharedAssetStoreError, match=INVALID_OBJECT):
        publish_file(root=root, source=tmp_path / "missing.bin")
    with pytest.raises(SharedAssetStoreError, match=INVALID_OBJECT):
        publish_file(root=root, source=tmp_path)
