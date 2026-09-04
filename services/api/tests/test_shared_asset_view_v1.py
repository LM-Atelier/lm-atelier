from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

import pytest

import local_lm.shared_asset_view_v1 as view
from local_lm.filesystem_links import AnchoredDirectory, open_entry
from local_lm.shared_asset_package_v1 import publish_package
from local_lm.shared_asset_root_v1 import default_shared_asset_root
from local_lm.shared_asset_store_v1 import object_path, publish_file
from local_lm.shared_asset_view_v1 import (
    INVALID_VIEW,
    MAX_VIEW_BYTES,
    SharedAssetViewError,
    close_package_view,
    open_package_view,
    view_member_path,
)


def _package(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "packages"
    source = tmp_path / "model.bin"
    source.write_bytes(b"view-bytes")
    weights = publish_file(root=root, source=source)
    digest = publish_package(root=root, members={"unet": weights})
    return root, digest, weights


def _record(root: Path, view_id: str) -> Path:
    return root / ".views" / f"{view_id}.json"


def test_two_views_resolve_one_object_and_close_keeps_every_object(tmp_path: Path) -> None:
    root, package, weights = _package(tmp_path)
    first = open_package_view(root=root, digest=package)
    second = open_package_view(root=root, digest=package)

    assert first != second
    stored = object_path(root=root, digest=weights)
    assert view_member_path(root=root, view_id=first, role="unet") == stored
    assert view_member_path(root=root, view_id=second, role="unet") == stored
    assert "unet" not in stored.parts

    record = json.loads(_record(root, first).read_text(encoding="ascii"))
    assert record == {
        "package": package,
        "schema": "lm-atelier-shared-asset-view-v1",
        "version": 1,
    }
    assert "unet" not in _record(root, first).read_text(encoding="ascii")
    entries = tuple(root.rglob("*"))
    assert {entry.relative_to(root) for entry in entries if entry.is_file()} == {
        object_path(root=root, digest=package).relative_to(root),
        stored.relative_to(root),
        Path(".views", f"{first}.json"),
        Path(".views", f"{second}.json"),
    }
    assert all(not entry.is_symlink() for entry in entries)

    close_package_view(root=root, view_id=first)
    assert not _record(root, first).exists()
    assert object_path(root=root, digest=package).is_file()
    assert stored.read_bytes() == b"view-bytes"
    assert view_member_path(root=root, view_id=second, role="unet") == stored
    with pytest.raises(SharedAssetViewError, match=INVALID_VIEW):
        view_member_path(root=root, view_id=first, role="unet")


def test_view_never_discovers_or_writes_the_desktop_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    root, package, weights = _package(tmp_path)

    view_id = open_package_view(root=root, digest=package)

    assert view_member_path(root=root, view_id=view_id, role="unet") == object_path(
        root=root, digest=weights
    )
    assert not (default_shared_asset_root() / ".views").exists()


def test_lookup_is_read_only_and_refuses_unknown_ids_and_roles(tmp_path: Path) -> None:
    root, package, _weights = _package(tmp_path)
    with pytest.raises(SharedAssetViewError, match=INVALID_VIEW):
        view_member_path(root=root, view_id="c" * 32, role="unet")
    assert not (root / ".views").exists()

    view_id = open_package_view(root=root, digest=package)
    for role in ("vae", "../unet", "unet/path", ""):
        with pytest.raises(SharedAssetViewError, match=INVALID_VIEW):
            view_member_path(root=root, view_id=view_id, role=role)
    with pytest.raises(SharedAssetViewError, match=INVALID_VIEW):
        open_package_view(root=root, digest="ab" * 32)
    with pytest.raises(SharedAssetViewError, match=INVALID_VIEW):
        close_package_view(root=root, view_id="d" * 32)


def test_view_creation_is_exclusive_and_never_overwrites_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, package, _weights = _package(tmp_path)
    chosen = "a" * 32
    monkeypatch.setattr(secrets, "token_hex", lambda _size: chosen)
    assert open_package_view(root=root, digest=package) == chosen
    original = _record(root, chosen).read_bytes()

    with pytest.raises(SharedAssetViewError, match=INVALID_VIEW):
        open_package_view(root=root, digest=package)

    assert _record(root, chosen).read_bytes() == original


@pytest.mark.parametrize("drifted", ["package", "member"])
def test_lookup_revalidates_the_package_and_member_bytes(tmp_path: Path, drifted: str) -> None:
    root, package, weights = _package(tmp_path)
    view_id = open_package_view(root=root, digest=package)
    changed = package if drifted == "package" else weights
    stored = object_path(root=root, digest=changed)
    stored.write_bytes(stored.read_bytes() + b" ")

    with pytest.raises(SharedAssetViewError, match=INVALID_VIEW):
        view_member_path(root=root, view_id=view_id, role="unet")


def test_record_is_bounded_before_json_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, package, _weights = _package(tmp_path)
    view_id = open_package_view(root=root, digest=package)
    _record(root, view_id).write_bytes(b"{" + b" " * MAX_VIEW_BYTES + b"}")

    def decode_was_reached(_raw: object) -> object:
        raise AssertionError("oversized view reached JSON decoding")

    monkeypatch.setattr(json, "loads", decode_was_reached)
    with pytest.raises(SharedAssetViewError, match=INVALID_VIEW):
        view_member_path(root=root, view_id=view_id, role="unet")


def test_close_uses_the_held_views_directory_not_its_path_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, package, weights = _package(tmp_path)
    view_id = open_package_view(root=root, digest=package)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_record = outside / f"{view_id}.json"
    outside_record.write_bytes(b"outside metadata")
    real_open = open_entry

    def relabel_after_open(anchor: AnchoredDirectory, name: str) -> int | None:
        descriptor = real_open(anchor, name)
        anchor.path = outside
        return descriptor

    monkeypatch.setattr(view, "open_entry", relabel_after_open)
    close_package_view(root=root, view_id=view_id)

    assert outside_record.read_bytes() == b"outside metadata"
    assert not _record(root, view_id).exists()
    assert object_path(root=root, digest=weights).read_bytes() == b"view-bytes"


def test_linked_views_directory_is_refused_without_following_it(tmp_path: Path) -> None:
    root, package, _weights = _package(tmp_path)
    outside = tmp_path / "outside-views"
    outside.mkdir()
    try:
        os.symlink(outside, root / ".views", target_is_directory=True)
    except OSError:
        pytest.skip("this host does not allow directory links")

    with pytest.raises(SharedAssetViewError, match=INVALID_VIEW):
        open_package_view(root=root, digest=package)
    assert list(outside.iterdir()) == []


def test_close_can_remove_corrupt_metadata_without_touching_objects(tmp_path: Path) -> None:
    root, package, weights = _package(tmp_path)
    view_id = open_package_view(root=root, digest=package)
    _record(root, view_id).write_bytes(b"not json")

    close_package_view(root=root, view_id=view_id)

    assert not _record(root, view_id).exists()
    assert object_path(root=root, digest=package).is_file()
    assert object_path(root=root, digest=weights).read_bytes() == b"view-bytes"


def test_all_public_refusals_are_fixed_and_non_echoing(tmp_path: Path) -> None:
    root, _package_digest, _weights = _package(tmp_path)
    cases = (
        lambda: open_package_view(root=Path("relative"), digest="f" * 64),
        lambda: view_member_path(root=root, view_id="../../outside", role="unet"),
        lambda: close_package_view(root=root, view_id="../../outside"),
    )

    for invoke in cases:
        with pytest.raises(SharedAssetViewError) as caught:
            invoke()
        assert type(caught.value) is SharedAssetViewError
        assert str(caught.value) == INVALID_VIEW
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
