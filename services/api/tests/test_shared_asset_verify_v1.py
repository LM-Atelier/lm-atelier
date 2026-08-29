from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import local_lm.shared_asset_verify_v1 as verify
from local_lm.filesystem_links import AnchoredDirectory, open_child_directory
from local_lm.shared_asset_root_v1 import default_shared_asset_root
from local_lm.shared_asset_store_v1 import object_path, publish_file
from local_lm.shared_asset_verify_v1 import (
    INVALID_VERIFY,
    SCHEMA_ID,
    SCHEMA_VERSION,
    SharedAssetVerifyError,
    verify_published_object,
)


def _published(tmp_path: Path, payload: bytes = b"verify-bytes") -> tuple[Path, str]:
    root = tmp_path / "packages"
    source = tmp_path / "model.bin"
    source.write_bytes(payload)
    digest = publish_file(root=root, source=source)
    return root, digest


def _expect_fixed_refusal(invoke: object) -> None:
    assert callable(invoke)
    with pytest.raises(SharedAssetVerifyError) as caught:
        invoke()
    assert type(caught.value) is SharedAssetVerifyError
    assert str(caught.value) == INVALID_VERIFY
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_verify_returns_the_same_size_without_changing_published_bytes(tmp_path: Path) -> None:
    payload = b"verify-bytes"
    root, digest = _published(tmp_path, payload)

    first = verify_published_object(root=root, digest=digest)
    second = verify_published_object(root=root, digest=digest)

    assert first == second == len(payload)
    assert object_path(root=root, digest=digest).read_bytes() == payload
    assert SCHEMA_ID == "lm-atelier-shared-asset-verify-v1"
    assert SCHEMA_VERSION == 1


def test_verify_refuses_missing_and_drifted_objects(tmp_path: Path) -> None:
    root, digest = _published(tmp_path)

    _expect_fixed_refusal(lambda: verify_published_object(root=root, digest="a" * 64))
    object_path(root=root, digest=digest).write_bytes(b"drifted")
    _expect_fixed_refusal(lambda: verify_published_object(root=root, digest=digest))


def test_verify_refuses_an_absent_object_inside_existing_shards(tmp_path: Path) -> None:
    root, digest = _published(tmp_path)
    stored = object_path(root=root, digest=digest)
    stored.unlink()

    assert stored.parent.is_dir()
    _expect_fixed_refusal(lambda: verify_published_object(root=root, digest=digest))


def test_verify_uses_the_held_shards_not_their_path_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = b"held bytes"
    root, digest = _published(tmp_path, original)
    object_path(root=root, digest=digest).write_bytes(b"drifted")
    outside = tmp_path / "outside" / digest[:2] / digest[2:4]
    outside.mkdir(parents=True)
    (outside / digest).write_bytes(original)
    real_open_child = open_child_directory

    def relabel(anchor: AnchoredDirectory, name: str, *, create: bool = False) -> AnchoredDirectory:
        child = real_open_child(anchor, name, create=create)
        if anchor.path.name == digest[:2] and name == digest[2:4]:
            child.path = outside
        return child

    monkeypatch.setattr(verify, "open_child_directory", relabel)

    _expect_fixed_refusal(lambda: verify_published_object(root=root, digest=digest))


def test_verify_uses_the_held_root_not_its_path_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = b"held root bytes"
    root, digest = _published(tmp_path, original)
    object_path(root=root, digest=digest).write_bytes(b"drifted")
    outside = tmp_path / "outside-root"
    target = outside / digest[:2] / digest[2:4]
    target.mkdir(parents=True)
    (target / digest).write_bytes(original)
    real_anchor = AnchoredDirectory

    def relabel_root(path: Path, *, create: bool = False) -> AnchoredDirectory:
        anchor = real_anchor(path, create=create)
        anchor.path = outside
        return anchor

    monkeypatch.setattr(verify, "AnchoredDirectory", relabel_root)

    _expect_fixed_refusal(lambda: verify_published_object(root=root, digest=digest))


@pytest.mark.parametrize("linked_level", ["first", "second", "object"])
def test_verify_refuses_linked_shards_and_objects(tmp_path: Path, linked_level: str) -> None:
    payload = b"linked bytes"
    root, digest = _published(tmp_path, payload)
    first = root / digest[:2]
    second = first / digest[2:4]
    stored = second / digest
    outside = tmp_path / f"outside-{linked_level}"
    outside.mkdir()

    try:
        if linked_level == "first":
            shutil.rmtree(first)
            target = outside / digest[2:4]
            target.mkdir()
            (target / digest).write_bytes(payload)
            os.symlink(outside, first, target_is_directory=True)
        elif linked_level == "second":
            shutil.rmtree(second)
            (outside / digest).write_bytes(payload)
            os.symlink(outside, second, target_is_directory=True)
        else:
            stored.unlink()
            target = outside / "object.bin"
            target.write_bytes(payload)
            os.symlink(target, stored)
    except OSError:
        pytest.skip("this host does not allow filesystem links")

    _expect_fixed_refusal(lambda: verify_published_object(root=root, digest=digest))


def test_verify_refuses_a_non_file_object_entry(tmp_path: Path) -> None:
    root, digest = _published(tmp_path)
    stored = object_path(root=root, digest=digest)
    stored.unlink()
    stored.mkdir()

    _expect_fixed_refusal(lambda: verify_published_object(root=root, digest=digest))


def test_verify_translates_an_unreadable_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, digest = _published(tmp_path)

    def unreadable(_descriptor: int, _size: int) -> bytes:
        raise OSError("opaque read refusal")

    monkeypatch.setattr(os, "read", unreadable)

    _expect_fixed_refusal(lambda: verify_published_object(root=root, digest=digest))


@pytest.mark.parametrize(
    ("root", "digest"),
    [
        (Path("relative"), "a" * 64),
        (Path("relative"), "../../object"),
        (Path.cwd(), "A" * 64),
        (Path.cwd(), "a" * 63),
    ],
)
def test_invalid_inputs_expose_only_the_fixed_refusal(root: Path, digest: str) -> None:
    _expect_fixed_refusal(lambda: verify_published_object(root=root, digest=digest))


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "../../object"])
def test_invalid_digests_do_not_reach_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, digest: str
) -> None:
    def unexpected_storage(*, root: Path, digest: str) -> int:
        raise AssertionError((root, digest))

    monkeypatch.setattr(verify, "_verify_published_object", unexpected_storage)

    _expect_fixed_refusal(lambda: verify_published_object(root=tmp_path, digest=digest))


def test_verify_does_not_discover_or_write_the_desktop_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    root, digest = _published(tmp_path)

    assert verify_published_object(root=root, digest=digest) == len(b"verify-bytes")
    desktop = default_shared_asset_root()
    assert not (desktop / digest[:2] / digest[2:4] / digest).exists()
