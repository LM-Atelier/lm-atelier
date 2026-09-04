from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import local_lm.shared_asset_restore_v1 as restore
from local_lm.filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryExists,
    link_entry,
    open_child_directory,
    rename_entry,
)
from local_lm.shared_asset_restore_v1 import (
    INVALID_RESTORE,
    SharedAssetRestoreError,
    restore_quarantined_object,
)
from local_lm.shared_asset_store_v1 import object_path, publish_file

_TOKEN = "collect-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _published(
    tmp_path: Path, payload: bytes = b"immutable shared bytes"
) -> tuple[Path, str, Path]:
    root = tmp_path / "library"
    source = tmp_path / f"source-{hashlib.sha256(payload).hexdigest()[:8]}.bin"
    source.write_bytes(payload)
    digest = publish_file(root=root, source=source)
    return root, digest, object_path(root=root, digest=digest)


def _stage(root: Path, digest: str, token: str = _TOKEN) -> Path:
    with AnchoredDirectory(root) as store:
        first = open_child_directory(store, digest[:2])
        try:
            second = open_child_directory(first, digest[2:4])
            try:
                rename_entry(second, digest, token, replace=False)
            finally:
                second.close()
        finally:
            first.close()
    return root / digest[:2] / digest[2:4] / token


def _refuses(root: Path, digest: str) -> None:
    with pytest.raises(SharedAssetRestoreError, match=INVALID_RESTORE):
        restore_quarantined_object(root=root, package_digest=digest)


def test_restores_one_matching_staged_object_into_the_empty_digest_slot(
    tmp_path: Path,
) -> None:
    root, digest, canonical = _published(tmp_path)
    staged = _stage(root, digest)

    restore_quarantined_object(root=root, package_digest=digest)

    assert canonical.read_bytes() == b"immutable shared bytes"
    assert not staged.exists()


def test_missing_and_drifted_staged_objects_both_refuse(tmp_path: Path) -> None:
    root, digest, canonical = _published(tmp_path)
    staged = _stage(root, digest)
    staged.write_bytes(b"different bytes")

    _refuses(root, digest)
    assert not canonical.exists()
    assert staged.read_bytes() == b"different bytes"

    staged.unlink()
    _refuses(root, digest)


def test_an_occupied_canonical_slot_preserves_both_entries(tmp_path: Path) -> None:
    root, digest, canonical = _published(tmp_path)
    staged = _stage(root, digest)
    canonical.write_bytes(b"newer bytes")

    _refuses(root, digest)

    assert canonical.read_bytes() == b"newer bytes"
    assert staged.read_bytes() == b"immutable shared bytes"


def test_an_occupied_same_object_slot_refuses_before_move_and_preserves_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, digest, canonical = _published(tmp_path)
    staged = canonical.with_name(_TOKEN)
    with AnchoredDirectory(root) as store:
        first = open_child_directory(store, digest[:2])
        try:
            second = open_child_directory(first, digest[2:4])
            try:
                assert link_entry(second, digest, staged.name)
            finally:
                second.close()
        finally:
            first.close()

    def move_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("occupied slot reached the move primitive")

    monkeypatch.setattr(restore, "_rename_entry", move_must_not_run)

    _refuses(root, digest)

    assert canonical.read_bytes() == b"immutable shared bytes"
    assert staged.read_bytes() == b"immutable shared bytes"


def test_ambiguous_matches_refuse_without_removing_either(tmp_path: Path) -> None:
    root, digest, _canonical = _published(tmp_path)
    first = _stage(root, digest)
    second = first.with_name("collect-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    os.link(first, second)

    _refuses(root, digest)

    assert first.read_bytes() == b"immutable shared bytes"
    assert second.read_bytes() == b"immutable shared bytes"


def test_unrelated_well_formed_staged_entries_are_left_alone(tmp_path: Path) -> None:
    root, digest, canonical = _published(tmp_path)
    staged = _stage(root, digest)
    unrelated = staged.with_name("collect-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    unrelated.write_bytes(b"other digest")

    restore_quarantined_object(root=root, package_digest=digest)

    assert canonical.read_bytes() == b"immutable shared bytes"
    assert unrelated.read_bytes() == b"other digest"
    assert not staged.exists()


@pytest.mark.parametrize(
    "unsafe_name",
    ["collect-not-a-token", "collect-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"],
)
def test_malformed_collect_shaped_entries_refuse(tmp_path: Path, unsafe_name: str) -> None:
    root, digest, _canonical = _published(tmp_path)
    staged = _stage(root, digest)
    malformed = staged.with_name(unsafe_name)
    malformed.write_bytes(b"unrelated")

    _refuses(root, digest)

    assert staged.read_bytes() == b"immutable shared bytes"
    assert malformed.read_bytes() == b"unrelated"


def test_a_collect_shaped_directory_refuses_without_touching_it(tmp_path: Path) -> None:
    root, digest, _canonical = _published(tmp_path)
    staged = _stage(root, digest)
    unsafe = staged.with_name("collect-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    unsafe.mkdir()

    _refuses(root, digest)

    assert staged.read_bytes() == b"immutable shared bytes"
    assert unsafe.is_dir()


def test_a_collect_shaped_link_refuses_without_following_it(tmp_path: Path) -> None:
    root, digest, _canonical = _published(tmp_path)
    staged = _stage(root, digest)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside bytes")
    unsafe = staged.with_name("collect-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    try:
        os.symlink(outside, unsafe)
    except OSError:
        pytest.skip("this host does not allow file links")

    _refuses(root, digest)

    assert staged.read_bytes() == b"immutable shared bytes"
    assert unsafe.is_symlink()
    assert outside.read_bytes() == b"outside bytes"


def test_missing_shards_refuse_without_creating_the_root(tmp_path: Path) -> None:
    root = tmp_path / "absent-library"

    _refuses(root, "f" * 64)

    assert not root.exists()


def test_a_destination_collision_immediately_before_move_preserves_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, digest, canonical = _published(tmp_path)
    staged = _stage(root, digest)
    real_rename = restore._rename_entry

    def collide(
        anchor: AnchoredDirectory,
        source: str,
        destination: str,
        *,
        replace: bool,
    ) -> None:
        if source == staged.name:
            canonical.write_bytes(b"concurrent bytes")
        real_rename(anchor, source, destination, replace=replace)

    monkeypatch.setattr(restore, "_rename_entry", collide)

    _refuses(root, digest)

    assert staged.read_bytes() == b"immutable shared bytes"
    assert canonical.read_bytes() == b"concurrent bytes"


def test_a_same_object_destination_collision_preserves_both_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, digest, canonical = _published(tmp_path)
    staged = _stage(root, digest)

    def collide_with_same_object(
        anchor: AnchoredDirectory,
        source: str,
        destination: str,
        *,
        replace: bool,
    ) -> None:
        if source == staged.name:
            os.link(staged, canonical)
        raise AnchoredEntryExists("contained operation refused")

    monkeypatch.setattr(restore, "_rename_entry", collide_with_same_object)

    _refuses(root, digest)

    assert staged.read_bytes() == b"immutable shared bytes"
    assert canonical.read_bytes() == b"immutable shared bytes"


def test_source_substitution_is_refused_without_moving_an_unproven_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, digest, canonical = _published(tmp_path)
    staged = _stage(root, digest)
    original = staged.with_name("original-away")
    real_rename = restore._rename_entry

    def substitute(
        anchor: AnchoredDirectory,
        source: str,
        destination: str,
        *,
        replace: bool,
    ) -> None:
        if source == staged.name:
            staged.rename(original)
            staged.write_bytes(b"immutable shared bytes")
        real_rename(anchor, source, destination, replace=replace)

    monkeypatch.setattr(restore, "_rename_entry", substitute)

    _refuses(root, digest)

    assert canonical.read_bytes() == b"immutable shared bytes"
    assert original.read_bytes() == b"immutable shared bytes"
    assert not staged.exists()


def test_in_place_drift_after_validation_is_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, digest, canonical = _published(tmp_path)
    staged = _stage(root, digest)
    real_rename = restore._rename_entry

    def drift(
        anchor: AnchoredDirectory,
        source: str,
        destination: str,
        *,
        replace: bool,
    ) -> None:
        if source == staged.name:
            staged.write_bytes(b"drifted staged bytes")
        real_rename(anchor, source, destination, replace=replace)

    monkeypatch.setattr(restore, "_rename_entry", drift)

    _refuses(root, digest)

    assert not canonical.exists()
    assert staged.read_bytes() == b"drifted staged bytes"


def test_a_replaced_destination_after_move_is_not_rolled_into_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, digest, canonical = _published(tmp_path)
    staged = _stage(root, digest)
    moved_away = canonical.with_name("moved-away")
    real_rename = restore._rename_entry

    def replace_destination(
        anchor: AnchoredDirectory,
        source: str,
        destination: str,
        *,
        replace: bool,
    ) -> None:
        real_rename(anchor, source, destination, replace=replace)
        if source == staged.name:
            canonical.rename(moved_away)
            canonical.write_bytes(b"new canonical bytes")

    monkeypatch.setattr(restore, "_rename_entry", replace_destination)

    _refuses(root, digest)

    assert not staged.exists()
    assert moved_away.read_bytes() == b"immutable shared bytes"
    assert canonical.read_bytes() == b"new canonical bytes"


def test_a_partial_non_replacing_move_keeps_the_staged_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, digest, canonical = _published(tmp_path)
    staged = _stage(root, digest)

    def link_then_refuse(
        anchor: AnchoredDirectory,
        source: str,
        destination: str,
        *,
        replace: bool,
    ) -> None:
        assert replace is False
        assert link_entry(anchor, source, destination)
        raise AnchoredDirectoryError("contained operation refused")

    monkeypatch.setattr(restore, "_rename_entry", link_then_refuse)

    _refuses(root, digest)

    assert staged.read_bytes() == b"immutable shared bytes"
    assert not canonical.exists()
