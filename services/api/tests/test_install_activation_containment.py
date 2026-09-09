from __future__ import annotations

import ctypes
import errno
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_lm.downloads import DownloadManager
from local_lm.filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryExists,
    publish_opened_file,
    take_regular_file,
)


def _make_link_dir(link: Path, target: Path) -> bool:
    """Point `link` at `target`, or report that this host will not allow it."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


def _make_link_file(link: Path, target: Path) -> bool:
    """Point `link` at a file, or report that this host will not allow it."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", str(link), str(target)], capture_output=True
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link)
    except OSError:
        return False
    return True


def test_staging_activation_adds_files_to_an_existing_revision(tmp_path: Path) -> None:
    destination = tmp_path / "installed"
    destination.mkdir()
    (destination / "first.safetensors").write_bytes(b"first")
    staging = tmp_path / "staging"
    nested = staging / "split_files" / "vae"
    nested.mkdir(parents=True)
    (nested / "second.safetensors").write_bytes(b"second")

    DownloadManager._activate_staging(staging, destination)

    assert (destination / "first.safetensors").read_bytes() == b"first"
    assert (destination / "split_files" / "vae" / "second.safetensors").read_bytes() == b"second"
    assert not staging.exists()


def test_activation_into_missing_destination_replaces_the_staging_tree(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "installed"
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "model.safetensors").write_bytes(b"weights")

    DownloadManager._activate_staging(staging, destination)

    assert (destination / "model.safetensors").read_bytes() == b"weights"
    assert not staging.exists()


def test_activation_refuses_a_linked_destination_and_keeps_staging(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real-install"
    real.mkdir()
    destination = tmp_path / "linked-install"
    if not _make_link_dir(destination, real):
        pytest.skip("this host cannot create a directory link")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="model staging cannot use filesystem links"):
        DownloadManager._activate_staging(staging, destination)

    assert (staging / "model.safetensors").read_bytes() == b"weights"
    assert list(real.iterdir()) == []


def test_activation_does_not_delete_through_a_staging_link(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "installed"
    nested = destination / "nested"
    nested.mkdir(parents=True)
    destination_payload = nested / "payload.bin"
    destination_payload.write_bytes(b"kept")
    outside = tmp_path / "outside"
    outside.mkdir()
    foreign = outside / "payload.bin"
    foreign.write_bytes(b"foreign")
    staging = tmp_path / "staging"
    staging.mkdir()
    linked = staging / "nested"
    if not _make_link_dir(linked, outside):
        pytest.skip("this host cannot create a directory link")

    with pytest.raises(ValueError, match="model staging cannot use filesystem links"):
        DownloadManager._activate_staging(staging, destination)

    assert foreign.read_bytes() == b"foreign"
    assert destination_payload.read_bytes() == b"kept"
    assert staging.exists()


def test_missing_destination_does_not_publish_a_staging_link(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "installed"
    outside = tmp_path / "outside"
    outside.mkdir()
    foreign = outside / "payload.bin"
    foreign.write_bytes(b"foreign")
    staging = tmp_path / "staging"
    staging.mkdir()
    linked = staging / "nested"
    if not _make_link_dir(linked, outside):
        pytest.skip("this host cannot create a directory link")

    with pytest.raises(ValueError, match="model staging cannot use filesystem links"):
        DownloadManager._activate_staging(staging, destination)

    assert not destination.exists()
    assert staging.exists()
    assert foreign.read_bytes() == b"foreign"


def test_activation_refuses_a_linked_destination_parent(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    if not _make_link_dir(linked_parent, real_parent):
        pytest.skip("this host cannot create a directory link")
    destination = linked_parent / "installed"
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="model staging cannot use filesystem links"):
        DownloadManager._activate_staging(staging, destination)

    assert (staging / "model.safetensors").read_bytes() == b"weights"
    assert list(real_parent.iterdir()) == []


def test_contained_tree_size_refuses_a_linked_child(tmp_path: Path) -> None:
    destination = tmp_path / "installed"
    destination.mkdir()
    (destination / "own.bin").write_bytes(b"abcd")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "huge.bin").write_bytes(b"x" * 100)
    linked = destination / "link"
    if not _make_link_dir(linked, outside):
        pytest.skip("this host cannot create a directory link")

    with pytest.raises(ValueError, match="model staging cannot use filesystem links"):
        DownloadManager._contained_tree_size(destination)


def test_held_file_publish_keeps_the_opened_object_after_a_name_swap(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "staged"
    dest_dir = tmp_path / "installed"
    source_dir.mkdir()
    dest_dir.mkdir()
    original = source_dir / "weights.bin"
    original.write_bytes(b"staged-bytes")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"foreign-bytes")

    with (
        AnchoredDirectory(source_dir) as staged,
        AnchoredDirectory(dest_dir) as dest,
    ):
        opened = take_regular_file(staged, "weights.bin")
        assert opened is not None
        try:
            if os.name == "nt":
                with pytest.raises(PermissionError):
                    original.unlink()
            else:
                original.unlink()
                if not _make_link_file(original, outside):
                    pytest.skip("this host cannot create a file link")
            publish_opened_file(
                staged,
                "weights.bin",
                opened,
                into=dest,
                destination="weights.bin",
            )
        finally:
            os.close(opened)

    assert (dest_dir / "weights.bin").read_bytes() == b"staged-bytes"
    assert outside.read_bytes() == b"foreign-bytes"


def test_take_regular_file_refuses_a_replacement_link(tmp_path: Path) -> None:
    source_dir = tmp_path / "staged"
    source_dir.mkdir()
    payload = source_dir / "weights.bin"
    payload.write_bytes(b"staged-bytes")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"foreign-bytes")
    payload.unlink()
    if not _make_link_file(payload, outside):
        pytest.skip("this host cannot create a file link")

    with AnchoredDirectory(source_dir) as staged, pytest.raises(AnchoredDirectoryError):
        take_regular_file(staged, "weights.bin")


def test_copy_fallback_preserves_bytes_when_a_write_is_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import filesystem_links

    payload = b"model data for the installed selection"
    source = tmp_path / "staged.bin"
    destination = tmp_path / "installed.bin"
    source.write_bytes(payload)
    opened = os.open(source, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    real_open = os.open
    real_write = os.write

    def open_destination(name: str, flags: int, mode: int, *, dir_fd: int) -> int:
        assert name == "installed.bin"
        assert dir_fd == 123
        return real_open(destination, flags | getattr(os, "O_BINARY", 0), mode)

    def short_write(descriptor: int, data: bytes) -> int:
        return real_write(descriptor, data[: max(1, len(data) // 2)])

    monkeypatch.setattr(filesystem_links.os, "open", open_destination)
    monkeypatch.setattr(filesystem_links.os, "write", short_write)
    try:
        filesystem_links._copy_opened_posix(opened, 123, "installed.bin")
    finally:
        os.close(opened)

    assert destination.read_bytes() == payload


class _DecliningLibc:
    """A libc whose `linkat` refuses with a chosen errno, as the real one would.

    Hosts that cannot hard-link cannot be arranged inside a test run: a second
    mount, a missing capability or an exhausted link count are properties of the
    machine. The refusal is produced here instead, at the same boundary and in
    the same way - errno set at call time, minus one returned - so what the
    module reads is what a kernel would have left it.
    """

    def __init__(self, failure: int) -> None:
        self._failure = failure

    def linkat(self, *_arguments: Any) -> int:
        ctypes.set_errno(self._failure)
        return -1


def _decline_linking(monkeypatch: pytest.MonkeyPatch, failure: int) -> None:
    from local_lm import filesystem_links

    monkeypatch.setattr(filesystem_links.ctypes, "CDLL", lambda *_a, **_k: _DecliningLibc(failure))


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("EXDEV", id="different-mounts"),
        pytest.param("EACCES", id="capability-withheld"),
        pytest.param("ENOENT", id="empty-path-unsupported"),
        pytest.param("EMLINK", id="link-count-exhausted"),
        pytest.param("ENOSYS", id="call-unavailable"),
    ],
)
def test_a_host_that_will_not_hard_link_falls_back_to_a_copy(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import filesystem_links

    code = getattr(errno, failure, None)
    if code is None:
        pytest.skip(f"{failure} is not defined on this host")
    _decline_linking(monkeypatch, code)

    assert filesystem_links._link_opened_posix(0, 0, "weights.bin") is False


def test_a_taken_destination_name_still_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_lm import filesystem_links

    _decline_linking(monkeypatch, errno.EEXIST)

    with pytest.raises(AnchoredEntryExists):
        filesystem_links._link_opened_posix(0, 0, "weights.bin")


@pytest.mark.skipif(os.name == "nt", reason="publication only links on POSIX hosts")
def test_publication_completes_by_copy_when_the_host_cannot_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "staged"
    dest_dir = tmp_path / "installed"
    source_dir.mkdir()
    dest_dir.mkdir()
    (source_dir / "weights.bin").write_bytes(b"staged-bytes")

    with (
        AnchoredDirectory(source_dir) as staged,
        AnchoredDirectory(dest_dir) as dest,
    ):
        opened = take_regular_file(staged, "weights.bin")
        assert opened is not None
        held = os.fstat(opened)
        try:
            _decline_linking(monkeypatch, errno.EXDEV)
            publish_opened_file(staged, "weights.bin", opened, into=dest)
        finally:
            os.close(opened)

    published = dest_dir / "weights.bin"
    assert published.read_bytes() == b"staged-bytes"
    assert not (source_dir / "weights.bin").exists()
    # A copy, not the same inode under a second name: the point of the fallback
    # is that publication finishes where linking is impossible.
    assert published.stat().st_ino != held.st_ino
