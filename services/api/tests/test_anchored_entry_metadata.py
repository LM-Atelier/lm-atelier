from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_lm import filesystem_links
from local_lm.filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntry,
    AnchoredEntryKind,
    _from_filetime,
    _posix_metadata,
    _windows_metadata,
    list_entries,
    remove_directory_entry,
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


def _entries(root: Path) -> dict[str, AnchoredEntry]:
    with AnchoredDirectory(root) as anchor:
        return {entry.name: entry for entry in list_entries(anchor)}


def test_a_file_and_a_directory_carry_size_and_time(tmp_path: Path) -> None:
    """The measurement both planned consumers need, and could not have.

    Catalogue prune and the artifact orphan sweep each enumerated, checked for
    a link, stat-ed and then deleted - four resolutions of one name, the last
    of them destructive. This is the third of those four, taken through the
    anchor instead.
    """

    root = tmp_path / "store"
    root.mkdir()
    (root / "payload.bin").write_bytes(b"x" * 1234)
    (root / "nested").mkdir()
    before = datetime.now(tz=UTC)

    entries = _entries(root)

    assert entries["payload.bin"].kind is AnchoredEntryKind.FILE
    assert entries["payload.bin"].size_bytes == 1234
    assert entries["nested"].kind is AnchoredEntryKind.DIRECTORY
    assert entries["nested"].size_bytes is not None
    for name in ("payload.bin", "nested"):
        entry = entries[name]
        assert entry.has_metadata
        assert entry.modified_at is not None
        assert entry.modified_at.tzinfo is not None, "a naive timestamp is not an instant"
        # Generous either way: the point is that it is a real recent instant
        # rather than 1601 or a zero, not that the clocks agree exactly.
        assert abs((entry.modified_at - before).total_seconds()) < 600


def test_a_link_carries_no_metadata_even_though_the_record_has_it(tmp_path: Path) -> None:
    """Withheld on purpose, and the reason is not tidiness.

    The Windows record supplies a size and a timestamp for a junction as
    readily as for a directory, and they describe the LINK rather than what it
    points at. Reporting them invites a consumer to reason about an entry it
    has just been told to skip.
    """

    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.bin").write_bytes(b"not ours")
    if not _make_link_dir(root / "redirect", outside):
        pytest.skip("this host does not permit directory links")

    entry = _entries(root)["redirect"]

    assert entry.kind is AnchoredEntryKind.LINK
    assert not entry.is_safe
    assert not entry.has_metadata
    assert entry.size_bytes is None
    assert entry.modified_at is None


def test_metadata_is_all_or_nothing() -> None:
    """One question for the consumer, not two that can disagree."""

    assert AnchoredEntry("a", AnchoredEntryKind.FILE, 1, datetime.now(tz=UTC)).has_metadata
    assert not AnchoredEntry("a", AnchoredEntryKind.FILE, 1, None).has_metadata
    assert not AnchoredEntry("a", AnchoredEntryKind.FILE, None, datetime.now(tz=UTC)).has_metadata
    assert not AnchoredEntry("a", AnchoredEntryKind.FILE).has_metadata


def test_the_windows_mapping_withholds_metadata_without_touching_anything() -> None:
    """A pure function of three values, so no fallback can hide in it."""

    written = 133_000_000_000_000_000
    size, modified = _windows_metadata(AnchoredEntryKind.FILE, 4096, written)
    assert size == 4096
    assert modified is not None and modified.tzinfo is not None

    for unsafe in (AnchoredEntryKind.LINK, AnchoredEntryKind.OTHER, AnchoredEntryKind.UNKNOWN):
        assert _windows_metadata(unsafe, 4096, written) == (None, None)
    # A record that did not record a time, and a nonsensical size.
    assert _windows_metadata(AnchoredEntryKind.FILE, 4096, 0) == (None, None)
    assert _windows_metadata(AnchoredEntryKind.FILE, -1, written) == (None, None)


def test_an_unrecorded_nt_timestamp_is_absent_rather_than_1601() -> None:
    """Zero means "not recorded", and 1601 is not a modification time."""

    assert _from_filetime(0) is None
    assert _from_filetime(-1) is None
    assert _from_filetime(2**63 - 1) is None
    recent = _from_filetime(133_000_000_000_000_000)
    assert recent is not None
    assert recent.year > 2000


def test_an_unsafe_posix_entry_is_never_reacquired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POSIX pays one anchored lookup, and only for a safe kind.

    The reacquisition is the one thing this platform does that Windows does
    not, so the case that must never reach it is pinned directly: every route
    back to the filesystem is armed to fail, and an unsafe kind must still
    answer without touching any of them.
    """

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("an unsafe entry was reacquired")

    monkeypatch.setattr(filesystem_links, "open_entry", refuse)
    monkeypatch.setattr(filesystem_links, "open_child_directory", refuse)
    monkeypatch.setattr(os, "stat", refuse)
    monkeypatch.setattr(os, "lstat", refuse)

    root = tmp_path / "store"
    root.mkdir()
    with AnchoredDirectory(root) as anchor:
        for unsafe in (
            AnchoredEntryKind.LINK,
            AnchoredEntryKind.OTHER,
            AnchoredEntryKind.UNKNOWN,
        ):
            assert _posix_metadata(anchor, "anything", unsafe) == (None, None)


def test_a_half_measured_posix_entry_carries_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The all-or-nothing rule proven where the pair is ASSEMBLED.

    `AnchoredEntry` promises size and time are absent together, never one
    without the other, and `_posix_metadata` upholds it today by returning both
    or neither. That makes the assembly site's own check unprovable by any
    ordinary run: the state it guards against never arises.

    Measured - changing `or` to `and` in `_with_posix_metadata` survives the
    whole suite. So the guard is driven directly here instead, with a stub that
    returns exactly the half pair a future measurement path could produce. A
    docstring that no test can falsify is prose, not a property.
    """

    root = tmp_path / "store"
    root.mkdir()
    (root / "payload.bin").write_bytes(b"x")

    def always(measured: tuple[int | None, datetime | None]):
        # Bound here rather than captured from the loop: a lambda closing over
        # the loop variable reads the LAST value if it ever outlives the
        # iteration, which is a real trap even where this use is safe.
        def measure(*_args: object, **_kwargs: object) -> tuple[int | None, datetime | None]:
            return measured

        return measure

    for half in ((1234, None), (None, datetime.now(tz=UTC))):
        monkeypatch.setattr(filesystem_links, "_posix_metadata", always(half))
        with AnchoredDirectory(root) as anchor:
            attached = filesystem_links._with_posix_metadata(
                anchor, AnchoredEntry("payload.bin", AnchoredEntryKind.FILE)
            )
        assert not attached.has_metadata
        assert attached.size_bytes is None
        assert attached.modified_at is None


def test_an_empty_directory_is_removed_through_the_held_parent(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "spent").mkdir()
    (root / "kept.bin").write_bytes(b"x")

    with AnchoredDirectory(root) as anchor:
        remove_directory_entry(anchor, "spent")
        remaining = sorted(entry.name for entry in list_entries(anchor))

    assert remaining == ["kept.bin"]


def test_absence_is_success_so_a_pruner_does_not_race_itself(tmp_path: Path) -> None:
    """Two sweeps over the same listing must not make the second one fail."""

    root = tmp_path / "store"
    root.mkdir()
    (root / "spent").mkdir()

    with AnchoredDirectory(root) as anchor:
        remove_directory_entry(anchor, "spent")
        remove_directory_entry(anchor, "spent")
        remove_directory_entry(anchor, "never-existed")


def test_a_populated_directory_refuses_rather_than_recursing(tmp_path: Path) -> None:
    """Not recursive on purpose. A caller wanting recursion must say so."""

    root = tmp_path / "store"
    root.mkdir()
    (root / "full").mkdir()
    (root / "full" / "inside.bin").write_bytes(b"keep me")

    with AnchoredDirectory(root) as anchor, pytest.raises(AnchoredDirectoryError):
        remove_directory_entry(anchor, "full")

    assert (root / "full" / "inside.bin").read_bytes() == b"keep me"


def test_a_file_refuses(tmp_path: Path) -> None:
    """A directory remover that removes files is a delete with a wrong name."""

    root = tmp_path / "store"
    root.mkdir()
    (root / "ordinary.bin").write_bytes(b"x")

    with AnchoredDirectory(root) as anchor, pytest.raises(AnchoredDirectoryError):
        remove_directory_entry(anchor, "ordinary.bin")

    assert (root / "ordinary.bin").read_bytes() == b"x"


def test_a_linked_directory_refuses_and_its_target_survives(tmp_path: Path) -> None:
    """The case that cannot be handled by rmdir on a path.

    On Windows a junction IS a directory to the ordinary APIs, so removing "the
    empty directory" would remove the link - and a caller pruning a tree would
    silently detach whatever it pointed at. The entry is opened through the
    held parent with FILE_OPEN_REPARSE_POINT and refused for carrying one.
    """

    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.bin").write_bytes(b"not ours")
    if not _make_link_dir(root / "redirect", outside):
        pytest.skip("this host does not permit directory links")

    with AnchoredDirectory(root) as anchor, pytest.raises(AnchoredDirectoryError):
        remove_directory_entry(anchor, "redirect")

    assert (outside / "victim.bin").read_bytes() == b"not ours"
    assert (root / "redirect").exists()


def test_a_link_to_an_EMPTY_directory_refuses_and_that_directory_survives(
    tmp_path: Path,
) -> None:
    """The same case with the accident removed, which is the whole point.

    The test above plants a file in the link's target, and that file is doing
    work nobody asked it to do. Measured: drop FILE_OPEN_REPARSE_POINT from the
    open and that test still passes, because the target is then opened as an
    ordinary directory and refuses only for being NON-EMPTY. The flag looks
    load-bearing and is not being tested.

    With an EMPTY target the accident is gone, and the same mutation deletes the
    target directory outright - refused False, the directory gone. So this is
    the case that actually pins the flag, and it is the shape a real pruner
    meets constantly: an empty directory behind a link.
    """

    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if not _make_link_dir(root / "redirect", outside):
        pytest.skip("this host does not permit directory links")

    with AnchoredDirectory(root) as anchor, pytest.raises(AnchoredDirectoryError):
        remove_directory_entry(anchor, "redirect")

    assert outside.is_dir(), "the link's target was removed through the link"
    assert (root / "redirect").exists()


def test_a_name_that_is_not_one_component_refuses(tmp_path: Path) -> None:
    """The same validation every other operation on this anchor applies."""

    root = tmp_path / "store"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "nested" / "deeper").mkdir()

    with AnchoredDirectory(root) as anchor:
        for hostile in ("nested/deeper", "..", ".", "", "nested\\deeper"):
            with pytest.raises(AnchoredDirectoryError):
                remove_directory_entry(anchor, hostile)

    assert (root / "nested" / "deeper").is_dir()
