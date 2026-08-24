from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from local_lm.filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntry,
    AnchoredEntryKind,
    list_entries,
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


def _kinds(root: Path) -> dict[str, AnchoredEntryKind]:
    with AnchoredDirectory(root) as anchor:
        return {entry.name: entry.kind for entry in list_entries(anchor)}


def test_ordinary_files_and_directories_carry_their_kind(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "artifact.bin").write_bytes(b"payload")
    (root / "nested").mkdir()
    (root / "nested" / "inner.bin").write_bytes(b"deeper")

    kinds = _kinds(root)

    assert kinds == {
        "artifact.bin": AnchoredEntryKind.FILE,
        "nested": AnchoredEntryKind.DIRECTORY,
    }


def test_the_listing_never_includes_dot_or_dot_dot(tmp_path: Path) -> None:
    """Both are real entries on Windows and both are traversals upward.

    A caller that deleted or swept everything it listed would, with `..`
    present, be handed its own parent as a name to act on.
    """

    root = tmp_path / "store"
    root.mkdir()
    (root / "only.bin").write_bytes(b"x")

    names = set(_kinds(root))

    assert names == {"only.bin"}


def test_a_linked_entry_is_reported_as_a_link_not_a_directory(tmp_path: Path) -> None:
    """The property every consumer depends on.

    A junction carries the directory attribute AND the reparse attribute, so a
    classifier that asks "is it a directory" first calls it a directory - and a
    sweep that then recurses or deletes follows the link out of the store. The
    kind must come from the reparse bit being checked first.
    """

    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.bin").write_bytes(b"not ours")
    if not _make_link_dir(root / "redirect", outside):
        pytest.skip("this host does not permit directory links")
    (root / "ordinary").mkdir()

    kinds = _kinds(root)

    assert kinds["redirect"] is AnchoredEntryKind.LINK
    assert kinds["ordinary"] is AnchoredEntryKind.DIRECTORY
    assert (outside / "victim.bin").read_bytes() == b"not ours"


def test_a_link_is_not_safe_and_an_ordinary_entry_is(tmp_path: Path) -> None:
    """`is_safe` is what consumers branch on, so it is pinned separately.

    Classifying correctly and then exposing a predicate that says yes to a
    link would leave every caller correct in theory and wrong in practice.
    """

    root = tmp_path / "store"
    root.mkdir()
    (root / "keep.bin").write_bytes(b"x")
    outside = tmp_path / "outside"
    outside.mkdir()
    if not _make_link_dir(root / "redirect", outside):
        pytest.skip("this host does not permit directory links")

    with AnchoredDirectory(root) as anchor:
        entries = {entry.name: entry for entry in list_entries(anchor)}

    assert entries["keep.bin"].is_safe
    assert not entries["redirect"].is_safe


def test_more_entries_than_the_bound_refuses(tmp_path: Path) -> None:
    """An unbounded listing is unbounded work for every caller of it."""

    root = tmp_path / "store"
    root.mkdir()
    for index in range(6):
        (root / f"entry-{index}.bin").write_bytes(b"x")

    with AnchoredDirectory(root) as anchor:
        assert len(list_entries(anchor, limit=6)) == 6
        with pytest.raises(AnchoredDirectoryError):
            list_entries(anchor, limit=5)


def test_a_zero_or_negative_bound_refuses(tmp_path: Path) -> None:
    """Zero is not "no limit"; it is a caller that has lost track of one."""

    root = tmp_path / "store"
    root.mkdir()

    with AnchoredDirectory(root) as anchor:
        with pytest.raises(AnchoredDirectoryError):
            list_entries(anchor, limit=0)
        with pytest.raises(AnchoredDirectoryError):
            list_entries(anchor, limit=-1)


def test_a_closed_anchor_refuses_rather_than_answering(tmp_path: Path) -> None:
    """Closing releases the guarantee, so the answer must go with it.

    A listing taken through a released anchor is a listing by path again, and
    it would look identical to a contained one.
    """

    root = tmp_path / "store"
    root.mkdir()
    (root / "present.bin").write_bytes(b"x")

    anchor = AnchoredDirectory(root)
    assert list_entries(anchor)
    anchor.close()

    with pytest.raises(AnchoredDirectoryError):
        list_entries(anchor)


def test_a_redirected_directory_cannot_be_anchored_or_listed(tmp_path: Path) -> None:
    """Acquisition refuses first, so nothing outside is ever enumerated."""

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.bin").write_bytes(b"not ours")
    redirected = tmp_path / "store"
    if not _make_link_dir(redirected, outside):
        pytest.skip("this host does not permit directory links")

    with (
        pytest.raises(AnchoredDirectoryError),
        AnchoredDirectory(redirected) as anchor,
    ):
        list_entries(anchor)


def test_an_empty_directory_lists_as_empty_rather_than_refusing(tmp_path: Path) -> None:
    """Absence of entries is an answer; a refusal here would be a lie.

    On Windows an empty directory still yields `.` and `..`, so this also
    proves those two are filtered rather than merely absent from the fixtures.
    """

    root = tmp_path / "store"
    root.mkdir()

    with AnchoredDirectory(root) as anchor:
        assert list_entries(anchor) == ()


def test_an_entry_record_cannot_be_edited_after_it_is_read(tmp_path: Path) -> None:
    """The kind is evidence from the kernel, not a field a caller can set."""

    entry = AnchoredEntry("name.bin", AnchoredEntryKind.LINK)

    with pytest.raises((AttributeError, TypeError)):
        entry.kind = AnchoredEntryKind.FILE  # type: ignore[misc]
    assert entry.kind is AnchoredEntryKind.LINK
    assert not entry.is_safe
