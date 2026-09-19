"""A contained walk and tree removal: nothing reached through a link is entered or touched."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from local_lm import filesystem_links
from local_lm.filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryKind,
    AnchoredListingStopped,
    WalkedEntry,
    read_entry,
    remove_tree,
    walk_entries,
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


def _outside(tmp_path: Path) -> Path:
    """Make a directory outside every tree under test, holding one file."""

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "foreign.bin").write_bytes(b"not part of the tree")
    return outside


def _tree(root: Path) -> None:
    """Build root/top.bin, root/a/middle.bin, root/a/b/deep.bin and an empty root/empty."""

    (root / "a" / "b").mkdir(parents=True)
    (root / "empty").mkdir()
    (root / "top.bin").write_bytes(b"top")
    (root / "a" / "middle.bin").write_bytes(b"middle")
    (root / "a" / "b" / "deep.bin").write_bytes(b"deep")


def _walk(
    root: Path, *, max_depth: int = 64, limit: int = 200_000
) -> list[tuple[tuple[str, ...], AnchoredEntryKind]]:
    with AnchoredDirectory(root) as anchor:
        return [
            (walked.parts, walked.entry.kind)
            for walked in walk_entries(anchor, max_depth=max_depth, limit=limit)
        ]


#: Where a link is planted: at the root, one level down and two levels down.
LINK_DEPTHS = [(), ("a",), ("a", "b")]


def test_a_walk_yields_every_entry_with_its_place_directories_first(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    _tree(root)

    walked = _walk(root)

    assert sorted(walked) == sorted(
        [
            (("top.bin",), AnchoredEntryKind.FILE),
            (("a",), AnchoredEntryKind.DIRECTORY),
            (("a", "middle.bin"), AnchoredEntryKind.FILE),
            (("a", "b"), AnchoredEntryKind.DIRECTORY),
            (("a", "b", "deep.bin"), AnchoredEntryKind.FILE),
            (("empty",), AnchoredEntryKind.DIRECTORY),
        ]
    )
    order = [parts for parts, _ in walked]
    assert order.index(("a",)) < order.index(("a", "middle.bin"))
    assert order.index(("a", "b")) < order.index(("a", "b", "deep.bin"))


@pytest.mark.parametrize("depth", LINK_DEPTHS, ids=["root", "one-down", "two-down"])
def test_a_walk_reports_a_link_at_any_depth_and_never_enters_it(
    tmp_path: Path, depth: tuple[str, ...]
) -> None:
    root = tmp_path / "tree"
    _tree(root)
    outside = _outside(tmp_path)
    if not _make_link_dir(root.joinpath(*depth, "link"), outside):
        pytest.skip("this host refuses to create a directory link")

    walked = _walk(root)

    assert ((*depth, "link"), AnchoredEntryKind.LINK) in walked
    assert not [parts for parts, _ in walked if "foreign.bin" in parts], (
        "the walk reported a file reached through the link as the tree's own"
    )
    assert (outside / "foreign.bin").read_bytes() == b"not part of the tree"


def test_the_held_parent_reads_the_entry_it_was_listed_with(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    _tree(root)

    with AnchoredDirectory(root) as anchor:
        contents = {
            walked.parts: read_entry(walked.parent, walked.entry.name)
            for walked in walk_entries(anchor)
            if walked.entry.kind is AnchoredEntryKind.FILE
        }

    assert contents == {
        ("top.bin",): b"top",
        ("a", "middle.bin"): b"middle",
        ("a", "b", "deep.bin"): b"deep",
    }


def test_a_directory_swapped_for_a_link_mid_walk_refuses_instead_of_being_entered(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    (root / "a").mkdir(parents=True)
    (root / "a" / "inner.bin").write_bytes(b"inner")
    outside = _outside(tmp_path)

    with AnchoredDirectory(root) as anchor:
        walk = walk_entries(anchor)
        first: WalkedEntry = next(walk)
        assert first.parts == ("a",)
        # The walk is paused after listing `a` and before entering it.
        shutil.rmtree(root / "a")
        if not _make_link_dir(root / "a", outside):
            pytest.skip("this host refuses to create a directory link")
        with pytest.raises(AnchoredDirectoryError):
            next(walk)

    assert (outside / "foreign.bin").read_bytes() == b"not part of the tree"


def test_a_tree_deeper_than_the_bound_refuses_and_one_at_the_bound_does_not(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    _tree(root)

    assert len(_walk(root, max_depth=3)) == 6
    with pytest.raises(AnchoredDirectoryError):
        _walk(root, max_depth=2)


def test_more_entries_than_the_bound_refuses(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    _tree(root)

    assert len(_walk(root, limit=6)) == 6
    with pytest.raises(AnchoredDirectoryError):
        _walk(root, limit=5)


@pytest.mark.parametrize("bound", [{"max_depth": 0}, {"limit": 0}, {"max_depth": -1}])
def test_a_zero_or_negative_bound_refuses(tmp_path: Path, bound: dict[str, int]) -> None:
    root = tmp_path / "tree"
    _tree(root)

    with pytest.raises(AnchoredDirectoryError):
        _walk(root, **bound)
    with AnchoredDirectory(tmp_path) as anchor, pytest.raises(AnchoredDirectoryError):
        remove_tree(anchor, "tree", **bound)
    assert (root / "a" / "b" / "deep.bin").exists()


def test_a_stop_request_abandons_the_walk_within_a_level_already_read(tmp_path: Path) -> None:
    """One flat level is listed whole before its first entry is yielded, so only
    a check before every entry can stop the walk partway through it."""
    root = tmp_path / "tree"
    root.mkdir()
    for index in range(4):
        (root / f"file-{index}.bin").write_bytes(b"flat")
    seen: list[tuple[str, ...]] = []

    with AnchoredDirectory(root) as anchor, pytest.raises(AnchoredListingStopped):
        for walked in walk_entries(anchor, should_stop=lambda: len(seen) >= 2):
            seen.append(walked.parts)

    assert len(seen) == 2


def test_removing_a_tree_removes_all_of_it_and_nothing_beside_it(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    _tree(root)
    (tmp_path / "sibling.bin").write_bytes(b"stays")

    with AnchoredDirectory(tmp_path) as anchor:
        remove_tree(anchor, "tree")

    assert not root.exists()
    assert (tmp_path / "sibling.bin").read_bytes() == b"stays"


@pytest.mark.parametrize("depth", LINK_DEPTHS, ids=["root", "one-down", "two-down"])
def test_removing_a_tree_that_holds_a_link_refuses_before_removing_anything(
    tmp_path: Path, depth: tuple[str, ...]
) -> None:
    root = tmp_path / "tree"
    _tree(root)
    # Listed ahead of the directories that lead to the link wherever a listing is
    # sorted, so a removal that met the link level by level would take these first.
    (root / "0-first.bin").write_bytes(b"first")
    (root / "a" / "0-first.bin").write_bytes(b"first")
    outside = _outside(tmp_path)
    if not _make_link_dir(root.joinpath(*depth, "link"), outside):
        pytest.skip("this host refuses to create a directory link")

    with AnchoredDirectory(tmp_path) as anchor, pytest.raises(AnchoredDirectoryError):
        remove_tree(anchor, "tree")

    assert (outside / "foreign.bin").read_bytes() == b"not part of the tree"
    assert root.joinpath(*depth, "link").exists(), "the link itself was removed"
    for kept in ("0-first.bin", "top.bin", "a/0-first.bin", "a/middle.bin", "a/b/deep.bin"):
        assert (root / kept).exists(), f"{kept} was removed before the refusal"


def test_removing_a_name_that_is_itself_a_link_refuses_and_keeps_both_ends(
    tmp_path: Path,
) -> None:
    outside = _outside(tmp_path)
    if not _make_link_dir(tmp_path / "tree", outside):
        pytest.skip("this host refuses to create a directory link")

    with AnchoredDirectory(tmp_path) as anchor, pytest.raises(AnchoredDirectoryError):
        remove_tree(anchor, "tree")

    assert (tmp_path / "tree").exists()
    assert (outside / "foreign.bin").read_bytes() == b"not part of the tree"


def test_removing_an_absent_tree_succeeds_and_a_file_refuses(tmp_path: Path) -> None:
    (tmp_path / "plain.bin").write_bytes(b"a file, not a tree")

    with AnchoredDirectory(tmp_path) as anchor:
        remove_tree(anchor, "never-there")
        with pytest.raises(AnchoredDirectoryError):
            remove_tree(anchor, "plain.bin")

    assert (tmp_path / "plain.bin").read_bytes() == b"a file, not a tree"


def test_removing_a_tree_deeper_than_the_bound_refuses_before_removing_anything(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    _tree(root)

    with AnchoredDirectory(tmp_path) as anchor:
        with pytest.raises(AnchoredDirectoryError):
            remove_tree(anchor, "tree", max_depth=2)
        assert (root / "top.bin").exists()
        remove_tree(anchor, "tree", max_depth=3)

    assert not root.exists()


def test_a_link_that_appears_after_the_first_walk_stops_removal_at_its_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first walk proves the tree as it was. A link planted after it is met
    level by level: that level refuses before any of it is touched."""
    root = tmp_path / "tree"
    _tree(root)
    outside = _outside(tmp_path)
    if not _make_link_dir(root / "a" / "link", outside):
        pytest.skip("this host refuses to create a directory link")
    # Stand in for a tree that was still safe when the first walk read it.
    monkeypatch.setattr(filesystem_links, "walk_entries", lambda *_args, **_kwargs: iter(()))

    with AnchoredDirectory(tmp_path) as anchor, pytest.raises(AnchoredDirectoryError):
        remove_tree(anchor, "tree")

    assert (outside / "foreign.bin").read_bytes() == b"not part of the tree"
    assert (root / "a" / "link").exists()
    assert (root / "a" / "middle.bin").read_bytes() == b"middle"
    assert (root / "a" / "b" / "deep.bin").read_bytes() == b"deep"


@pytest.mark.parametrize("kind", ["file link", "directory link"])
def test_a_link_put_in_a_listed_files_place_is_never_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The one moment no check closes: between listing a file and removing it
    by name. A link substituted then may itself be unlinked, or the removal
    refuses; either way its target and everything outside stay as they were."""
    root = tmp_path / "tree"
    root.mkdir()
    entry = root / "entry.bin"
    entry.write_bytes(b"listed as a file")
    outside = _outside(tmp_path)
    target = outside / "foreign.bin" if kind == "file link" else outside
    real_remove = filesystem_links.remove_entry

    def substitute_then_remove(anchor: AnchoredDirectory, name: str) -> None:
        entry.unlink()
        if kind == "file link":
            try:
                entry.symlink_to(target)
            except OSError:
                pytest.skip("this host refuses to create a file link")
        elif not _make_link_dir(entry, target):
            pytest.skip("this host refuses to create a directory link")
        real_remove(anchor, name)

    monkeypatch.setattr(filesystem_links, "remove_entry", substitute_then_remove)
    refused = False
    with AnchoredDirectory(tmp_path) as anchor:
        try:
            remove_tree(anchor, "tree")
        except AnchoredDirectoryError:
            refused = True

    assert (outside / "foreign.bin").read_bytes() == b"not part of the tree"
    assert sorted(path.name for path in outside.iterdir()) == ["foreign.bin"]
    if not refused:
        assert not os.path.lexists(entry), "a completed removal left the substituted link"
