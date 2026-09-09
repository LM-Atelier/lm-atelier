from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_lm.filesystem_links import LinkInspectionFailure, is_link_or_reparse


def test_regular_file_and_symbolic_link_are_distinguished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regular = tmp_path / "regular"
    regular.write_text("content", encoding="utf-8")
    assert not is_link_or_reparse(
        regular,
        missing="raise",
        unreadable="raise",
    )

    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _path: SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0),
    )
    assert is_link_or_reparse(
        tmp_path / "synthetic-link",
        missing="raise",
        unreadable="raise",
    )


def _make_junction(link: Path, target: Path) -> bool:
    """Point `link` at `target` as a real junction, or report host refusal."""

    completed = subprocess.run(  # noqa: S603 - fixed argv, test-local paths
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],  # noqa: S607
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


@pytest.mark.skipif(os.name != "nt", reason="junctions exist only on Windows")
def test_a_symlink_only_check_does_not_see_a_real_junction(tmp_path: Path) -> None:
    """The weaker form of this check cannot come back without failing here.

    The other junction case in this module builds a synthetic one by patching
    `lstat` to report the reparse attribute. That shows the function reads the
    attribute; it cannot show that a real junction carries it, because the fake
    supplies the very fact under test. This makes the real thing and asks both
    questions of it.

    `Path.is_symlink()` answers False for a junction, because a junction is a
    reparse point and not a symbolic link. A containment check written that way
    passes a junction straight through, which is the regression this pins.

    Being exact about what this adds over the synthetic case above, since that
    one does fail if the reparse test is removed from `is_link_or_reparse`:
    what it cannot fail on is the premise itself. It asserts the attribute
    because it supplied the attribute. If a real junction ever stopped
    reporting what the fake reports, every synthetic case here would still pass
    while the production check quietly stopped seeing junctions. This one reads
    the attribute off the filesystem, so the premise is measured rather than
    assumed.

    On Linux the distinction does not exist - `os.symlink` makes a symbolic
    link and `is_symlink()` sees it - so there is nothing to guard there and
    the case is skipped rather than weakened into something that passes
    everywhere.
    """

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "junction"
    if not _make_junction(link, target):
        pytest.skip("this host refuses to create a junction")

    assert link.is_dir(), "the junction should resolve to the target directory"
    assert not link.is_symlink(), "a junction is a reparse point, not a symlink"

    # The premise every synthetic case in this module assumes, taken from the
    # filesystem instead of supplied by a fake: a real junction sets the
    # reparse attribute, and its mode is a directory rather than a link.
    metadata = link.lstat()
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    assert int(getattr(metadata, "st_file_attributes", 0)) & reparse
    assert not stat.S_ISLNK(metadata.st_mode)

    assert is_link_or_reparse(link, missing="raise", unreadable="raise")


def test_windows_reparse_attribute_is_recognized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _path: SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400),
    )
    assert is_link_or_reparse(
        tmp_path / "synthetic-junction",
        missing="raise",
        unreadable="raise",
    )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("assume_link", True), ("assume_regular", False)],
)
def test_missing_path_policy_is_explicit(
    tmp_path: Path,
    policy: LinkInspectionFailure,
    expected: bool,
) -> None:
    assert (
        is_link_or_reparse(
            tmp_path / "missing",
            missing=policy,
            unreadable="raise",
        )
        is expected
    )


def test_missing_path_can_propagate(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        is_link_or_reparse(
            tmp_path / "missing",
            missing="raise",
            unreadable="raise",
        )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("assume_link", True), ("assume_regular", False)],
)
def test_unreadable_path_policy_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: LinkInspectionFailure,
    expected: bool,
) -> None:
    def unavailable(_path: Path) -> None:
        raise PermissionError("unavailable")

    monkeypatch.setattr(Path, "lstat", unavailable)
    assert (
        is_link_or_reparse(
            tmp_path / "unreadable",
            missing="raise",
            unreadable=policy,
        )
        is expected
    )


def test_unreadable_path_can_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_path: Path) -> None:
        raise PermissionError("unavailable")

    monkeypatch.setattr(Path, "lstat", unavailable)
    with pytest.raises(PermissionError, match="unavailable"):
        is_link_or_reparse(
            tmp_path / "unreadable",
            missing="raise",
            unreadable="raise",
        )


def test_held_directory_identity_matches_another_open_of_the_same_directory(tmp_path: Path) -> None:
    from local_lm import filesystem_links as links

    with links.AnchoredDirectory(tmp_path) as first, links.AnchoredDirectory(tmp_path) as second:
        identity = links.directory_identity(first)
        assert links.directory_identity(first) == identity
        assert links.directory_identity(second) == identity


def test_held_directory_identity_distinguishes_another_directory(tmp_path: Path) -> None:
    from local_lm import filesystem_links as links

    other = tmp_path / "other"
    other.mkdir()
    with links.AnchoredDirectory(tmp_path) as first, links.AnchoredDirectory(other) as second:
        assert links.directory_identity(first) != links.directory_identity(second)


def test_held_directory_identity_does_not_reopen_its_path(tmp_path: Path) -> None:
    from local_lm import filesystem_links as links

    other = tmp_path / "other"
    other.mkdir()
    with links.AnchoredDirectory(tmp_path) as anchor:
        identity = links.directory_identity(anchor)
        anchor.path = other
        assert links.directory_identity(anchor) == identity


def test_directory_identity_distinguishes_a_replacement_at_the_same_path(tmp_path: Path) -> None:
    from local_lm import filesystem_links as links

    selected = tmp_path / "selected"
    selected.mkdir()
    with links.AnchoredDirectory(selected) as anchor:
        identity = links.directory_identity(anchor)
    selected.rename(tmp_path / "previous")
    selected.mkdir()
    with links.AnchoredDirectory(selected) as replacement:
        assert links.directory_identity(replacement) != identity


def test_closed_directory_identity_is_refused(tmp_path: Path) -> None:
    from local_lm import filesystem_links as links

    anchor = links.AnchoredDirectory(tmp_path)
    anchor.close()
    with pytest.raises(links.AnchoredDirectoryError):
        links.directory_identity(anchor)
