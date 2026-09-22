"""What a recursive delete does when a link sits inside the tree it removes.

Several modules delete a directory tree with `shutil.rmtree` and rely on it
sparing whatever a link inside that tree points at. That reliance is sound and
it is not ours: the sparing is done by the standard library, and on Windows it
is done by a private name. Nothing else here exercises it, so a release that
changed it would be found by the incident rather than by the suite.

These two cases are that missing alarm. One reads the mechanism and one reads
the behaviour, because either can change without the other: a coincidence could
keep the behaviour while the mechanism went, and a renamed mechanism could keep
working until the day it did not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _link_dir(link: Path, target: Path) -> bool:
    """Point `link` at `target`, or report that this host will not allow it.

    A junction on Windows, because that is the case the standard library needs
    a private flag to survive, and a symbolic link elsewhere.
    """

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


@pytest.mark.skipif(os.name != "nt", reason="the private flag this guards is the Windows path")
def test_the_flag_a_recursive_delete_needs_on_windows_still_exists() -> None:
    """`shutil` reaches for a private name to tell a junction from a directory.

    On Windows `_rmtree_unsafe` walks with
    `followlinks=os._walk_symlinks_as_files`, a sentinel that makes a link
    appear among the file names so it is unlinked rather than entered. The
    public `followlinks=False` does not do this, because a junction is not a
    link to the predicate that flag consults. If the sentinel goes, every
    recursive delete in this application starts deleting through junctions on
    the same day, and this is the only line that would say so.
    """

    assert hasattr(os, "_walk_symlinks_as_files")


def test_a_recursive_delete_spares_what_a_link_inside_it_points_at(tmp_path: Path) -> None:
    """The behaviour the callers actually rely on, whatever supplies it."""

    tree = tmp_path / "tree"
    (tree / "inner").mkdir(parents=True)
    (tree / "own.bin").write_bytes(b"part of the tree")
    outside = tmp_path / "outside"
    outside.mkdir()
    kept = outside / "kept.bin"
    kept.write_bytes(b"not part of the tree")
    if not _link_dir(tree / "inner" / "link", outside):
        pytest.skip("this host does not allow a directory link to be created")

    shutil.rmtree(tree)

    # Both halves: the tree really went, so the delete was not a no-op that
    # would leave the target intact for the wrong reason.
    assert not tree.exists()
    assert outside.is_dir()
    assert kept.read_bytes() == b"not part of the tree"
