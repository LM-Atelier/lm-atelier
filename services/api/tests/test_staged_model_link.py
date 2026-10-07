"""Staged model inspection must not read a file through a link."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from local_lm.downloads import DownloadManager


def _make_link_dir(link: Path, target: Path) -> bool:
    """Point ``link`` at ``target``, or report that this host will not allow it."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


def test_staged_inspection_does_not_read_a_linked_file(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    marker = b'{"outside":"marker"}\n'
    outside.write_bytes(marker)
    staging = tmp_path / "staging"
    staging.mkdir()
    link = staging / "card.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(ValueError, match="filesystem link"):
        DownloadManager._inspect_staged_model(staging, ["card.json"], "image")

    assert outside.read_bytes() == marker
    assert link.is_symlink()


def test_staged_inspection_reads_a_regular_json_file(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "card.json").write_bytes(b'{"x": 1}\n')

    inspection = DownloadManager._inspect_staged_model(staging, ["card.json"], "image")

    assert "card.json" in inspection.metadata_files


def test_staged_inspection_does_not_read_through_a_linked_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = b'{"outside":"marker"}\n'
    (outside / "card.json").write_bytes(marker)
    staging = tmp_path / "staging"
    staging.mkdir()
    link = staging / "vae"
    if not _make_link_dir(link, outside):
        pytest.skip("this host refuses to create a directory link")

    with pytest.raises(ValueError, match="filesystem link"):
        DownloadManager._inspect_staged_model(staging, ["vae/card.json"], "image")

    assert (outside / "card.json").read_bytes() == marker
