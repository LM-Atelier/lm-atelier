"""Download cleanup removes and measures its own files only, never through a link."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from local_lm import filesystem_links
from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.domain import JobKind, JobStatus
from local_lm.downloads import DownloadManager, _prune_empty_directories
from local_lm.events import EventBroker
from local_lm.filesystem_links import AnchoredDirectory, AnchoredEntry
from local_lm.models import Job, ModelInstall


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
    """Make a directory outside managed storage holding one file and one empty directory."""

    outside = tmp_path / "outside"
    (outside / "empty").mkdir(parents=True)
    (outside / "foreign.bin").write_bytes(b"x" * 1000)
    return outside


def _assert_outside_intact(outside: Path) -> None:
    assert (outside / "foreign.bin").read_bytes() == b"x" * 1000
    assert (outside / "empty").is_dir(), "a directory outside the tree was pruned"


def _manager(settings: Settings) -> DownloadManager:
    settings.prepare()
    configure_database(settings)
    init_db()
    return DownloadManager(settings, EventBroker())


def test_a_tree_size_counts_its_own_files_and_nothing_through_a_link(
    settings: Settings, tmp_path: Path
) -> None:
    partial = settings.download_dir / "sized.partial"
    (partial / "nested").mkdir(parents=True)
    (partial / "own.bin").write_bytes(b"a" * 10)
    (partial / "nested" / "own.bin").write_bytes(b"b" * 5)
    outside = _outside(tmp_path)
    if not _make_link_dir(partial / "nested" / "link", outside):
        pytest.skip("this host refuses to create a directory link")

    assert DownloadManager._path_size(partial) == 15
    assert DownloadManager._path_size(partial / "nested" / "link") == 0


def test_partial_cleanup_leaves_a_partial_holding_a_link_and_counts_nothing_for_it(
    settings: Settings, tmp_path: Path
) -> None:
    manager = _manager(settings)
    plain = settings.download_dir / "job_plain.partial"
    linked = settings.download_dir / "job_linked.partial"
    (plain / "nested").mkdir(parents=True)
    (plain / "nested" / "chunk").write_bytes(b"p" * 7)
    (linked / "nested").mkdir(parents=True)
    (linked / "nested" / "chunk").write_bytes(b"l" * 3)
    outside = _outside(tmp_path)
    if not _make_link_dir(linked / "nested" / "link", outside):
        pytest.skip("this host refuses to create a directory link")

    with SessionLocal() as session:
        removed_count, reclaimed_bytes = manager.cleanup_partials(session)

    assert (removed_count, reclaimed_bytes) == (1, 7)
    assert not plain.exists()
    assert (linked / "nested" / "chunk").read_bytes() == b"l" * 3
    _assert_outside_intact(outside)


def test_quarantine_cleanup_never_reaches_through_a_link(
    settings: Settings, tmp_path: Path
) -> None:
    manager = _manager(settings)
    quarantine = settings.download_dir / ".discarded-installs"
    plain = quarantine / "job_done-discard_plain"
    linked = quarantine / "job_done-discard_linked"
    plain.mkdir(parents=True)
    (plain / "model").write_bytes(b"m" * 4)
    linked.mkdir()
    (linked / "model").write_bytes(b"k" * 2)
    outside = _outside(tmp_path)
    if not _make_link_dir(linked / "link", outside):
        pytest.skip("this host refuses to create a directory link")

    with SessionLocal() as session:
        removed_count, reclaimed_bytes = manager.cleanup_partials(session)

    assert (removed_count, reclaimed_bytes) == (1, 4)
    assert not plain.exists()
    assert (linked / "model").read_bytes() == b"k" * 2
    _assert_outside_intact(outside)


def test_discarding_a_partial_that_holds_a_link_leaves_it_and_everything_outside(
    settings: Settings, tmp_path: Path
) -> None:
    manager = _manager(settings)
    partial = settings.download_dir / "job_cancelled.partial"
    partial.mkdir(parents=True)
    (partial / "chunk").write_bytes(b"c")
    outside = _outside(tmp_path)
    if not _make_link_dir(partial / "link", outside):
        pytest.skip("this host refuses to create a directory link")

    manager._discard_partial("job_cancelled")

    assert (partial / "chunk").read_bytes() == b"c"
    _assert_outside_intact(outside)
    # With the link gone, the same discard removes the whole partial.
    if os.name == "nt":
        (partial / "link").rmdir()
    else:
        (partial / "link").unlink()
    manager._discard_partial("job_cancelled")
    assert not partial.exists()


def test_pruning_below_an_install_removes_only_its_own_empty_directories(
    tmp_path: Path,
) -> None:
    install = tmp_path / "install"
    (install / "empty" / "deeper").mkdir(parents=True)
    (install / "kept").mkdir()
    (install / "kept" / "file.bin").write_bytes(b"k")
    outside = _outside(tmp_path)
    if not _make_link_dir(install / "link", outside):
        pytest.skip("this host refuses to create a directory link")

    _prune_empty_directories(install)

    assert not (install / "empty").exists()
    assert (install / "kept" / "file.bin").read_bytes() == b"k"
    assert (install / "link").exists()
    assert install.is_dir()
    _assert_outside_intact(outside)


def test_a_shared_install_is_pruned_without_reaching_through_a_link(
    settings: Settings, tmp_path: Path
) -> None:
    """Discarding one of two installs sharing a directory moves its own files away and
    prunes what emptied; a link inside the shared directory is neither entered nor pruned."""
    manager = _manager(settings)
    shared = settings.model_dir / "shared"
    (shared / "own").mkdir(parents=True)
    (shared / "own" / "discarded.bin").write_bytes(b"d")
    (shared / "kept.bin").write_bytes(b"k")
    outside = _outside(tmp_path)
    if not _make_link_dir(shared / "link", outside):
        pytest.skip("this host refuses to create a directory link")
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="install_kept",
                name="Kept",
                role="image",
                engine="comfyui",
                local_path=str(shared),
                manifest_json={"files": ["kept.bin"]},
                active=True,
            )
        )
        session.add(
            Job(id="job_discard", kind=JobKind.DOWNLOAD.value, status=JobStatus.FAILED.value)
        )
        session.commit()
        discarded = ModelInstall(
            id="install_discarded",
            name="Discarded",
            role="image",
            engine="comfyui",
            local_path=str(shared),
            manifest_json={"files": ["own/discarded.bin"]},
            active=False,
        )
        moves, quarantined = manager._quarantine_provisional_files(
            session, discarded, "job_discard"
        )

    assert quarantined is not None and len(moves) == 1
    assert not (shared / "own").exists()
    assert (shared / "kept.bin").read_bytes() == b"k"
    assert (shared / "link").exists()
    _assert_outside_intact(outside)


def test_a_partial_whose_child_disappears_mid_removal_is_neither_removed_nor_counted(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Something vanishing inside the tree stops the removal before the partial
    goes. That is a refusal, not evidence the partial is gone."""
    manager = _manager(settings)
    partial = settings.download_dir / "job_raced.partial"
    child = partial / "vanishes"
    child.mkdir(parents=True)
    (partial / "kept.bin").write_bytes(b"retained")
    real_list_entries = filesystem_links.list_entries
    removed_child = False

    def remove_child_after_listing(
        anchor: AnchoredDirectory,
        *,
        limit: int = 8192,
        include_metadata: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[AnchoredEntry, ...]:
        nonlocal removed_child
        entries = real_list_entries(
            anchor, limit=limit, include_metadata=include_metadata, should_stop=should_stop
        )
        if anchor.path == partial and not include_metadata and not removed_child:
            child.rmdir()
            removed_child = True
        return entries

    monkeypatch.setattr(filesystem_links, "list_entries", remove_child_after_listing)
    with SessionLocal() as session:
        result = manager.cleanup_partials(session)

    assert removed_child
    assert partial.is_dir()
    assert (partial / "kept.bin").read_bytes() == b"retained"
    assert result == (0, 0)
