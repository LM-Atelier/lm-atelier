"""The walk for unindexed files stops on a clock and goes on from where it stopped.

Every retention pass ends by walking the store's shard directories for files
no artifact row names. It runs with the database writer held, and on a store
of some fifteen thousand files it took six to seven seconds, past the five
every other writer waits before giving up; a sweep reached it on every pass,
including one with nothing to delete. The automatic sweep now walks for its
clock and resumes after the last leaf shard it finished, so a pass walks the
whole store across several short batches. A person's own cleanup and a preview
still walk everything at once.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm import artifacts as artifacts_module
from local_lm import main as main_module
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base


@pytest.fixture
def store_and_session(tmp_path: Path) -> Iterator[tuple[ArtifactStore, Session]]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_engine(f"sqlite:///{settings.data_dir / 'walk.sqlite3'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield ArtifactStore(settings), session
    engine.dispose()


def _age(path: Path) -> None:
    old = (datetime.now(UTC) - timedelta(hours=25)).timestamp()
    os.utime(path, (old, old))


def _aged_orphans(store: ArtifactStore, count: int) -> list[Path]:
    """Unindexed canonical files, a day old, each under its own digest's shards."""

    orphans = []
    for index in range(count):
        content = f"unindexed {index}".encode()
        digest = hashlib.sha256(content).hexdigest()
        orphan = store.root / digest[:2] / digest[2:4] / digest
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(content)
        _age(orphan)
        orphans.append(orphan)
    return orphans


def _canonical(store: ArtifactStore, leaf: str, index: int = 0, *, aged: bool) -> Path:
    """An unindexed file named as the store names one, in the leaf shard `leaf` spells."""

    name = f"{leaf}{index:060x}"
    path = store.root / name[:2] / name[2:4] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())
    if aged:
        _age(path)
    return path


def _leaf(path: Path) -> tuple[str, str]:
    return path.parent.parent.name, path.parent.name


def _clean(
    store: ArtifactStore,
    session: Session,
    walk_seconds: float | None,
    *,
    max_deletions: int | None = None,
) -> tuple[int, bool]:
    summary = store.cleanup_retention(
        session,
        retention_days=30,
        temporary_hours=24,
        dry_run=False,
        max_deletions=max_deletions,
        walk_seconds=walk_seconds,
    )
    session.commit()
    return summary.removed_orphan_file_count, summary.truncated


def test_a_walk_its_clock_cuts_short_goes_on_where_it_stopped(
    store_and_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = store_and_session
    orphans = _aged_orphans(store, 5)
    leaves = sorted({_leaf(path) for path in orphans})
    assert len(leaves) == 5, "fixture needs five distinct leaf shards"

    # A clock of zero still walks one leaf per call, so the walk moves forward.
    calls = [_clean(store, session, 0.0) for _ in leaves]

    assert calls == [(1, True)] * 4 + [(1, False)]
    assert not any(path.exists() for path in orphans)


@pytest.mark.parametrize("listing", ["as listed", "reversed"])
def test_the_resume_point_advances_one_leaf_at_a_time_in_name_order(
    store_and_session: tuple[ArtifactStore, Session],
    monkeypatch: pytest.MonkeyPatch,
    listing: str,
) -> None:
    store, session = store_and_session
    orphans = [_canonical(store, leaf, aged=True) for leaf in ("0b01", "0a00", "0b00", "0a01")]
    if listing == "reversed":
        # Windows lists a directory in name order, which would hide a walk
        # that follows the listing; a reversed one is how another looks.
        list_entries = artifacts_module.list_entries
        monkeypatch.setattr(
            artifacts_module,
            "list_entries",
            lambda *args, **kwargs: list(reversed(list_entries(*args, **kwargs))),
        )

    places = []
    for _ in orphans:
        _clean(store, session, 0.0)
        places.append(store._walk_resume_after)

    # Each call finishes the next leaf in name order, two of them inside each
    # first-level shard; the call that reaches the end of the store clears
    # the place, so the next walk starts again at the top.
    assert places == [("0a", "00"), ("0a", "01"), ("0b", "00"), None]
    assert not any(path.exists() for path in orphans)


def test_a_walk_passes_leaves_it_cannot_empty_to_reach_the_ones_after(
    store_and_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = store_and_session
    young = [_canonical(store, leaf, aged=False) for leaf in ("0a00", "0a01", "0b00")]
    orphan = _canonical(store, "0b01", aged=True)

    # Each call walks one leaf. The first three hold only files too new to
    # remove, so they stay; only a walk that goes on from where it stopped
    # ever reaches the last one.
    calls = [_clean(store, session, 0.0) for _ in range(4)]

    assert calls == [(0, True)] * 3 + [(1, False)]
    assert not orphan.exists()
    assert all(path.exists() for path in young)


def test_a_leaf_the_walk_cut_short_is_walked_again(
    store_and_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = store_and_session
    orphans = [_canonical(store, "0a00", index, aged=True) for index in range(2)]

    # The deletion bound stops the first call inside the leaf, so the leaf is
    # not finished: the next call walks it again rather than going on after it.
    calls = [_clean(store, session, 60.0, max_deletions=1) for _ in range(2)]

    assert calls == [(1, True), (1, False)]
    assert not any(path.exists() for path in orphans)


def test_an_unbounded_walk_still_reaches_the_end_in_one_call(
    store_and_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = store_and_session
    orphans = _aged_orphans(store, 5)

    assert _clean(store, session, None) == (5, False)
    assert not any(path.exists() for path in orphans)
    assert store._walk_resume_after is None


def test_a_preview_walks_the_whole_store_whatever_clock_it_is_given(
    store_and_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = store_and_session
    orphans = _aged_orphans(store, 3)

    summary = store.cleanup_retention(
        session, retention_days=30, temporary_hours=24, dry_run=True, walk_seconds=0.0
    )

    # A preview reports everything a cleanup would remove, and moves no place
    # a later sweep would resume from.
    assert (summary.removed_orphan_file_count, summary.truncated) == (3, False)
    assert store._walk_resume_after is None
    assert all(path.exists() for path in orphans)


@pytest.mark.asyncio
async def test_the_sweep_walks_a_large_store_across_short_batches(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    store = ArtifactStore(settings)
    orphans = _aged_orphans(store, 4)
    caplog.set_level(logging.INFO)

    await main_module.sweep_artifact_retention(store, settings, pause_seconds=0, batch_seconds=0)

    assert not any(path.exists() for path in orphans)
    removed_per_batch = [
        int(match.group(1))
        for match in (
            re.search(r"(\d+) unindexed file\(s\) removed; elapsed", record.getMessage())
            for record in caplog.records
            if "retention batch committed" in record.getMessage()
        )
        if match
    ]
    # With a clock of zero each batch walks one leaf, so no batch walks more
    # than one file's worth of the store while it holds the writer.
    assert sum(removed_per_batch) == 4
    assert max(removed_per_batch) == 1
    assert len(removed_per_batch) >= 4
