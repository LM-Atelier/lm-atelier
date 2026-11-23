"""A retention pass reads each artifact once, however many batches it takes.

Every batch of a pass used to start again at the oldest artifact, so the kept
artifacts ahead of the deletable ones were read again, under the database
writer, by every batch after the first, and a batch with nothing to remove was
bounded only by the size of the store. Each batch now goes on where the one
before it stopped, and its clock bounds what it reads as well as what it
removes. A pass still ends only when its rows and its walk of the store's files
are both finished, and the sweep runs another whenever any batch of a pass
removed something.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import UTC, datetime, timedelta
from itertools import count
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import delete, event, func, select, text
from sqlalchemy.orm import Session

from local_lm import artifact_reference_index as reference_index_module
from local_lm import artifacts as artifacts_module
from local_lm import main as main_module
from local_lm.artifact_reference_index import IndexedReferenceSnapshot
from local_lm.artifacts import ArtifactStore, RetentionCleanupSummary, RetentionCursor
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.models import Artifact, Job
from local_lm.retention_policy import windows_for

OLD = datetime(2020, 1, 1)


def _artifact(
    store: ArtifactStore,
    session: Session,
    label: str,
    *,
    created_at: datetime,
    temporary: bool,
    kind: ArtifactKind = ArtifactKind.IMAGE,
    **metadata: Any,
) -> Artifact:
    """An artifact created at a pinned time: a preview the sweep may remove, or a kept one."""

    artifact = store.ingest_bytes(
        session,
        label.encode(),
        kind=kind,
        media_type="video/mp4" if kind is ArtifactKind.VIDEO else "image/png",
        metadata={"temporary_preview": True, **metadata} if temporary else dict(metadata),
    )
    session.flush()
    artifact.created_at = created_at
    session.flush()
    return artifact


def _count(session: Session) -> int:
    return int(session.scalar(select(func.count(Artifact.id))) or 0)


def _recording(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, RetentionCleanupSummary]]:
    """Every batch the sweep runs: the cursor it was given and the summary it gave back."""

    calls: list[tuple[Any, RetentionCleanupSummary]] = []
    real = ArtifactStore.cleanup_retention

    def recording(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        summary = real(self, session, **kwargs)
        calls.append((kwargs.get("cursor"), summary))
        return summary

    monkeypatch.setattr(ArtifactStore, "cleanup_retention", recording)
    return calls


def _first_pass(calls: list[tuple[Any, RetentionCleanupSummary]]) -> list[RetentionCleanupSummary]:
    batches: list[RetentionCleanupSummary] = []
    for _cursor, summary in calls:
        batches.append(summary)
        if not summary.truncated:
            break
    return batches


def _clock_came_off_only_after_idle_batches(
    caplog: pytest.LogCaptureFixture, calls: list[tuple[Any, RetentionCleanupSummary]]
) -> None:
    """The clock may come off only after a batch that examined nothing and walked no shard.

    Such a batch, a walk whose clock ran out while it listed the store, would be
    repeated exactly by the next; a batch that read artifacts or finished a
    shard moved the pass on and keeps the clock.
    """

    for record in caplog.records:
        message = record.getMessage()
        if "continuing without the clock" in message:
            number = int(message.split("batch ", 1)[1].split(" ", 1)[0])
            idle = calls[number - 1][1]
            assert idle.examined_count == 0 and idle.walked_shards == 0, message


async def test_a_pass_reads_each_artifact_once_however_many_batches_it_takes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thirty kept artifacts sit ahead of twenty that may go, five a batch.

    Read from the start every time, the kept thirty were read by each of the
    four batches that removed something, and by the fifth, which only had the
    walk of the files left after the fourth spent its deletions on the last
    previews. Going on from the batch before, each artifact is read once, plus
    once more for each batch that stopped at an artifact it was about to
    remove, since the next batch begins with it.
    """

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(30):
            _artifact(
                store,
                session,
                f"kept {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=False,
            )
        for index in range(20):
            _artifact(
                store,
                session,
                f"expired preview {index}",
                created_at=OLD + timedelta(days=1, minutes=index),
                temporary=True,
            )
        session.commit()
        stored = _count(session)
    calls = _recording(monkeypatch)

    await main_module.sweep_artifact_retention(
        store, settings, batch_deletions=5, pause_seconds=0, batch_seconds=3600
    )

    first = _first_pass(calls)
    stopped_among_rows = sum(
        1
        for summary in first
        if summary.truncated and summary.cursor is not None and not summary.cursor.rows_done
    )
    assert [summary.removed_count for summary in first] == [5, 5, 5, 5, 0]
    assert first[-1].examined_count == 0, "the last batch had only the walk left"
    assert sum(summary.examined_count for summary in first) == stored + stopped_among_rows, (
        "a later batch of the pass read again what an earlier one had read"
    )
    with SessionLocal() as check:
        assert _count(check) == stored - 20


async def test_a_batch_with_nothing_to_remove_stops_on_its_clock_and_the_next_goes_on(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Forty kept artifacts and a clock that runs out after a handful of them.

    Without a place to go on from, a batch that found nothing to remove read the
    whole store before it could stop, so its clock never bounded it. Now the
    clock ends it among the kept artifacts, and the batches after it cover the
    rest exactly once.
    """

    caplog.set_level(logging.INFO)
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(40):
            _artifact(
                store,
                session,
                f"kept {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=False,
            )
        session.commit()
        stored = _count(session)
    calls = _recording(monkeypatch)
    # A tenth of a second a reading: each artifact the batch examines reads the
    # clock once, so a one-second budget runs out a few artifacts in.
    readings = count()
    monkeypatch.setattr(
        main_module, "time", SimpleNamespace(monotonic=lambda: next(readings) * 0.1)
    )

    await asyncio.wait_for(
        main_module.sweep_artifact_retention(store, settings, pause_seconds=0, batch_seconds=1.0),
        timeout=30,
    )

    first = _first_pass(calls)
    reading = [summary for summary in first if summary.examined_count]
    assert len(reading) > 1, "one batch read every kept artifact"
    assert reading[0].truncated and reading[0].cursor is not None
    assert reading[0].examined_count < stored
    assert sum(summary.examined_count for summary in first) == stored
    # A batch that read artifacts moved the pass on: the clock stays for the next.
    _clock_came_off_only_after_idle_batches(caplog, calls)
    with SessionLocal() as check:
        assert _count(check) == stored


async def test_a_pass_whose_last_batch_removes_nothing_still_earns_another(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poster freed by a deletion in an earlier batch of the same pass still goes.

    The poster comes first and cannot go while the video naming it survives.
    The video goes in the first batch, whose clock then runs out among kept
    artifacts, so the batch that finishes the pass removes nothing. Deciding on
    that batch alone would end the sweep with the poster still stored.
    """

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        poster = _artifact(store, session, "poster", created_at=OLD, temporary=True)
        video = _artifact(
            store,
            session,
            "video naming the poster",
            created_at=OLD + timedelta(minutes=1),
            temporary=True,
            kind=ArtifactKind.VIDEO,
            poster_artifact_id=poster.id,
        )
        for index in range(30):
            _artifact(
                store,
                session,
                f"kept {index}",
                created_at=OLD + timedelta(minutes=2 + index),
                temporary=False,
            )
        session.commit()
        poster_id, video_id = poster.id, video.id
    calls = _recording(monkeypatch)
    readings = count()
    monkeypatch.setattr(
        main_module, "time", SimpleNamespace(monotonic=lambda: next(readings) * 0.1)
    )

    await main_module.sweep_artifact_retention(store, settings, pause_seconds=0, batch_seconds=1.0)

    first = _first_pass(calls)
    assert first[0].removed_count == 1 and first[0].truncated, "the video went in the first batch"
    assert first[-1].removed_count == 0, "the batch that finished the pass removed nothing"
    with SessionLocal() as check:
        assert check.get(Artifact, video_id) is None
        assert check.get(Artifact, poster_id) is None, "the sweep ended with the poster stored"


def test_artifacts_created_at_the_same_moment_are_each_examined_once(
    settings: Settings,
) -> None:
    """The order breaks a tie in creation time by id, and the place does the same."""

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(12):
            _artifact(store, session, f"tied {index}", created_at=OLD, temporary=False)
        session.commit()
        tied = set(session.scalars(select(Artifact.id).where(Artifact.created_at == OLD)))
        stored = _count(session)
    seen: list[int] = []
    cursor: RetentionCursor | None = RetentionCursor()
    examining = False
    asked = 0

    def report(name: str) -> None:
        nonlocal examining
        examining = examining or name == "examine-artifacts"

    def stop_after_five() -> bool:
        # Counted only once the artifacts are being examined, so the stop falls
        # exactly between the fifth and the sixth.
        nonlocal asked
        if not examining:
            return False
        asked += 1
        return asked > 5

    with SessionLocal() as session:
        summary = store.cleanup_retention(
            session,
            retention_days=30,
            temporary_hours=24,
            dry_run=False,
            should_stop=stop_after_five,
            report_phase=report,
            cursor=cursor,
        )
        session.commit()
        seen.append(summary.examined_count)
        assert summary.truncated and summary.cursor is not None
        assert summary.cursor.after is not None and summary.cursor.after[1] in tied
        cursor = summary.cursor
    with SessionLocal() as session:
        summary = store.cleanup_retention(
            session, retention_days=30, temporary_hours=24, dry_run=False, cursor=cursor
        )
        session.commit()
        seen.append(summary.examined_count)

    assert seen[0] == 5
    assert sum(seen) == stored, "a tied artifact was skipped or read twice"
    assert summary.cursor is None and not summary.truncated


def test_a_batch_that_only_walks_the_files_takes_no_snapshot(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the rows of a pass are done, the batches walking the files read no references."""

    store = ArtifactStore(settings)
    snapshots: list[bool] = []
    real = ArtifactStore.referenced_artifact_ids

    def counting(session: Session, *, for_deletion: bool = False) -> Any:
        snapshots.append(for_deletion)
        return real(session, for_deletion=for_deletion)

    monkeypatch.setattr(ArtifactStore, "referenced_artifact_ids", staticmethod(counting))
    with SessionLocal() as session:
        summary = store.cleanup_retention(
            session,
            retention_days=30,
            temporary_hours=24,
            dry_run=False,
            cursor=RetentionCursor(rows_done=True),
        )
        session.commit()

    assert snapshots == []
    assert summary.examined_count == 0
    assert summary.cursor is None and not summary.truncated


async def test_a_batch_whose_file_walk_runs_out_of_time_keeps_its_clock(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Walking the files moves a pass on even when it removes nothing.

    The walk goes on from the last shard it finished and walks at least one each
    batch, so a batch that ran out of time in it made progress. Taking the clock
    off after it, as for a batch that could do nothing at all, left every batch
    of the next pass free to read the whole store under the writer.
    """

    caplog.set_level(logging.INFO)
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(8):
            _artifact(
                store,
                session,
                f"kept {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=False,
            )
        session.commit()
    calls = _recording(monkeypatch)
    requested: list[int | None] = []
    recorded = ArtifactStore.cleanup_retention

    def asking(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        requested.append(kwargs.get("max_deletions"))
        return recorded(self, session, **kwargs)

    monkeypatch.setattr(ArtifactStore, "cleanup_retention", asking)
    # Every walk ends after one shard, whatever its clock says.
    monkeypatch.setattr(artifacts_module._BoundedWalk, "over", lambda walk: walk.leaves > 0)

    await main_module.sweep_artifact_retention(store, settings, pause_seconds=0, batch_seconds=3600)

    walked = [summary for _cursor, summary in calls if summary.examined_count == 0]
    assert calls[0][1].truncated and calls[0][1].removed_count == 0
    assert walked, "no batch went on with the walk alone"
    assert "continuing without the clock" not in caplog.text
    assert set(requested) == {artifacts_module.RETENTION_BATCH_DELETIONS}


def test_a_batch_stops_once_it_has_as_much_metadata_to_rewrite_as_it_may(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Marking a kept artifact rewrites its metadata, so a batch marks only so many."""

    monkeypatch.setattr(artifacts_module, "RETENTION_BATCH_METADATA_CHANGES", 5)
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(12):
            _artifact(
                store,
                session,
                f"unreferenced {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=False,
            )
        session.commit()
    marked: list[int] = []
    cursor: RetentionCursor | None = RetentionCursor()
    while cursor is not None and len(marked) < 10:
        with SessionLocal() as session:
            summary = store.cleanup_retention(
                session, retention_days=30, temporary_hours=24, dry_run=False, cursor=cursor
            )
            session.commit()
        marked.append(summary.marked_count)
        cursor = summary.cursor

    assert marked[:2] == [5, 5], "a batch stopped short of what it may rewrite"
    assert max(marked) <= 5, "a batch rewrote more metadata than it may"
    with SessionLocal() as check:
        documents = check.scalars(
            select(Artifact.metadata_json).where(Artifact.created_at < OLD + timedelta(days=1))
        )
        assert sum(1 for metadata in documents if "unreferenced_at" in metadata) == 12


def test_a_pass_goes_on_past_creation_times_stored_without_fractions(
    settings: Settings,
) -> None:
    """Times written by older versions, or by hand, have no fraction of a second.

    Bound back as a parsed time, the place would read `...00.000000`, which sorts
    after a stored `...00`, and every artifact of that second with a larger id
    would be passed by. The place is compared exactly as stored instead.
    """

    store = ArtifactStore(settings)
    stored = [
        "2020-01-01 00:00:00",
        "2020-01-01 00:00:00",
        "2020-01-01 00:00:00",
        "2020-01-01 00:00:00.000000",
        "2020-01-01 00:00:00.000000",
        "2020-01-01 00:00:01.000000",
    ]
    with SessionLocal() as session:
        made = [
            _artifact(store, session, f"preview {index}", created_at=OLD, temporary=True).id
            for index in range(len(stored))
        ]
        for artifact_id, created in zip(made, stored, strict=True):
            session.execute(
                text("UPDATE artifacts SET created_at = :created WHERE id = :id"),
                {"created": created, "id": artifact_id},
            )
        session.commit()
    removed: list[int] = []
    cursor: RetentionCursor | None = RetentionCursor()
    while cursor is not None and len(removed) < 12:
        with SessionLocal() as session:
            summary = store.cleanup_retention(
                session,
                retention_days=30,
                temporary_hours=24,
                dry_run=False,
                max_deletions=1,
                cursor=cursor,
            )
            session.commit()
        removed.append(summary.removed_count)
        cursor = summary.cursor

    assert sum(removed) == len(stored), "the pass went past an artifact without examining it"
    with SessionLocal() as check:
        assert not any(check.get(Artifact, artifact_id) for artifact_id in made)


def test_the_artifacts_after_a_place_are_found_through_the_index(settings: Settings) -> None:
    """A chunk past a place is a range in the index, not a sort of the whole table."""

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(3):
            _artifact(
                store,
                session,
                f"kept {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=False,
            )
        session.commit()
        first = session.scalars(select(Artifact.id).order_by(Artifact.created_at)).first()
    reads: list[tuple[str, Any]] = []

    def capture(_connection: Any, _cursor: Any, statement: str, parameters: Any, *_: Any) -> None:
        if "FROM artifacts" in statement and "LIMIT" in statement and ") > (" in statement:
            reads.append((statement, parameters))

    engine = SessionLocal.kw["bind"]
    event.listen(engine, "before_cursor_execute", capture)
    try:
        with SessionLocal() as session:
            store.cleanup_retention(
                session,
                retention_days=30,
                temporary_hours=24,
                dry_run=False,
                cursor=RetentionCursor(after=("2020-01-01 00:00:00.000000", str(first))),
            )
            session.rollback()
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    assert reads, "no chunk was read past the place"
    with engine.connect() as connection:
        plan = [
            str(row[-1])
            for row in connection.exec_driver_sql("EXPLAIN QUERY PLAN " + reads[0][0], reads[0][1])
        ]
    assert any("ix_artifacts_created_at_id" in step for step in plan), plan
    assert not any("TEMP B-TREE" in step for step in plan), plan


async def test_a_batch_with_a_budget_examines_an_artifact_however_the_clock_jumps(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A clock that moved past the budget between two readings still lets a batch move on.

    The budget is armed at the first artifact. Had that arming been able to end
    the batch at once, a long pause between two readings would leave a batch
    that did nothing, and the sweep would take its clock off for the rest of the
    sweep, leaving every later batch free to read the whole store.
    """

    caplog.set_level(logging.INFO)
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(6):
            _artifact(
                store,
                session,
                f"kept {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=False,
            )
        session.commit()
        stored = _count(session)
    calls = _recording(monkeypatch)
    # A whole second a reading against a half-second budget: every reading after
    # the arming one is past the deadline.
    readings = count()
    monkeypatch.setattr(
        main_module, "time", SimpleNamespace(monotonic=lambda: next(readings) * 1.0)
    )

    await main_module.sweep_artifact_retention(store, settings, pause_seconds=0, batch_seconds=0.5)

    first = _first_pass(calls)
    reading = [summary for summary in first if summary.examined_count]
    _clock_came_off_only_after_idle_batches(caplog, calls)
    assert all(summary.examined_count == 1 for summary in reading)
    assert sum(summary.examined_count for summary in first) == stored


@pytest.mark.parametrize("how", ["the sweep, one deletion a batch", "whole passes"])
async def test_the_sweep_ends_where_whole_passes_end(settings: Settings, how: str) -> None:
    """However a pass is cut into batches, the store ends the same way.

    Posters older and younger than the videos naming them, artifacts created at
    the same moment, kept artifacts to mark, a favorite and a young preview: the
    sweep at one deletion a batch leaves exactly what repeated whole passes leave,
    and marks the same artifacts.
    """

    store = ArtifactStore(settings)
    ids: dict[str, str] = {}
    with SessionLocal() as session:
        older_poster = _artifact(store, session, "older poster", created_at=OLD, temporary=True)
        ids["older poster"] = older_poster.id
        ids["video naming an older poster"] = _artifact(
            store,
            session,
            "video naming an older poster",
            created_at=OLD + timedelta(minutes=1),
            temporary=True,
            kind=ArtifactKind.VIDEO,
            poster_artifact_id=older_poster.id,
        ).id
        younger_poster = _artifact(
            store, session, "younger poster", created_at=OLD + timedelta(minutes=3), temporary=True
        )
        ids["younger poster"] = younger_poster.id
        ids["video naming a younger poster"] = _artifact(
            store,
            session,
            "video naming a younger poster",
            created_at=OLD + timedelta(minutes=2),
            temporary=True,
            kind=ArtifactKind.VIDEO,
            poster_artifact_id=younger_poster.id,
        ).id
        for index in range(4):
            ids[f"tied {index}"] = _artifact(
                store,
                session,
                f"tied {index}",
                created_at=OLD + timedelta(minutes=5),
                temporary=True,
            ).id
        for index in range(3):
            ids[f"kept {index}"] = _artifact(
                store,
                session,
                f"kept {index}",
                created_at=OLD + timedelta(minutes=6 + index),
                temporary=False,
            ).id
        favorite = _artifact(
            store, session, "favorite", created_at=OLD + timedelta(minutes=9), temporary=True
        )
        favorite.favorite = True
        ids["favorite"] = favorite.id
        ids["young"] = _artifact(
            store, session, "young", created_at=datetime.now(), temporary=True
        ).id
        session.commit()

    if how == "whole passes":
        for _attempt in range(10):
            with SessionLocal() as session:
                summary = store.cleanup_retention(
                    session,
                    windows_from=lambda held: windows_for(held, settings),
                    dry_run=False,
                )
                session.commit()
            if summary.removed_count == 0:
                break
    else:
        await main_module.sweep_artifact_retention(
            store, settings, batch_deletions=1, pause_seconds=0, batch_seconds=3600
        )

    with SessionLocal() as check:
        rows = {label: check.get(Artifact, artifact_id) for label, artifact_id in ids.items()}
        survivors = {label for label, row in rows.items() if row is not None}
        marked = {
            label
            for label, row in rows.items()
            if row is not None and "unreferenced_at" in (row.metadata_json or {})
        }
    assert survivors == {"kept 0", "kept 1", "kept 2", "favorite", "young"}
    assert marked == {"kept 0", "kept 1", "kept 2"}


def test_a_preview_takes_no_cursor(settings: Settings) -> None:
    store = ArtifactStore(settings)
    with SessionLocal() as session, pytest.raises(TypeError):
        store.cleanup_retention(
            session,
            retention_days=30,
            temporary_hours=24,
            dry_run=True,
            cursor=RetentionCursor(),
        )


def _aged_leftovers(store: ArtifactStore, how_many: int) -> None:
    """Staging files left in the store's root, old enough for the walk to remove."""

    aged = (datetime.now() - timedelta(days=30)).timestamp()
    for index in range(how_many):
        leftover = store.root / f"ingest-{index:016x}.tmp"
        leftover.write_bytes(b"left over")
        os.utime(leftover, (aged, aged))


async def test_a_batch_that_only_walks_the_files_stops_on_its_clock(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging leftovers in the root, far more than a batch's clock allows it to remove.

    A batch with only the walk left started no clock, so it removed up to its
    whole deletion count however long that took, all of it under the writer.
    """

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(3):
            _artifact(
                store,
                session,
                f"kept {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=False,
            )
        session.commit()
    _aged_leftovers(store, 300)
    calls = _recording(monkeypatch)
    readings = count()
    monkeypatch.setattr(
        main_module, "time", SimpleNamespace(monotonic=lambda: next(readings) * 0.1)
    )

    await asyncio.wait_for(
        main_module.sweep_artifact_retention(store, settings, pause_seconds=0, batch_seconds=1.0),
        timeout=30,
    )

    walking = [summary for cursor, summary in calls if cursor is not None and cursor.rows_done]
    assert walking, "no batch had only the walk left"
    assert max(summary.removed_orphan_file_count for summary in walking) <= (
        artifacts_module.RETENTION_UNTIMED_BATCH_DELETIONS
    )
    assert not list(store.root.glob("ingest-*.tmp"))


def _between_batches(
    monkeypatch: pytest.MonkeyPatch, before: dict[int, Any]
) -> list[RetentionCleanupSummary]:
    """Commit `before[n]` in its own session just before the sweep's batch n starts."""

    summaries: list[RetentionCleanupSummary] = []
    real = ArtifactStore.cleanup_retention

    def wrapped(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        action = before.get(len(summaries) + 1)
        if action is not None:
            with SessionLocal() as other:
                action(other)
                other.commit()
        summary = real(self, session, **kwargs)
        summaries.append(summary)
        return summary

    monkeypatch.setattr(ArtifactStore, "cleanup_retention", wrapped)
    # Every walk ends after one shard, so a pass takes several batches.
    monkeypatch.setattr(artifacts_module._BoundedWalk, "over", lambda walk: walk.leaves > 0)
    return summaries


def _kept_after(store: ArtifactStore, session: Session, how_many: int) -> None:
    for index in range(how_many):
        _artifact(
            store,
            session,
            f"kept {index}",
            created_at=OLD + timedelta(minutes=1 + index),
            temporary=False,
        )


async def test_a_preview_released_behind_the_pass_goes_in_the_same_sweep(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job holds an expired preview when its batch reads it, and is deleted before the next.

    Reading every batch from the oldest saw the release in the same sweep; a
    pass that goes on from its place has to read the store once more for it.
    """

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        preview = _artifact(store, session, "held preview", created_at=OLD, temporary=True)
        job = Job(payload_json={"artifact_id": preview.id}, result_json={}, status="complete")
        session.add(job)
        _kept_after(store, session, 12)
        session.commit()
        preview_id, job_id = preview.id, job.id

    summaries = _between_batches(
        monkeypatch, {2: lambda other: other.execute(delete(Job).where(Job.id == job_id))}
    )
    await main_module.sweep_artifact_retention(store, settings, pause_seconds=0, batch_seconds=3600)

    assert len(summaries) > 2, "the pass took a single batch"
    with SessionLocal() as check:
        assert check.get(Artifact, preview_id) is None


async def test_a_reference_added_behind_the_pass_clears_its_mark_in_the_same_sweep(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marked artifact is referenced after its batch went by.

    Left in place, the old mark would let the next start delete the artifact on
    it once the reference goes, up to a whole window earlier than it should.
    """

    store = ArtifactStore(settings)
    marked = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    with SessionLocal() as session:
        target = _artifact(
            store, session, "pending", created_at=OLD, temporary=False, unreferenced_at=marked
        )
        _kept_after(store, session, 12)
        session.commit()
        target_id = target.id

    def reference(other: Session) -> None:
        other.add(Job(payload_json={"artifact_id": target_id}, result_json={}, status="complete"))

    _between_batches(monkeypatch, {2: reference})
    await main_module.sweep_artifact_retention(store, settings, pause_seconds=0, batch_seconds=3600)

    with SessionLocal() as check:
        artifact = check.get(Artifact, target_id)
        assert artifact is not None
        assert "unreferenced_at" not in (artifact.metadata_json or {})


@pytest.mark.parametrize("with_cursor", [False, True])
def test_a_cleanup_reads_past_its_first_chunk_of_rows(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, with_cursor: bool
) -> None:
    monkeypatch.setattr(artifacts_module, "_RETENTION_READ_CHUNK", 3)
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(10):
            _artifact(
                store,
                session,
                f"kept {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=False,
            )
        for index in range(10):
            _artifact(
                store,
                session,
                f"expired preview {index}",
                created_at=OLD + timedelta(days=1, minutes=index),
                temporary=True,
            )
        session.commit()
        summary = store.cleanup_retention(
            session,
            retention_days=30,
            temporary_hours=24,
            dry_run=False,
            cursor=RetentionCursor() if with_cursor else None,
        )
        session.commit()

    assert summary.examined_count == 20
    assert summary.removed_row_count == 10
    assert not summary.truncated


async def test_a_batch_stops_among_favorites_and_the_sweep_goes_on_past_them(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forty favorite expired previews, more than a batch's clock reads.

    A batch that stopped among artifacts it keeps has to record where, or every
    batch after it reads the same ones again and the sweep never ends.
    """

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(40):
            artifact = _artifact(
                store,
                session,
                f"favorite {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=True,
            )
            artifact.favorite = True
        session.commit()
        stored = _count(session)
    calls = _recording(monkeypatch)
    readings = count()
    monkeypatch.setattr(
        main_module, "time", SimpleNamespace(monotonic=lambda: next(readings) * 0.1)
    )

    await asyncio.wait_for(
        main_module.sweep_artifact_retention(store, settings, pause_seconds=0, batch_seconds=1.0),
        timeout=30,
    )

    first = _first_pass(calls)
    assert len([summary for summary in first if summary.examined_count]) > 1
    assert sum(summary.examined_count for summary in first) == stored
    with SessionLocal() as check:
        assert _count(check) == stored


def test_a_cursor_batch_asks_its_snapshot_about_each_artifact_instead_of_reading_it_whole(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading every retained artifact and naming in the store took seconds under the writer.

    Favorites, so the batch has no marks to write: the flush that writes marks
    asks the snapshot for its namings itself, which is not what this pins.
    """

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(5):
            artifact = _artifact(
                store,
                session,
                f"favorite {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=False,
            )
            artifact.favorite = True
        session.commit()
    # The suite's own check of every snapshot against the reference walk reads a
    # second snapshot whole; this counts only what the batch asks of its own.
    monkeypatch.setattr(reference_index_module, "CHECK_AGAINST_WALK", False)
    whole: list[str] = []
    for name in ("retained", "namings"):
        real = getattr(IndexedReferenceSnapshot, name)

        def counting(
            self: IndexedReferenceSnapshot, *, _real: Any = real, _name: str = name
        ) -> Any:
            whole.append(_name)
            return _real(self)

        monkeypatch.setattr(IndexedReferenceSnapshot, name, counting)

    with SessionLocal() as session:
        summary = store.cleanup_retention(
            session,
            retention_days=30,
            temporary_hours=24,
            dry_run=False,
            cursor=RetentionCursor(),
        )
        session.commit()

    assert summary.examined_count == 5
    assert whole == []


def test_clearing_old_marks_counts_toward_a_batchs_rewrite_limit(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(artifacts_module, "RETENTION_BATCH_METADATA_CHANGES", 3)
    marked = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(10):
            artifact = _artifact(
                store,
                session,
                f"favorite {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=False,
                unreferenced_at=marked,
            )
            artifact.favorite = True
        session.commit()
        summary = store.cleanup_retention(
            session,
            retention_days=30,
            temporary_hours=24,
            dry_run=False,
            cursor=RetentionCursor(),
        )
        session.commit()
        cleared = sum(
            1
            for artifact in session.scalars(select(Artifact))
            if "unreferenced_at" not in (artifact.metadata_json or {})
        )

    assert summary.truncated
    assert cleared == 3


def test_a_persons_cleanup_is_not_held_to_a_batchs_rewrite_limit(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(artifacts_module, "RETENTION_BATCH_METADATA_CHANGES", 3)
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(10):
            _artifact(
                store,
                session,
                f"kept {index}",
                created_at=OLD + timedelta(minutes=index),
                temporary=False,
            )
        session.commit()
        summary = store.cleanup_retention(
            session, retention_days=30, temporary_hours=24, dry_run=False
        )
        session.commit()

    assert summary.marked_count == 10
    assert not summary.truncated


async def test_a_store_with_no_rows_whose_walk_is_cut_each_batch_keeps_its_clock(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Young unindexed files in several shards, and a walk that ends after one each batch.

    Each batch finished the rows, which there are none of, and walked a shard,
    so each moved the pass on and none may take the clock off.
    """

    caplog.set_level(logging.INFO)
    store = ArtifactStore(settings)
    for index in range(4):
        content = f"young unindexed {index}".encode()
        digest = hashlib.sha256(content).hexdigest()
        young = store.root / digest[:2] / digest[2:4] / digest
        young.parent.mkdir(parents=True, exist_ok=True)
        young.write_bytes(content)
    calls = _recording(monkeypatch)
    monkeypatch.setattr(artifacts_module._BoundedWalk, "over", lambda walk: walk.leaves > 0)

    await main_module.sweep_artifact_retention(store, settings, pause_seconds=0, batch_seconds=3600)

    assert len(calls) > 1, "the walk took a single batch"
    assert "continuing without the clock" not in caplog.text


async def test_a_batch_that_finishes_the_rows_keeps_its_clock_though_its_walk_finished_no_shard(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No rows, and more shards to list than the first batch's clock allows.

    The first batch finished the rows, which there are none of, so it moved the
    pass on even though its clock ran out while it listed the shards. Only a
    later batch that did nothing at all may take the clock off.
    """

    caplog.set_level(logging.INFO)
    store = ArtifactStore(settings)
    for index in range(30):
        content = f"young unindexed {index}".encode()
        digest = hashlib.sha256(content).hexdigest()
        young = store.root / digest[:2] / digest[2:4] / digest
        young.parent.mkdir(parents=True, exist_ok=True)
        young.write_bytes(content)
    calls = _recording(monkeypatch)
    readings = count()
    monkeypatch.setattr(
        main_module, "time", SimpleNamespace(monotonic=lambda: next(readings) * 0.1)
    )

    await asyncio.wait_for(
        main_module.sweep_artifact_retention(store, settings, pause_seconds=0, batch_seconds=1.0),
        timeout=30,
    )

    first = calls[0][1]
    assert first.truncated and first.examined_count == 0 and first.walked_shards == 0
    assert first.cursor is not None and first.cursor.rows_done
    assert "batch 1 removed nothing" not in caplog.text
