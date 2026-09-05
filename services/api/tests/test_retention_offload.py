"""The retention sweep runs after the port opens, in committed batches.

A sweep over thousands of aged previews pays the reference trigger's scan per
deletion and takes hours. Run before the lifespan yield in one session, it kept
the port closed for that long and lost every deletion to a kill. These tests
pin the two halves of the repair: the sweep's own deletion bound and stop flag,
and a lifespan that serves while batches commit one at a time and stops a batch
at its boundary on shutdown.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from local_lm import main as main_module
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base, SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.main import create_app
from local_lm.models import Artifact


@pytest.fixture
def sweepable(tmp_path: Path) -> tuple[ArtifactStore, Session]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_engine(f"sqlite:///{settings.data_dir / 'retention.sqlite3'}")
    Base.metadata.create_all(engine)
    store = ArtifactStore(settings)
    with Session(engine) as session:
        yield store, session


def _aged_temporary(store: ArtifactStore, session: Session, index: int) -> Artifact:
    """A temporary preview two days old, unreferenced, and so eligible."""

    artifact = store.ingest_bytes(
        session,
        f"preview {index}".encode(),
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        metadata={"temporary_preview": True},
    )
    artifact.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=48)
    session.flush()
    return artifact


def _count(session: Session) -> int:
    return int(session.scalar(select(func.count(Artifact.id))) or 0)


def test_a_deletion_bound_truncates_the_pass_and_leaves_the_rest_for_the_next(
    sweepable: tuple[ArtifactStore, Session],
) -> None:
    store, session = sweepable
    for index in range(5):
        _aged_temporary(store, session, index)
    session.commit()

    first = store.cleanup_retention(
        session, retention_days=30, temporary_hours=24, dry_run=False, max_deletions=2
    )
    session.commit()
    assert first.removed_count == 2
    assert first.truncated is True
    assert _count(session) == 3

    second = store.cleanup_retention(session, retention_days=30, temporary_hours=24, dry_run=False)
    session.commit()
    assert second.removed_count == 3
    assert second.truncated is False
    assert _count(session) == 0


def test_a_stop_request_ends_the_pass_at_the_next_deletion_boundary(
    sweepable: tuple[ArtifactStore, Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag is read against state, not a call schedule: stop once one row is gone."""

    store, session = sweepable
    for index in range(3):
        _aged_temporary(store, session, index)
    session.commit()
    deleted = 0
    real_delete = ArtifactStore._delete_artifact

    def counting_delete(self: ArtifactStore, *args: Any, **kwargs: Any) -> Any:
        nonlocal deleted
        deleted += 1
        return real_delete(self, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "_delete_artifact", counting_delete)

    summary = store.cleanup_retention(
        session,
        retention_days=30,
        temporary_hours=24,
        dry_run=False,
        should_stop=lambda: deleted >= 1,
    )
    session.commit()
    assert summary.removed_count == 1
    assert summary.truncated is True
    assert _count(session) == 2


def test_a_truncated_pass_leaves_orphan_files_to_the_pass_that_completes(
    sweepable: tuple[ArtifactStore, Session],
) -> None:
    """The orphan sweep describes the whole store, so a partial pass skips it."""

    store, session = sweepable
    for index in range(2):
        _aged_temporary(store, session, index)
    session.commit()
    content = b"orphaned after a failed database commit"
    digest = hashlib.sha256(content).hexdigest()
    orphan = store.root / digest[:2] / digest[2:4] / digest
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(content)
    old = datetime.now(UTC) - timedelta(hours=25)
    os.utime(orphan, (old.timestamp(), old.timestamp()))

    partial = store.cleanup_retention(
        session, retention_days=30, temporary_hours=24, dry_run=False, max_deletions=1
    )
    session.commit()
    assert partial.truncated is True
    assert orphan.exists()

    complete = store.cleanup_retention(
        session, retention_days=30, temporary_hours=24, dry_run=False
    )
    session.commit()
    assert complete.truncated is False
    assert not orphan.exists()


async def test_the_app_serves_while_the_sweep_commits_one_batch_at_a_time(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The port is open before the sweep completes, and every batch commits.

    The first batch is held at its entry until the application has answered a
    request, which the pre-yield stage could never allow. Each batch records
    how many artifacts a fresh session can see when it starts; a sequence that
    shrinks batch by batch proves the commits are per batch rather than one at
    the end.
    """

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(5):
            _aged_temporary(store, session, index)
        session.commit()
    monkeypatch.setattr(main_module, "RETENTION_BATCH_DELETIONS", 2)
    monkeypatch.setattr(main_module, "RETENTION_BATCH_PAUSE_SECONDS", 0.0)

    stages: list[str] = []
    real_stage = main_module._startup_stage

    def recording_stage(name: str, **kwargs: Any) -> Any:
        stages.append(name)
        return real_stage(name, **kwargs)

    monkeypatch.setattr(main_module, "_startup_stage", recording_stage)

    release = threading.Event()
    visible_before: list[int] = []
    real_cleanup = ArtifactStore.cleanup_retention

    def spying_cleanup(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        with SessionLocal() as other:
            visible_before.append(_count(other))
        if len(visible_before) == 1:
            assert release.wait(timeout=10), "the first batch was never released"
        return real_cleanup(self, session, **kwargs)

    monkeypatch.setattr(ArtifactStore, "cleanup_retention", spying_cleanup)

    app = create_app(settings)
    remaining = -1
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            for _ in range(500):
                if visible_before:
                    break
                await asyncio.sleep(0.01)
            assert visible_before == [5], "the first batch did not start once the app was up"
            response = await client.get("/api/health")
            assert response.status_code == 200
            release.set()
            # Await the sweep rather than polling the store: leaving the
            # lifespan the moment the last row is gone would cancel the final
            # pass, and that pass is part of what is being pinned.
            await asyncio.wait_for(app.state.retention_sweep, timeout=30)
            with SessionLocal() as check:
                remaining = _count(check)
    assert remaining == 0
    # Three truncated batches, then one completed pass that finds nothing:
    # that final pass is what ends the sweep.
    assert visible_before == [5, 3, 1, 0], "each batch must commit before the next begins"
    assert "artifact-retention-cleanup" not in stages
    assert "session-commit" in stages


async def test_shutdown_stops_a_batch_at_its_next_boundary(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Leaving the lifespan mid-batch asks the batch to stop and waits for it.

    The batch stands in for a long one: it holds until the stop request
    arrives, then runs the real sweep, which sees the request before its first
    deletion and returns with nothing removed. Shutdown must complete promptly
    and the store must be exactly as it was.
    """

    caplog.set_level(logging.INFO)
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(4):
            _aged_temporary(store, session, index)
        session.commit()
    monkeypatch.setattr(main_module, "RETENTION_BATCH_DELETIONS", 1)
    monkeypatch.setattr(main_module, "RETENTION_BATCH_PAUSE_SECONDS", 0.0)

    entered = threading.Event()
    real_cleanup = ArtifactStore.cleanup_retention

    def stalling_cleanup(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        entered.set()
        should_stop = kwargs["should_stop"]
        for _ in range(3000):
            if should_stop():
                break
            time.sleep(0.01)
        return real_cleanup(self, session, **kwargs)

    monkeypatch.setattr(ArtifactStore, "cleanup_retention", stalling_cleanup)

    app = create_app(settings)
    started = time.monotonic()
    async with app.router.lifespan_context(app):
        for _ in range(500):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set(), "the sweep did not start once the app was up"
    assert time.monotonic() - started < 20
    with SessionLocal() as check:
        assert _count(check) == 4
    assert "resumes at the next start" in caplog.text


async def test_a_failing_sweep_leaves_the_application_running(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A sweep that raises is logged and dropped; it must never take the app down."""

    caplog.set_level(logging.ERROR)

    def failing_cleanup(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic sweep failure")

    monkeypatch.setattr(ArtifactStore, "cleanup_retention", failing_cleanup)

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await app.state.retention_sweep
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/health")
            assert response.status_code == 200
    assert app.state.retention_sweep.done()
    assert app.state.retention_sweep.exception() is None
    assert "synthetic sweep failure" in caplog.text
    assert "runs again at the next start" in caplog.text


async def test_a_batch_is_bounded_by_time_held_not_by_count(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow deletion ends the batch at the budget, well under busy_timeout.

    Each deletion is made to take 0.2 s against a 0.3 s budget, so a batch can
    hold the writer for at most two deletions even though the count bound
    would allow ten. Five previews therefore take at least three batches, each
    committed before the next begins.
    """

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(5):
            _aged_temporary(store, session, index)
        session.commit()
    monkeypatch.setattr(main_module, "RETENTION_BATCH_DELETIONS", 10)
    monkeypatch.setattr(main_module, "RETENTION_BATCH_SECONDS", 0.3)
    monkeypatch.setattr(main_module, "RETENTION_BATCH_PAUSE_SECONDS", 0.0)

    real_delete = ArtifactStore._delete_artifact

    def slow_delete(self: ArtifactStore, *args: Any, **kwargs: Any) -> Any:
        time.sleep(0.2)
        return real_delete(self, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "_delete_artifact", slow_delete)

    visible_before: list[int] = []
    real_cleanup = ArtifactStore.cleanup_retention

    def spying_cleanup(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        with SessionLocal() as other:
            visible_before.append(_count(other))
        return real_cleanup(self, session, **kwargs)

    monkeypatch.setattr(ArtifactStore, "cleanup_retention", spying_cleanup)

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
    with SessionLocal() as check:
        assert _count(check) == 0
    assert len(visible_before) >= 3, visible_before
    steps = [a - b for a, b in zip(visible_before, visible_before[1:] + [0], strict=True)]
    assert all(step <= 2 for step in steps), steps


async def test_a_budget_smaller_than_the_fixed_work_still_makes_progress(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A zero budget cannot delete anything; the sweep must not spin on it.

    After one batch that removes nothing within its budget, the sweep switches
    to one deletion per batch and still drains the store.
    """

    caplog.set_level(logging.INFO)
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(3):
            _aged_temporary(store, session, index)
        session.commit()
    monkeypatch.setattr(main_module, "RETENTION_BATCH_SECONDS", 0.0)
    monkeypatch.setattr(main_module, "RETENTION_BATCH_PAUSE_SECONDS", 0.0)

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
    with SessionLocal() as check:
        assert _count(check) == 0
    assert "continuing one deletion per batch" in caplog.text
    assert "sweep complete" in caplog.text


async def test_a_batch_that_loses_the_writer_is_retried_not_abandoned(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Contention is waited out; only corruption or a real failure ends the sweep."""

    caplog.set_level(logging.WARNING)
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(2):
            _aged_temporary(store, session, index)
        session.commit()
    monkeypatch.setattr(main_module, "RETENTION_BATCH_PAUSE_SECONDS", 0.0)

    calls = 0
    real_cleanup = ArtifactStore.cleanup_retention

    def contended_cleanup(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OperationalError("BEGIN IMMEDIATE", {}, Exception("database is locked"))
        return real_cleanup(self, session, **kwargs)

    monkeypatch.setattr(ArtifactStore, "cleanup_retention", contended_cleanup)

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
    with SessionLocal() as check:
        assert _count(check) == 0
    assert calls >= 2
    assert "waited on the database writer; retrying" in caplog.text


async def test_a_completed_pass_that_removed_something_is_followed_by_another(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep ends on a pass that finds nothing, not on the first full pass.

    A poster is freed only by the pass that removes the artifact naming it, so
    one pass per start used to leave such chains one link behind. Three
    previews fit one batch; the sweep still runs a second pass to prove the
    store is clean.
    """

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(3):
            _aged_temporary(store, session, index)
        session.commit()
    monkeypatch.setattr(main_module, "RETENTION_BATCH_PAUSE_SECONDS", 0.0)

    removed_per_call: list[int] = []
    real_cleanup = ArtifactStore.cleanup_retention

    def recording_cleanup(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        summary = real_cleanup(self, session, **kwargs)
        removed_per_call.append(summary.removed_count)
        return summary

    monkeypatch.setattr(ArtifactStore, "cleanup_retention", recording_cleanup)

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
    assert removed_per_call == [3, 0]


def _aged_orphan(store: ArtifactStore, index: int) -> Path:
    """An unindexed canonical file, 25 hours old, in its shard directory."""

    content = f"orphan {index}".encode()
    digest = hashlib.sha256(content).hexdigest()
    orphan = store.root / digest[:2] / digest[2:4] / digest
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(content)
    old = datetime.now(UTC) - timedelta(hours=25)
    os.utime(orphan, (old.timestamp(), old.timestamp()))
    return orphan


def test_an_orphan_dominated_store_is_removed_within_the_same_bound(
    sweepable: tuple[ArtifactStore, Session],
) -> None:
    """The orphan walk spends the pass's budget; it is not a free final batch."""

    store, session = sweepable
    orphans = [_aged_orphan(store, index) for index in range(5)]
    session.commit()

    first = store.cleanup_retention(
        session, retention_days=30, temporary_hours=24, dry_run=False, max_deletions=2
    )
    assert first.removed_count == 2
    assert first.truncated is True
    assert sum(1 for path in orphans if path.exists()) == 3

    second = store.cleanup_retention(
        session, retention_days=30, temporary_hours=24, dry_run=False, max_deletions=2
    )
    assert second.removed_count == 2
    assert second.truncated is True

    third = store.cleanup_retention(
        session, retention_days=30, temporary_hours=24, dry_run=False, max_deletions=2
    )
    assert third.removed_count == 1
    assert third.truncated is False
    assert not any(path.exists() for path in orphans)


def test_a_stop_at_the_row_to_orphan_boundary_leaves_the_orphans_alone(
    sweepable: tuple[ArtifactStore, Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop that arrives on the last row deletion is honoured before the walk."""

    store, session = sweepable
    _aged_temporary(store, session, 0)
    session.commit()
    orphans = [_aged_orphan(store, index) for index in range(2)]
    deleted = 0
    real_delete = ArtifactStore._delete_artifact

    def counting_delete(self: ArtifactStore, *args: Any, **kwargs: Any) -> Any:
        nonlocal deleted
        deleted += 1
        return real_delete(self, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "_delete_artifact", counting_delete)

    stopped = store.cleanup_retention(
        session,
        retention_days=30,
        temporary_hours=24,
        dry_run=False,
        should_stop=lambda: deleted >= 1,
    )
    session.commit()
    assert stopped.removed_count == 1
    assert stopped.truncated is True
    assert all(path.exists() for path in orphans)
    assert _count(session) == 0

    resumed = store.cleanup_retention(session, retention_days=30, temporary_hours=24, dry_run=False)
    session.commit()
    assert resumed.removed_count == 2
    assert resumed.truncated is False
    assert not any(path.exists() for path in orphans)


async def test_orphan_batches_commit_one_at_a_time_through_the_lifespan(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five aged orphan files and no rows drain over several bounded batches."""

    store = ArtifactStore(settings)
    orphans = [_aged_orphan(store, index) for index in range(5)]
    monkeypatch.setattr(main_module, "RETENTION_BATCH_DELETIONS", 2)
    monkeypatch.setattr(main_module, "RETENTION_BATCH_PAUSE_SECONDS", 0.0)

    left_before: list[int] = []
    real_walk = ArtifactStore._cleanup_orphan_files

    def spying_walk(self: ArtifactStore, *args: Any, **kwargs: Any) -> Any:
        left_before.append(sum(1 for path in orphans if path.exists()))
        return real_walk(self, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "_cleanup_orphan_files", spying_walk)

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
    assert not any(path.exists() for path in orphans)
    assert left_before == [5, 3, 1, 0], left_before


async def test_a_slow_snapshot_keeps_the_default_batch_and_deletion_budget(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(25):
            _aged_temporary(store, session, index)
        session.commit()
    clock = 0.0
    snapshots = 0
    real_references = ArtifactStore.referenced_artifact_ids
    real_cleanup = ArtifactStore.cleanup_retention
    removed: list[int] = []

    def slow_references(session: Session, *, for_deletion: bool = False) -> Any:
        nonlocal clock, snapshots
        clock += 10.0
        snapshots += 1
        return real_references(session, for_deletion=for_deletion)

    def counted_cleanup(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        result = real_cleanup(self, session, **kwargs)
        removed.append(result.removed_count)
        return result

    monkeypatch.setattr(main_module, "time", SimpleNamespace(monotonic=lambda: clock))
    monkeypatch.setattr(ArtifactStore, "referenced_artifact_ids", staticmethod(slow_references))
    monkeypatch.setattr(ArtifactStore, "cleanup_retention", counted_cleanup)
    caplog.set_level(logging.INFO)

    await main_module.sweep_artifact_retention(store, settings, pause_seconds=0)

    assert removed == [25, 0]
    assert snapshots == 2
    assert "continuing one deletion per batch" not in caplog.text
