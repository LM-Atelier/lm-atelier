"""Writes that meet a held or stale database: what they keep, and what they never show.

SQLite has one writer. A write that finds it held waits out the busy timeout
and fails; a write on a connection that still holds an old read fails at once.
Neither may end work that is only waiting, replace an outcome, or put the
values a statement carried into the error's text.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from run_waits import PATIENCE_SECONDS
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from test_scheduler import _claim, _one_queued_job, _real_sqlite_error

from local_lm import db as database_module
from local_lm.config import Settings
from local_lm.db import SessionLocal, init_db
from local_lm.domain import JobStatus, utcnow
from local_lm.models import Job
from local_lm.scheduler import ResourceScheduler


def _database(settings: Settings) -> Path:
    return settings.state_dir / "local-lm.sqlite3"


def test_a_result_left_open_does_not_hand_its_read_to_the_next_writer(
    settings: Settings,
) -> None:
    """A connection goes back to the pool with nothing still reading on it.

    The result below outlives its session, as a stream read partway would.
    Without the pool closing it, the connection carries the result's read to
    the next session that draws it, and that session's write fails at once
    with SQLITE_BUSY_SNAPSHOT as soon as anyone else has committed since.
    """

    _one_queued_job(settings)
    database_module.engine.dispose()
    with Session(database_module.engine) as reader:
        kept = reader.connection().exec_driver_sql("SELECT name FROM sqlite_master")
        assert kept.fetchone() is not None
    # A row has to change: a write that changes nothing leaves the old read current.
    other = sqlite3.connect(_database(settings), timeout=5)
    try:
        other.execute("UPDATE jobs SET queue_ticket = 'ticket-changed' WHERE id = 'job_waiting'")
        other.commit()
    finally:
        other.close()
    try:
        with Session(database_module.engine) as writer:
            writer.connection().exec_driver_sql("UPDATE jobs SET updated_at = updated_at WHERE 0")
            writer.commit()
    finally:
        del kept


def test_a_database_error_names_its_statement_but_not_its_values(settings: Settings) -> None:
    init_db()
    with SessionLocal() as session, pytest.raises(OperationalError) as raised:
        session.execute(
            text("SELECT * FROM no_such_table WHERE label = :label"),
            {"label": "neutral-marker-4417"},
        )

    shown = str(raised.value)
    assert "no_such_table" in shown
    assert "neutral-marker-4417" not in shown


async def test_a_held_database_while_old_claims_are_cleared_does_not_fail_a_waiting_job(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing an old claim is a write; a held database only delays the next pass."""

    _one_queued_job(settings)
    scheduler = ResourceScheduler(session_factory=SessionLocal)
    original = scheduler._expire_foreign_claims
    refusals: list[int] = []

    def held_once(group: str) -> Any:
        if not refusals:
            refusals.append(1)
            raise _real_sqlite_error(tmp_path / "busy", contended=True)
        return original(group)

    monkeypatch.setattr(scheduler, "_expire_foreign_claims", held_once)

    claim = await _claim(scheduler)

    assert refusals == [1], "the held pass has to have happened"
    assert claim.token
    with SessionLocal() as session:
        job = session.get(Job, "job_waiting")
        assert job is not None and job.status == JobStatus.RUNNING.value


async def test_a_release_a_held_database_refuses_keeps_the_outcome_and_frees_the_slot(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _one_queued_job(settings)
    scheduler = ResourceScheduler(session_factory=SessionLocal)
    contended = _real_sqlite_error(tmp_path / "busy", contended=True)

    async def held_release(job_id: str, token: str, group: str) -> None:
        raise contended

    monkeypatch.setattr(scheduler, "_release_job", held_release)

    async with scheduler.job_lease("job_waiting", resource="primary", group="primary") as claim:
        pass

    assert claim.token in scheduler._abandoned_claims
    lock = scheduler._lock("primary", 1)
    await asyncio.wait_for(lock.acquire(), timeout=PATIENCE_SECONDS)
    lock.release()

    # A failure inside the lease is still the failure the caller sees. Bounded,
    # so a claim the release left behind fails here instead of hanging.
    _one_queued_job_again(settings)

    async def failing_work() -> None:
        async with scheduler.job_lease("job_again", resource="primary", group="primary"):
            raise LookupError("the work itself failed")

    with pytest.raises(LookupError):
        await asyncio.wait_for(failing_work(), timeout=PATIENCE_SECONDS)


async def test_a_release_that_fails_for_another_reason_still_raises(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a held database is waited out; a broken one is reported."""

    _one_queued_job(settings)
    scheduler = ResourceScheduler(session_factory=SessionLocal)
    broken = _real_sqlite_error(tmp_path / "broken", contended=False)

    async def broken_release(job_id: str, token: str, group: str) -> None:
        raise broken

    monkeypatch.setattr(scheduler, "_release_job", broken_release)

    async def work() -> None:
        async with scheduler.job_lease("job_waiting", resource="primary", group="primary"):
            pass

    with pytest.raises(OperationalError):
        await asyncio.wait_for(work(), timeout=PATIENCE_SECONDS)


async def test_a_paused_job_left_with_a_claim_by_a_held_release_is_cleared(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job paused while its lease ended keeps a claim nothing else clears.

    Its group counted it as running, so the next job never started until
    someone resumed or cancelled the paused one.
    """

    _one_queued_job(settings)
    scheduler = ResourceScheduler(session_factory=SessionLocal)
    contended = _real_sqlite_error(tmp_path / "busy", contended=True)
    original_release = scheduler._release_job

    async def pause_and_hold(job_id: str, token: str, group: str) -> None:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None
            job.status = JobStatus.PAUSED.value
            session.commit()
        raise contended

    monkeypatch.setattr(scheduler, "_release_job", pause_and_hold)
    async with scheduler.job_lease("job_waiting", resource="primary", group="primary"):
        pass
    monkeypatch.setattr(scheduler, "_release_job", original_release)
    _one_queued_job_again(settings)

    async def next_job() -> None:
        async with scheduler.job_lease("job_again", resource="primary", group="primary"):
            pass

    await asyncio.wait_for(next_job(), timeout=PATIENCE_SECONDS)
    with SessionLocal() as session:
        paused = session.get(Job, "job_waiting")
        assert paused is not None
        assert paused.status == JobStatus.PAUSED.value
        assert paused.claim_owner is None


def _one_queued_job_again(settings: Settings) -> None:
    with SessionLocal() as session:
        session.add(
            Job(
                id="job_again",
                kind="image",
                status=JobStatus.QUEUED.value,
                queue_group="primary",
                queue_ticket="ticket-b",
                enqueued_at=utcnow(),
            )
        )
        session.commit()


def test_an_invalidated_connection_reports_no_cursor_it_could_not_close(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """An invalidated connection is closed before it goes back and is never handed out.

    Closing its cursors then can only fail, and the warning would point anyone
    reading the log after a failed request at a leak that never happened.
    """

    database_module.engine.dispose()
    with database_module.engine.connect() as connection:
        kept = connection.exec_driver_sql("SELECT name FROM sqlite_master")
        assert kept.fetchone() is not None
        with caplog.at_level(logging.WARNING, logger="local_lm.db"):
            connection.invalidate()
    del kept

    assert not [record for record in caplog.records if record.name == "local_lm.db"]
