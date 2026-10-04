"""Heartbeat write failures preserve claim renewal and execution outcomes."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from unittest.mock import DEFAULT, Mock

import pytest
from httpx2 import AsyncClient
from sqlalchemy import Update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from test_scheduler_claim_hold import _change, _queued, _until

from local_lm import scheduler as scheduler_module
from local_lm.db import SessionLocal
from local_lm.models import Job
from local_lm.scheduler import ResourceScheduler


@pytest.mark.parametrize("write", ["execute", "commit"])
async def test_a_contended_heartbeat_retries_and_preserves_the_current_claim(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, write: str
) -> None:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.01)
    _queued("contended-heartbeat")
    before = datetime(2026, 1, 1, tzinfo=UTC)
    _change("contended-heartbeat", status="running", claim_owner="current", heartbeat_at=before)
    writes = 0

    def fail_first_write(statement: object = None, *args: object, **kwargs: object) -> object:
        nonlocal writes
        if write == "commit" or isinstance(statement, Update):
            writes += 1
            if writes == 1:
                contention = sqlite3.OperationalError("database is locked")
                contention.sqlite_errorcode = sqlite3.SQLITE_BUSY
                raise OperationalError("renew claim", {}, contention)
        return DEFAULT

    def sessions() -> Session:
        session = SessionLocal()
        monkeypatch.setattr(
            session, write, Mock(wraps=getattr(session, write), side_effect=fail_first_write)
        )
        return session

    scheduler = ResourceScheduler(session_factory=sessions)
    heartbeat = asyncio.create_task(scheduler._heartbeat("contended-heartbeat", "current"))
    try:
        await _until(lambda: writes >= 2 or heartbeat.done())
        assert not heartbeat.done()
        with SessionLocal() as session:
            job = session.get(Job, "contended-heartbeat")
            assert job is not None and job.claim_owner == "current"
            assert job.heartbeat_at is not None and job.heartbeat_at.replace(tzinfo=UTC) > before
            assert job.claim_expires_at is not None
        _change("contended-heartbeat", claim_owner="replacement")
        await asyncio.wait_for(heartbeat, timeout=2)
        with SessionLocal() as session:
            job = session.get(Job, "contended-heartbeat")
            assert job is not None and job.claim_owner == "replacement"
    finally:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)


@pytest.mark.parametrize("outcome", ["error", "cancelled", "success"])
async def test_lease_cleanup_preserves_the_execution_outcome_after_heartbeat_failure(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    outcome: str,
) -> None:
    _queued("heartbeat-cleanup")
    scheduler = ResourceScheduler()
    failed = asyncio.Event()
    heartbeat_error = RuntimeError("The heartbeat could not renew its claim.")
    execution_error = ValueError("The execution could not finish.")

    async def fail_heartbeat(job_id: str, token: str) -> None:
        failed.set()
        raise heartbeat_error

    monkeypatch.setattr(scheduler, "_heartbeat", fail_heartbeat)

    async def execute() -> None:
        async with scheduler.job_lease(
            "heartbeat-cleanup", resource="media_compute", group="primary"
        ):
            await failed.wait()
            if outcome == "error":
                raise execution_error
            if outcome == "cancelled":
                await asyncio.Event().wait()

    task = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(failed.wait(), timeout=2)
        if outcome == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "error":
            with pytest.raises(ValueError) as caught:
                await task
            assert caught.value is execution_error
        else:
            with pytest.raises(RuntimeError) as caught_heartbeat:
                await task
            assert caught_heartbeat.value is heartbeat_error
        assert not scheduler._lock("primary", 1).locked()
        with SessionLocal() as session:
            job = session.get(Job, "heartbeat-cleanup")
            assert job is not None and job.claim_owner is None
        if outcome != "success":
            assert (
                "A scheduler heartbeat failed while the execution was already ending."
                in caplog.messages
            )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_an_unexpected_heartbeat_database_failure_still_propagates(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.01)
    _queued("unexpected-heartbeat")
    _change("unexpected-heartbeat", status="running", claim_owner="current")
    failure = OperationalError(
        "renew claim", {}, sqlite3.OperationalError("no such table: missing")
    )

    def sessions() -> Session:
        session = SessionLocal()
        monkeypatch.setattr(session, "execute", Mock(side_effect=failure))
        return session

    scheduler = ResourceScheduler(session_factory=sessions)
    with pytest.raises(OperationalError) as caught:
        await asyncio.wait_for(scheduler._heartbeat("unexpected-heartbeat", "current"), timeout=2)
    assert caught.value is failure
