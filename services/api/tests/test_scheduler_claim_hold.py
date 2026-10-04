"""An attempt keeps the computer awake only while it still owns its claim."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient
from sqlalchemy import delete, update

from local_lm import scheduler as scheduler_module
from local_lm.db import SessionLocal
from local_lm.models import Job
from local_lm.power_inhibition import PowerInhibitor
from local_lm.scheduler import ResourceScheduler


class _Backend:
    name = "test"
    supported = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def acquire(self, reason: str) -> object:
        self.calls.append("acquire")
        return object()

    def release(self, handle: object) -> None:
        self.calls.append("release")


def _queued(job_id: str) -> None:
    with SessionLocal() as session:
        session.add(
            Job(
                id=job_id,
                kind="image",
                status="queued",
                queue_group="primary",
                queue_resource="media_compute",
                enqueued_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        session.commit()


def _change(job_id: str, **values: object) -> None:
    with SessionLocal() as session:
        session.execute(update(Job).where(Job.id == job_id).values(**values))
        session.commit()


async def _until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the condition never held"
        await asyncio.sleep(0.02)


async def test_an_older_attempt_that_ends_leaves_the_newer_attempts_hold(
    client: AsyncClient,
) -> None:
    backend = _Backend()
    power = PowerInhibitor(backend)
    scheduler = ResourceScheduler(power=power)
    _queued("work")
    older = scheduler.job_lease("work", resource="media_compute", group="primary", capacity=2)
    await older.__aenter__()
    ended = False
    try:
        # The job is queued again and unclaimed, as a retry leaves it, and a
        # newer attempt claims it while the older one is still finishing.
        _change("work", status="queued", claim_owner=None, claim_expires_at=None)
        async with scheduler.job_lease(
            "work", resource="media_compute", group="primary", capacity=2
        ):
            await older.__aexit__(None, None, None)
            ended = True

            assert power.state().holder_count == 1
            assert backend.calls == ["acquire"]
        assert backend.calls == ["acquire", "release"]
    finally:
        if not ended:
            await older.__aexit__(None, None, None)


@pytest.mark.parametrize("gone", ["claimed by another attempt", "removed"])
async def test_the_hold_ends_within_a_heartbeat_once_the_claim_is_gone(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, gone: str
) -> None:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.05)
    backend = _Backend()
    power = PowerInhibitor(backend)
    scheduler = ResourceScheduler(power=power)
    _queued("work")

    async with scheduler.job_lease("work", resource="media_compute", group="primary"):
        assert power.state().holder_count == 1
        if gone == "removed":
            with SessionLocal() as session:
                session.execute(delete(Job).where(Job.id == "work"))
                session.commit()
        else:
            _change("work", claim_owner="another-attempt")

        await _until(lambda: power.state().holder_count == 0)
        assert backend.calls == ["acquire", "release"]
    assert backend.calls == ["acquire", "release"]


async def test_a_paused_row_that_still_names_the_attempt_keeps_its_hold(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.05)
    backend = _Backend()
    power = PowerInhibitor(backend)
    scheduler = ResourceScheduler(power=power)
    _queued("work")

    async with scheduler.job_lease("work", resource="media_compute", group="primary"):
        _change("work", status="paused")
        await asyncio.sleep(0.4)

        assert power.state().holder_count == 1
        assert backend.calls == ["acquire"]
    assert backend.calls == ["acquire", "release"]
