"""A claimed job keeps the computer awake for exactly as long as it holds its claim."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.main import create_app
from local_lm.models import Job
from local_lm.power_inhibition import POWER_REASON, PowerInhibitor
from local_lm.scheduler import ResourceScheduler


class _Backend:
    name = "test"
    supported = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def acquire(self, reason: str) -> object:
        assert reason == POWER_REASON
        self.calls.append("acquire")
        return object()

    def release(self, handle: object) -> None:
        self.calls.append("release")


def _queued(*job_ids: str, resource: str = "media_compute") -> None:
    with SessionLocal() as session:
        session.add_all(
            Job(
                id=job_id,
                kind="image",
                status="queued",
                queue_group="primary",
                queue_resource=resource,
                enqueued_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            for job_id in job_ids
        )
        session.commit()


async def test_a_claimed_media_job_holds_the_computer_awake_until_its_lease_ends(
    client: AsyncClient,
) -> None:
    backend = _Backend()
    power = PowerInhibitor(backend)
    scheduler = ResourceScheduler(power=power)
    _queued("first")

    async with scheduler.job_lease("first", resource="media_compute", group="primary"):
        assert backend.calls == ["acquire"]
        assert power.state().kind_counts == {"generation": 1}

    assert backend.calls == ["acquire", "release"]
    assert power.state().holder_count == 0


async def test_a_job_waiting_its_turn_holds_nothing(client: AsyncClient) -> None:
    backend = _Backend()
    power = PowerInhibitor(backend)
    scheduler = ResourceScheduler(power=power)
    _queued("first", "second")
    second_claimed = asyncio.Event()

    async def second() -> None:
        async with scheduler.job_lease("second", resource="media_compute", group="primary"):
            second_claimed.set()

    async with scheduler.job_lease("first", resource="media_compute", group="primary"):
        waiting = asyncio.create_task(second())
        await asyncio.sleep(0.5)
        assert not second_claimed.is_set()
        assert power.state().holder_count == 1
    await asyncio.wait_for(waiting, timeout=PATIENCE_SECONDS)

    assert backend.calls == ["acquire", "release", "acquire", "release"]


@pytest.mark.parametrize(
    ("resource", "kind"),
    [("network_transfer", "download"), ("primary_compute", "runtime_preparation")],
)
async def test_downloads_and_preparation_hold_under_their_own_kind(
    client: AsyncClient, resource: str, kind: str
) -> None:
    power = PowerInhibitor(_Backend())
    scheduler = ResourceScheduler(power=power)
    _queued("work", resource=resource)

    async with scheduler.job_lease("work", resource=resource, group="primary"):
        assert power.state().kind_counts == {kind: 1}


async def test_a_chat_turn_holds_nothing(client: AsyncClient) -> None:
    backend = _Backend()
    scheduler = ResourceScheduler(power=PowerInhibitor(backend))
    _queued("turn", resource="interactive_compute")

    async with scheduler.job_lease("turn", resource="interactive_compute", group="primary"):
        pass

    assert backend.calls == []


async def test_the_hold_ends_even_when_the_work_fails_or_the_claim_cannot_be_released(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _Backend()
    power = PowerInhibitor(backend)
    scheduler = ResourceScheduler(power=power)
    _queued("first")

    async def refusing_release(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the database refused the release")

    monkeypatch.setattr(scheduler, "_release_job", refusing_release)
    with pytest.raises(RuntimeError):
        async with scheduler.job_lease("first", resource="media_compute", group="primary"):
            raise RuntimeError("the work failed")

    assert backend.calls == ["acquire", "release"]
    assert power.state().holder_count == 0


async def test_a_scheduler_sharing_a_pool_shares_its_power_hold(client: AsyncClient) -> None:
    backend = _Backend()
    pool = ResourceScheduler(power=PowerInhibitor(backend))
    scheduler = ResourceScheduler(resource_pool=pool)
    _queued("first")

    async with scheduler.job_lease("first", resource="media_compute", group="primary"):
        assert backend.calls == ["acquire"]


@pytest.mark.parametrize("enabled", [True, False])
async def test_the_application_wires_the_setting_into_its_scheduler(
    settings: Settings, enabled: bool
) -> None:
    settings.keep_awake_during_work = enabled
    app: FastAPI = create_app(settings)
    power = app.state.services.scheduler._power

    assert isinstance(power, PowerInhibitor)
    assert power.state().enabled is enabled
