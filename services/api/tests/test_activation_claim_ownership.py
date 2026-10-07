"""A displaced activation stops without completing a different claim's job."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from sqlalchemy import delete, update
from test_scheduler_claim_hold import _until

from local_lm import scheduler as scheduler_module
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus, utcnow
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.models import Job, ModelInstall
from local_lm.scheduler import ResourceScheduler


@dataclass
class _Activation:
    task: asyncio.Task[None]
    heartbeat: asyncio.Task[None]
    gate: asyncio.Event


async def _activation(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> _Activation:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.02)
    entered = asyncio.Event()
    gate = asyncio.Event()
    manager = DownloadManager(settings, EventBroker(), scheduler=ResourceScheduler())

    async def probe(**kwargs: object) -> object:
        entered.set()
        await gate.wait()
        return object()

    monkeypatch.setattr(manager, "_activate_chat_install", probe)
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="activation-model",
                name="Neutral activation fixture",
                role="chat",
                engine="llama.cpp",
                local_path=str(settings.model_dir / "activation-fixture"),
                manifest_json={"expected_sha256": {"fixture.gguf": "ab" * 32}},
                active=False,
            )
        )
        session.add(
            Job(
                id="activation-claim",
                kind=JobKind.ACTIVATE.value,
                status=JobStatus.QUEUED.value,
                payload_json={"install_id": "activation-model"},
                queue_resource="primary_compute",
                queue_group="primary",
                enqueued_at=utcnow(),
            )
        )
        session.commit()
    manager.start_activation("activation-claim")
    task = manager._tasks["activation-claim"]
    try:
        await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
        heartbeat = next(
            t for t in asyncio.all_tasks() if t.get_name() == "job-heartbeat-activation-claim"
        )
        return _Activation(task, heartbeat, gate)
    except BaseException:
        gate.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


def _move_claim(disposition: str) -> None:
    with SessionLocal() as session:
        if disposition == "removed":
            session.execute(delete(Job).where(Job.id == "activation-claim"))
        else:
            values: dict[str, object] = {"status": JobStatus.RUNNING.value, "claim_owner": None}
            if disposition == "replaced":
                values.update(claim_owner="replacement-activation", attempt=2)
            session.execute(update(Job).where(Job.id == "activation-claim").values(**values))
        session.commit()


def _state() -> tuple[object, ...]:
    with SessionLocal() as session:
        job = session.get(Job, "activation-claim")
        assert job is not None
        return (
            job.status,
            job.claim_owner,
            job.attempt,
            job.completed_at,
            job.error,
            dict(job.result_json),
        )


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "removed"])
async def test_an_activation_stops_after_its_claim_is_lost(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch, disposition: str
) -> None:
    work = await _activation(settings, monkeypatch)
    try:
        _move_claim(disposition)
        await _until(work.heartbeat.done, timeout=PATIENCE_SECONDS)
        await _until(work.task.done, timeout=PATIENCE_SECONDS)
        await asyncio.gather(work.task, return_exceptions=True)
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)


async def test_an_owned_activation_finishes_normally(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = await _activation(settings, monkeypatch)
    try:
        work.gate.set()
        await asyncio.wait_for(work.task, timeout=PATIENCE_SECONDS)
        assert _state()[0] == JobStatus.COMPLETE.value
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)


@pytest.mark.parametrize("disposition", ["cleared", "replaced"])
async def test_displaced_activation_completion_preserves_the_current_job(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch, disposition: str
) -> None:
    work = await _activation(settings, monkeypatch)
    try:
        _move_claim(disposition)
        before = _state()
        await _until(work.heartbeat.done, timeout=PATIENCE_SECONDS)
        work.gate.set()
        await asyncio.wait_for(
            asyncio.gather(work.task, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        assert _state() == before
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)
