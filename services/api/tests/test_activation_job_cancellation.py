"""Cancelling an activation stops the actual manager task and keeps the cancelled result."""

from __future__ import annotations

import asyncio

import pytest
import test_activation_probe_ownership as probe_module
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from test_activation_claim_ownership import _Activation, _state
from test_activation_probe_ownership import _probe
from test_scheduler_claim_hold import _until

from local_lm.config import Settings
from local_lm.downloads import DownloadManager
from local_lm.scheduler import ResourceScheduler


async def _managed_probe(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> _Activation:
    manager = app.state.services.downloads
    assert isinstance(manager, DownloadManager)

    def bind_service(*args: object, **kwargs: object) -> DownloadManager:
        for name in ("chat_adapter", "processes", "scheduler"):
            value = ResourceScheduler(manager.events) if name == "scheduler" else kwargs[name]
            monkeypatch.setattr(manager, name, value)
        return manager

    monkeypatch.setattr(probe_module, "DownloadManager", bind_service)
    work, _processes = await _probe(settings, monkeypatch)
    assert manager._tasks["activation-claim"] is work.task
    return work


async def test_cancelling_an_activation_stops_its_manager_task(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = await _managed_probe(app, settings, monkeypatch)
    try:
        response = await client.post("/api/jobs/activation-claim/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        await _until(work.task.done, timeout=PATIENCE_SECONDS)
        await asyncio.gather(work.task, return_exceptions=True)
        snapshots = [
            event.payload["job"]
            for event in app.state.services.events.since(0)
            if event.type == "job.progress" and event.entity_id == "activation-claim"
        ]
        assert snapshots and snapshots[-1]["status"] == "cancelled"
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)


async def test_a_cancelled_activation_cannot_publish_a_late_success(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = await _managed_probe(app, settings, monkeypatch)
    try:
        response = await client.post("/api/jobs/activation-claim/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        before = _state()
        work.gate.set()
        await asyncio.wait_for(
            asyncio.gather(work.task, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        assert _state() == before
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)
