"""Runtime setup can restore an empty worker state before an interpreter exists."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI

from local_lm.config import Settings
from local_lm.processes import ProcessSupervisor
from local_lm.schemas import WorkerStatus
from local_lm.workflow_package_runtime import workflow_package_runtime

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("outcome", ["complete", "error", "cancel"])
async def test_fresh_runtime_setup_restores_the_stopped_worker_on_every_exit(
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    processes: ProcessSupervisor = app.state.services.processes
    monkeypatch.setattr(settings, "comfy_executable", None)
    monkeypatch.setattr(settings, "comfy_directory", None)
    initial = processes.statuses()
    media = next(worker for worker in initial if worker.name == "media")
    running = False
    stops: list[str] = []
    starts: list[str] = []

    def statuses() -> list[WorkerStatus]:
        return [
            worker.model_copy(
                update={"running": running, "state": "ready" if running else "stopped"}
            )
            if worker.name == "media"
            else worker
            for worker in initial
        ]

    async def stop(name: str) -> WorkerStatus:
        nonlocal running
        stops.append(name)
        assert name == "media"
        running = False
        return media.model_copy(update={"running": False, "state": "stopped"})

    async def start() -> WorkerStatus:
        starts.append("media")
        return media

    monkeypatch.setattr(processes, "statuses", statuses)
    monkeypatch.setattr(processes, "stop", stop)
    monkeypatch.setattr(processes, "start_media", start)

    async def setup() -> None:
        nonlocal running
        async with workflow_package_runtime(processes):
            assert settings.comfy_executable is None
            monkeypatch.setattr(settings, "comfy_directory", tmp_path / "new-runtime")
            monkeypatch.setattr(settings, "comfy_executable", tmp_path / "new-python")
            running = True
            if outcome == "error":
                raise ValueError("Neutral setup failure")
            if outcome == "cancel":
                raise asyncio.CancelledError

    if outcome == "complete":
        await setup()
    elif outcome == "error":
        with pytest.raises(ValueError, match="Neutral setup failure"):
            await setup()
    else:
        with pytest.raises(asyncio.CancelledError):
            await setup()
    assert not running and stops == ["media"] and starts == []
