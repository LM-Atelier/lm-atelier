from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_lm.config import Settings
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker


class FakeWorker:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 1

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        return self.returncode or 0


class FakeCompletedWorker(FakeWorker):
    def __init__(self, command: list[str]) -> None:
        super().__init__()
        self.command = command
        self.payload = b""

    def communicate(self, payload: bytes) -> tuple[bytes, bytes]:
        self.payload = payload
        self.returncode = 0
        return json.dumps({"path": "C:/models/model.gguf"}).encode(), b""


async def test_stop_task_terminates_transfer_process_before_controller(
    settings: Settings,
) -> None:
    manager = DownloadManager(settings, EventBroker())
    worker = FakeWorker()
    controller_started = asyncio.Event()

    async def controller() -> None:
        controller_started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(controller())
    await controller_started.wait()
    manager._tasks["job_test"] = task
    manager._workers["job_test"] = worker  # type: ignore[assignment]

    await manager._stop_task("job_test")

    assert worker.terminated is True
    assert task.cancelled() is True


async def test_download_worker_receives_token_over_stdin_not_command_line(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DownloadManager(settings, EventBroker())
    manager.settings.hf_token = "secret-token"
    workers: list[FakeCompletedWorker] = []

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeCompletedWorker:
        worker = FakeCompletedWorker(command)
        workers.append(worker)
        return worker

    monkeypatch.setattr("local_lm.downloads.subprocess.Popen", fake_popen)
    path = await manager._download_file(
        job_id="job_test",
        remote_id="owner/model",
        filename="model.gguf",
        revision="a" * 40,
        staging=Path("C:/staging"),
    )

    assert path == "C:/models/model.gguf"
    assert workers[0].command[-2:] == ["-m", "local_lm.download_worker"]
    assert "secret-token" not in " ".join(workers[0].command)
    assert json.loads(workers[0].payload)["token"] == "secret-token"
    assert "job_test" not in manager._workers
