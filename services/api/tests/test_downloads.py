from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.domain import JobKind, JobStatus
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.models import Job
from local_lm.schemas import DownloadRequest


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


def test_duplicate_active_download_requests_reuse_the_existing_job(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    manager = DownloadManager(settings, EventBroker())
    monkeypatch.setattr(manager, "start", lambda _job_id: None)
    request = DownloadRequest(
        remote_id="owner/model",
        revision="abc123",
        role="chat",
        engine="llama.cpp",
        allow_patterns=["model-Q4_K_M.gguf"],
    )
    starts: list[str] = []
    with SessionLocal() as session:
        monkeypatch.setattr(manager, "start", starts.append)
        first = manager.create(session, request)
        second = manager.create(session, request)
        first.status = JobStatus.PAUSED.value
        session.commit()
        starts.clear()
        paused = manager.create(session, request)
        jobs = list(session.query(Job).all())
    assert second.id == first.id
    assert paused.id == first.id
    assert starts == []
    assert len(jobs) == 1


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


async def test_download_file_retries_transient_worker_failures(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DownloadManager(settings, EventBroker())
    attempts = 0
    sleeps: list[int] = []

    async def download_once(**_kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("incomplete HTTP read")
        return "C:/models/model.gguf"

    async def sleep(seconds: int) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(manager, "_download_file_once", download_once)
    monkeypatch.setattr(asyncio, "sleep", sleep)

    path = await manager._download_file(
        job_id="job_test",
        remote_id="owner/model",
        filename="model.gguf",
        revision="a" * 40,
        staging=Path("C:/staging"),
    )

    assert path == "C:/models/model.gguf"
    assert attempts == 3
    assert sleeps == [1, 2]


async def test_transfer_monitor_reports_process_tree_bytes(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    broker = EventBroker()
    manager = DownloadManager(settings, broker)
    with SessionLocal() as session:
        session.add(
            Job(
                id="job_progress",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.RUNNING.value,
                progress=0,
                phase="inspecting",
            )
        )
        session.commit()

    staging = settings.download_dir / "job_progress.partial"
    staging.mkdir(parents=True)
    write_samples = iter([0, 50])
    monkeypatch.setattr(
        manager,
        "_process_tree_write_bytes",
        lambda _pid: next(write_samples, 50),
    )
    stop = asyncio.Event()

    async def stop_after_sample() -> None:
        await asyncio.sleep(0.01)
        stop.set()

    stopper = asyncio.create_task(stop_after_sample())
    await manager._monitor_transfer(
        job_id="job_progress",
        filename="model.safetensors",
        staging=staging,
        process=SimpleNamespace(pid=123),  # type: ignore[arg-type]
        file_size=100,
        completed_bytes=0,
        total_size=100,
        stop=stop,
    )
    await stopper

    with SessionLocal() as session:
        job = session.get(Job, "job_progress")
        assert job
        assert job.phase == "downloading model.safetensors"
        assert job.progress == pytest.approx(0.45)
    event = broker.since(0)[0]
    assert event.type == "download.progress"
    assert event.payload["downloaded_bytes"] == 50
    assert event.payload["file_size_bytes"] == 100
