"""A waiting activation cleanup must preserve a replacement worker."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

import pytest
from httpx2 import AsyncClient
from test_activation_claim_ownership import _move_claim, _state

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus, utcnow
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.models import Job, ModelInstall, ModelProfile
from local_lm.processes import ProcessSupervisor, WorkerRecord, _RotatingWorkerLog
from local_lm.scheduler import ResourceScheduler
from local_lm.schemas import WorkerStatus


class _ObservedLock(asyncio.Lock):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    async def acquire(self) -> Literal[True]:
        self.entered.set()
        return await super().acquire()


class _ControlledProcess(asyncio.subprocess.Process):
    def __init__(self) -> None:
        self.pid = 2_000_000_000
        self.stdout = None
        self.stderr = None
        self.result: int | None = None
        self.terminated = False
        self.exited = asyncio.Event()

    @property
    def returncode(self) -> int | None:
        return self.result

    def terminate(self) -> None:
        self.terminated = True
        self.result = 0
        self.exited.set()

    def kill(self) -> None:
        self.result = -9
        self.exited.set()

    async def wait(self) -> int:
        await self.exited.wait()
        assert self.result is not None
        return self.result


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "owned"])
async def test_waiting_activation_cleanup_checks_the_claim_after_the_worker_lock(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
) -> None:
    supervisor = ProcessSupervisor(settings)
    monkeypatch.setattr(supervisor, "_matching_worker_processes", lambda _name: [])
    monkeypatch.setattr(supervisor, "_descendant_processes", lambda _pid: [])
    monkeypatch.setattr(supervisor, "_refresh_worker_identities_after_stop", lambda _name: None)
    monkeypatch.setattr(supervisor, "_process_tree_rss", lambda _pid: None)
    lock = _ObservedLock()
    supervisor._locks["chat"] = lock
    await lock.acquire()
    lock.entered.clear()

    async def load_chat(
        _profile: ModelProfile, _install: ModelInstall, **_kwargs: object
    ) -> WorkerStatus:
        return next(status for status in supervisor.statuses() if status.name == "chat")

    monkeypatch.setattr(supervisor, "load_chat", load_chat)

    class Chat:
        async def capabilities(self) -> object:
            from types import SimpleNamespace

            return SimpleNamespace(
                healthy=True, version="neutral-runtime", input_modalities=["text"]
            )

        async def count_tokens(self, _messages: list[dict[str, object]]) -> int:
            return 4

        async def stream(self, _request: ChatRequest) -> AsyncIterator[ChatEvent]:
            yield ChatEvent(type="token", text="OK")
            yield ChatEvent(type="complete")

    from unittest.mock import Mock

    chat = Chat()
    manager = DownloadManager(
        settings,
        EventBroker(),
        chat_adapter=lambda: Mock(spec_set=Chat, wraps=chat),
        processes=supervisor,
        scheduler=ResourceScheduler(),
    )
    with SessionLocal() as session:
        session.add_all(
            [
                ModelInstall(
                    id="activation-model",
                    name="Neutral activation fixture",
                    role="chat",
                    engine="llama.cpp",
                    local_path=str(settings.model_dir / "activation-fixture"),
                    manifest_json={"expected_sha256": {"fixture.gguf": "ab" * 32}},
                    active=False,
                ),
                Job(
                    id="activation-claim",
                    kind=JobKind.ACTIVATE.value,
                    status=JobStatus.QUEUED.value,
                    payload_json={"install_id": "activation-model"},
                    queue_resource="primary_compute",
                    queue_group="primary",
                    enqueued_at=utcnow(),
                ),
            ]
        )
        session.commit()
    manager.start_activation("activation-claim")
    task = manager._tasks["activation-claim"]
    process = _ControlledProcess()
    record = WorkerRecord(
        "chat",
        process,
        ["neutral-worker"],
        _RotatingWorkerLog(tmp_path / "replacement-worker.log"),
        profile_id="replacement-profile",
        state="ready",
    )
    try:
        await asyncio.wait_for(lock.entered.wait(), timeout=2)
        assert not task.done()
        supervisor._workers["chat"] = record
        if disposition != "owned":
            _move_claim(disposition)
        before = _state()
        lock.release()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2)
        if disposition == "owned":
            assert process.terminated
            assert "chat" not in supervisor._workers
            with SessionLocal() as session:
                job = session.get(Job, "activation-claim")
                assert job is not None and job.status == JobStatus.COMPLETE.value
        else:
            assert not process.terminated
            assert supervisor._workers["chat"] is record
            assert record.state == "ready" and not record.stopping
            assert _state() == before
    finally:
        task.cancel()
        if lock.locked():
            lock.release()
        await asyncio.gather(task, return_exceptions=True)
        record.log.close()
