"""An activation waiting to launch must preserve a newer worker."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from test_activation_claim_ownership import _move_claim, _state
from test_activation_worker_replacement import _ControlledProcess, _ObservedLock

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus, utcnow
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.models import Job, ModelInstall
from local_lm.processes import ProcessSupervisor, WorkerRecord, _RotatingWorkerLog
from local_lm.scheduler import ResourceScheduler


@pytest.mark.parametrize("check", ["worker", "engine"])
@pytest.mark.parametrize("disposition", ["cleared", "replaced", "owned"])
async def test_waiting_activation_start_checks_the_claim_after_the_worker_lock(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
    check: str,
) -> None:
    executable = tmp_path / "neutral-runtime.exe"
    executable.write_bytes(b"neutral runtime marker")
    settings.llama_executable = executable
    model_root = settings.model_dir / "activation-fixture"
    model_root.mkdir(parents=True)
    (model_root / "fixture.gguf").write_bytes(b"GGUF")
    supervisor = ProcessSupervisor(settings)
    monkeypatch.setattr(supervisor, "_matching_worker_processes", lambda _name: [])
    monkeypatch.setattr(supervisor, "_descendant_processes", lambda _pid: [])
    monkeypatch.setattr(supervisor, "_refresh_worker_identities_after_stop", lambda _name: None)
    monkeypatch.setattr(supervisor, "_record_worker_process_tree", lambda _name, _pid: None)
    monkeypatch.setattr(supervisor, "_process_tree_rss", lambda _pid: None)
    monkeypatch.setattr(supervisor, "_reclaim_port_from_our_own_children", AsyncMock())
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())
    monkeypatch.setattr(supervisor, "_wait_healthy", AsyncMock())
    monkeypatch.setattr(supervisor, "_capture_process_output", AsyncMock())
    monkeypatch.setattr(supervisor, "_monitor_worker", AsyncMock())
    created: list[_ControlledProcess] = []

    async def create_process(*_command: str, **_kwargs: object) -> _ControlledProcess:
        process = _ControlledProcess()
        created.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    lock = _ObservedLock()
    supervisor._locks["chat"] = lock
    await lock.acquire()
    lock.entered.clear()

    class Chat:
        async def capabilities(self) -> SimpleNamespace:
            return SimpleNamespace(
                healthy=True, version="neutral-runtime", input_modalities=["text"]
            )

        async def count_tokens(self, _messages: list[dict[str, object]]) -> int:
            return 4

        async def stream(self, _request: ChatRequest) -> AsyncIterator[ChatEvent]:
            yield ChatEvent(type="token", text="OK")
            yield ChatEvent(type="complete")

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
                    local_path=str(model_root),
                    manifest_json={
                        "files": ["fixture.gguf"],
                        "expected_sha256": {"fixture.gguf": "ab" * 32},
                    },
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
    replacement = _ControlledProcess()
    record = WorkerRecord(
        "chat",
        replacement,
        ["neutral-replacement-worker"],
        _RotatingWorkerLog(tmp_path / "replacement-worker.log"),
        profile_id="replacement-profile",
        state="ready",
    )
    try:
        await asyncio.wait_for(lock.entered.wait(), timeout=PATIENCE_SECONDS)
        assert not task.done()
        supervisor._workers["chat"] = record
        if disposition != "owned":
            _move_claim(disposition)
            settings.chat_engine = "vllm"
        before = _state()
        lock.release()
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        if disposition == "owned":
            assert replacement.terminated
            assert len(created) == 1 and created[0].terminated
            assert "chat" not in supervisor._workers
            with SessionLocal() as session:
                job = session.get(Job, "activation-claim")
                assert job is not None and job.status == JobStatus.COMPLETE.value
            assert settings.chat_engine == "llama.cpp"
        elif check == "engine":
            assert settings.chat_engine == "vllm"
        else:
            assert not replacement.terminated
            assert not created
            assert supervisor._workers["chat"] is record
            assert record.state == "ready" and not record.stopping
            assert _state() == before
    finally:
        task.cancel()
        if lock.locked():
            lock.release()
        await asyncio.gather(task, return_exceptions=True)
        record.log.close()
        for current in supervisor._workers.values():
            current.log.close()
