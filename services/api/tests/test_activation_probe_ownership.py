"""Only the current activation may publish probe evidence or change the chat worker."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_activation_claim_ownership import _Activation, _move_claim, _state
from test_scheduler_claim_hold import _until

from local_lm import scheduler as scheduler_module
from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus, utcnow
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.models import Job, ModelCapabilityEvidence, ModelInstall, ModelProfile
from local_lm.scheduler import ResourceScheduler


class _Processes:
    def __init__(self) -> None:
        self.stopped: list[str] = []

    def statuses(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="chat", running=False, profile_id=None)]

    async def load_chat(
        self, _profile: ModelProfile, _install: ModelInstall, **_kwargs: object
    ) -> None:
        return None

    async def stop(self, name: str, **_kwargs: object) -> None:
        self.stopped.append(name)


async def _probe(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> tuple[_Activation, _Processes]:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.02)
    entered = asyncio.Event()
    gate = asyncio.Event()
    processes = _Processes()

    class Chat:
        async def capabilities(self) -> SimpleNamespace:
            return SimpleNamespace(
                healthy=True, version="neutral-runtime", input_modalities=["text"]
            )

        async def count_tokens(self, _messages: list[dict[str, object]]) -> int:
            return 4

        async def stream(self, _request: ChatRequest) -> AsyncIterator[ChatEvent]:
            entered.set()
            await gate.wait()
            yield ChatEvent(type="token", text="OK")
            yield ChatEvent(type="complete")

    manager = DownloadManager(
        settings,
        EventBroker(),
        chat_adapter=lambda: Mock(spec_set=Chat, wraps=Chat()),
        processes=Mock(spec_set=_Processes, wraps=processes),
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
                    manifest_json={
                        "files": ["fixture.gguf"],
                        "expected_sha256": {"fixture.gguf": "ab" * 32},
                    },
                    active=False,
                ),
                ModelProfile(
                    id="activation-profile",
                    model_install_id="activation-model",
                    name="Neutral activation profile",
                    role="chat",
                    engine="llama.cpp",
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
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        heartbeat = next(
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "job-heartbeat-activation-claim"
        )
        return _Activation(task, heartbeat, gate), processes
    except BaseException:
        gate.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


@pytest.mark.parametrize("disposition", ["cleared", "replaced"])
async def test_a_displaced_chat_probe_keeps_current_rows_and_capability_evidence(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch, disposition: str
) -> None:
    work, _processes = await _probe(settings, monkeypatch)
    try:
        _move_claim(disposition)
        before = _state()
        await _until(work.heartbeat.done)
        work.gate.set()
        await asyncio.wait_for(asyncio.gather(work.task, return_exceptions=True), timeout=2)
        with SessionLocal() as session:
            install = session.get(ModelInstall, "activation-model")
            assert install is not None and not install.active
            assert session.scalar(select(func.count()).select_from(ModelCapabilityEvidence)) == 0
        assert _state() == before
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)


@pytest.mark.parametrize("disposition", ["cleared", "replaced"])
async def test_a_displaced_chat_probe_does_not_stop_the_current_worker(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch, disposition: str
) -> None:
    work, processes = await _probe(settings, monkeypatch)
    try:
        _move_claim(disposition)
        await _until(work.heartbeat.done)
        work.gate.set()
        await asyncio.wait_for(asyncio.gather(work.task, return_exceptions=True), timeout=2)
        assert processes.stopped == []
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)


async def test_an_owned_chat_probe_records_evidence_and_stops_its_worker(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, processes = await _probe(settings, monkeypatch)
    try:
        work.gate.set()
        await asyncio.wait_for(work.task, timeout=2)
        with SessionLocal() as session:
            install = session.get(ModelInstall, "activation-model")
            assert install is not None and install.active
            assert session.scalar(select(func.count()).select_from(ModelCapabilityEvidence)) == 1
        assert _state()[0] == JobStatus.COMPLETE.value
        assert processes.stopped == ["chat"]
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)
