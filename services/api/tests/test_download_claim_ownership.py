"""A displaced transfer cannot publish an install or finish a different claim."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from httpx2 import AsyncClient
from huggingface_hub import HfApi
from run_waits import PATIENCE_SECONDS
from sqlalchemy import delete, func, select, update
from test_downloads import gguf_bytes
from test_scheduler_claim_hold import _until

from local_lm import scheduler as scheduler_module
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus, utcnow
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.models import Job, ModelInstall, ModelProfile
from local_lm.scheduler import ResourceScheduler
from local_lm.schemas import DownloadRequest


@dataclass
class _Transfer:
    task: asyncio.Task[None]
    heartbeat: asyncio.Task[None]
    gate: asyncio.Event
    content: bytes


async def _transfer(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, *, fail: bool = False
) -> _Transfer:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.02)
    entered = asyncio.Event()
    gate = asyncio.Event()
    content = gguf_bytes("llama")
    digest = hashlib.sha256(content).hexdigest()
    manager = DownloadManager(settings, EventBroker(), scheduler=ResourceScheduler())
    catalog = Mock(spec=HfApi)
    catalog.model_info.return_value = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="fixture.gguf", size=len(content), lfs={"sha256": digest})
        ],
        sha="c" * 40,
        pipeline_tag="text-generation",
        tags=["gguf"],
        gated=False,
    )
    manager._api = catalog

    async def download_file(**kwargs: object) -> str:
        staging, filename = kwargs["staging"], kwargs["filename"]
        assert isinstance(staging, Path) and isinstance(filename, str)
        target = staging / filename
        target.write_bytes(content)
        entered.set()
        await gate.wait()
        if fail:
            raise ValueError("Neutral transfer failure")
        return str(target)

    monkeypatch.setattr(manager, "_download_file", download_file)
    request = DownloadRequest(
        remote_id="neutral/claim-transfer",
        revision="c" * 40,
        role="chat",
        engine="llama.cpp",
        allow_patterns=["fixture.gguf"],
        expected_sha256={"fixture.gguf": digest},
    )
    with SessionLocal() as session:
        session.add(
            Job(
                id="download-claim",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.QUEUED.value,
                payload_json=request.model_dump(mode="json"),
                queue_resource="network_transfer",
                queue_group="network",
                enqueued_at=utcnow(),
            )
        )
        session.commit()
    manager.start("download-claim")
    task = manager._tasks["download-claim"]
    try:
        await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
        heartbeat = next(
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "job-heartbeat-download-claim"
        )
        return _Transfer(task, heartbeat, gate, content)
    except BaseException:
        gate.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


def _move_claim(disposition: str) -> None:
    with SessionLocal() as session:
        if disposition == "removed":
            session.execute(delete(Job).where(Job.id == "download-claim"))
        else:
            values: dict[str, object] = {"claim_owner": None}
            if disposition == "replaced":
                values.update(claim_owner="replacement-download", attempt=2)
            session.execute(update(Job).where(Job.id == "download-claim").values(**values))
        session.commit()


def _state() -> tuple[object, ...]:
    with SessionLocal() as session:
        job = session.get(Job, "download-claim")
        assert job is not None
        return (
            job.status,
            job.claim_owner,
            job.attempt,
            job.progress,
            job.completed_at,
            job.error,
            dict(job.result_json),
            session.scalar(select(func.count()).select_from(ModelInstall)),
            session.scalar(select(func.count()).select_from(ModelProfile)),
        )


async def _finish(work: _Transfer) -> None:
    work.gate.set()
    work.task.cancel()
    await asyncio.gather(work.task, return_exceptions=True)


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "removed"])
async def test_a_download_stops_after_its_claim_is_lost(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch, disposition: str
) -> None:
    work = await _transfer(settings, monkeypatch)
    try:
        _move_claim(disposition)
        await _until(work.heartbeat.done, timeout=PATIENCE_SECONDS)
        await _until(work.task.done, timeout=PATIENCE_SECONDS)
        await asyncio.gather(work.task, return_exceptions=True)
    finally:
        await _finish(work)


@pytest.mark.parametrize("disposition", ["cleared", "replaced"])
@pytest.mark.parametrize("fail", [False, True], ids=["completion", "failure"])
async def test_a_displaced_download_preserves_the_current_job_and_installs(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
    fail: bool,
) -> None:
    work = await _transfer(settings, monkeypatch, fail=fail)
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
        await _finish(work)


async def test_an_owned_download_publishes_its_verified_file_and_profile(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = await _transfer(settings, monkeypatch)
    before = _state()
    try:
        work.gate.set()
        await asyncio.wait_for(work.task, timeout=PATIENCE_SECONDS)
        after = _state()
        assert after[0] == JobStatus.COMPLETE.value
        assert after[7:9] == (int(str(before[7])) + 1, int(str(before[8])) + 1)
        with SessionLocal() as session:
            job = session.get(Job, "download-claim")
            assert job is not None
            install = session.get(ModelInstall, job.result_json["model_install_id"])
            assert install is not None and install.active
            assert (Path(install.local_path) / "fixture.gguf").read_bytes() == work.content
    finally:
        await _finish(work)


async def test_an_owned_download_reports_a_transfer_failure(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = await _transfer(settings, monkeypatch, fail=True)
    before = _state()
    try:
        work.gate.set()
        await asyncio.wait_for(work.task, timeout=PATIENCE_SECONDS)
        after = _state()
        assert after[0] == JobStatus.FAILED.value
        assert after[5] == "Neutral transfer failure"
        assert after[7:9] == before[7:9]
    finally:
        await _finish(work)
