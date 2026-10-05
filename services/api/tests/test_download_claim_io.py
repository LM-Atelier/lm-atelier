"""Download cancellation joins its disk operation before releasing the claim."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from httpx2 import AsyncClient
from huggingface_hub import HfApi
from test_download_claim_ownership import _finish, _move_claim, _state, _transfer
from test_downloads import gguf_bytes
from test_scheduler_claim_hold import _until

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus, utcnow
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.models import Job, ModelComponentManifest, ModelInstall
from local_lm.scheduler import ResourceScheduler
from local_lm.schemas import DownloadRequest


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "cancelled", "owned"])
async def test_download_disk_work_finishes_before_its_execution_releases_the_lease(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
) -> None:
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_digest = DownloadManager._sha256_file

    def digest(path: Path) -> str:
        entered.set()
        assert release.wait(timeout=5), "The neutral disk operation was not released."
        try:
            return real_digest(path)
        finally:
            finished.set()

    monkeypatch.setattr(DownloadManager, "_sha256_file", staticmethod(digest))
    work = await _transfer(settings, monkeypatch)
    try:
        work.gate.set()
        await _until(entered.is_set)
        if disposition in {"cleared", "replaced"}:
            _move_claim(disposition)
            before = _state()
            await _until(work.heartbeat.done)
            # Cancellation must wait for the disk operation to finish.
            work.task.cancel()
        elif disposition == "cancelled":
            before = _state()
            work.task.cancel()
        else:
            before = _state()
        await asyncio.sleep(0.04)
        assert not work.task.done() and not finished.is_set()
        if disposition == "cancelled":
            assert _state()[1:3] == before[1:3]
        release.set()
        await asyncio.wait_for(asyncio.gather(work.task, return_exceptions=True), timeout=3)
        assert finished.is_set()
        if disposition in {"cleared", "replaced"}:
            assert _state() == before
        elif disposition == "owned":
            assert _state()[0] == "complete"
    finally:
        release.set()
        await _finish(work)


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "owned"])
async def test_a_reused_component_is_published_only_by_its_current_download_claim(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
) -> None:
    content = gguf_bytes("llama")
    digest = hashlib.sha256(content).hexdigest()
    installed = settings.model_dir / "neutral-reuse-source"
    installed.mkdir()
    (installed / "fixture.gguf").write_bytes(content)
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
    request = DownloadRequest(
        remote_id="neutral/claim-reuse",
        revision="c" * 40,
        role="chat",
        engine="llama.cpp",
        allow_patterns=["fixture.gguf"],
        expected_sha256={"fixture.gguf": digest},
    )
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="neutral-reuse-model",
                name="Neutral reuse fixture",
                role="chat",
                engine="llama.cpp",
                local_path=str(installed),
                active=False,
            )
        )
        session.flush()
        session.add(
            ModelComponentManifest(
                model_install_id="neutral-reuse-model",
                kind="gguf_model",
                relative_path="fixture.gguf",
                target_folder="models",
                sha256=digest,
                size_bytes=len(content),
                required=True,
                metadata_json={},
            )
        )
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
    real_copy = shutil.copyfile
    before: list[tuple[object, ...]] = []

    def copy(source: Path, destination: Path) -> str:
        result = real_copy(source, destination)
        if source == installed / "fixture.gguf":
            if disposition != "owned":
                _move_claim(disposition)
            before.append(_state())
        return str(result)

    monkeypatch.setattr("local_lm.downloads.shutil.copyfile", copy)
    task = asyncio.create_task(manager._download("download-claim"))
    try:
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
        assert len(before) == 1
        staged = settings.download_dir / "download-claim.partial" / "fixture.gguf"
        if disposition == "owned":
            assert _state()[0] == "complete"
        else:
            assert _state() == before[0]
            assert not staged.exists()
        assert (installed / "fixture.gguf").read_bytes() == content
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
