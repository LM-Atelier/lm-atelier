"""Only the current image or video activation may publish probe evidence and workflows."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from sqlalchemy import func, select
from test_activation_claim_ownership import _Activation, _move_claim, _state
from test_scheduler_claim_hold import _until

from local_lm import scheduler as scheduler_module
from local_lm.adapters.base import MediaRequest
from local_lm.comfy_templates import ComfyModelDependency, ComfyTemplate, CompiledComfyTemplate
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus, utcnow
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.models import Job, ModelCapabilityEvidence, ModelInstall, WorkflowRevision
from local_lm.scheduler import ResourceScheduler


async def _media_probe(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, role: Literal["image", "video"]
) -> _Activation:
    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.02)
    entered = asyncio.Event()
    gate = asyncio.Event()
    operation = "text_to_image" if role == "image" else "text_to_video"
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="neutral-activation-" + role,
            path=settings.state_dir / "neutral-template.json",
            role=role,
            operation=operation,
            score=1,
            sha256="a" * 64,
            dependencies=(
                ComfyModelDependency(
                    remote_id="neutral/media",
                    revision="neutral",
                    path="fixture.safetensors",
                    directory="checkpoints",
                    name="fixture.safetensors",
                    url="",
                ),
            ),
        ),
        ui_graph={"nodes": []},
        api_graph={"latent": {"class_type": "EmptyLatentImage", "inputs": {}}},
        input_schema={"type": "object", "properties": {}},
    )

    class Media:
        async def object_info(self) -> dict[str, object]:
            return {}

        async def validate_workflow(self, _graph: dict[str, object]) -> list[str]:
            return []

        async def probe_workflow(self, request: MediaRequest, *, timeout_seconds: float) -> None:
            assert request.operation == operation and timeout_seconds == 300
            entered.set()
            await gate.wait()

        async def capabilities(self) -> SimpleNamespace:
            return SimpleNamespace(healthy=True, version="neutral-media-runtime")

    class Processes:
        async def start_media(self, _model_paths: object = None, **_kwargs: object) -> None:
            return None

    manager = DownloadManager(
        settings,
        EventBroker(),
        media_adapter=Mock(spec_set=Media, wraps=Media()),
        processes=Mock(spec_set=Processes, wraps=Processes()),
        scheduler=ResourceScheduler(),
    )
    monkeypatch.setattr(manager.comfy_templates, "compile", lambda *_args, **_kwargs: compiled)
    with SessionLocal() as session:
        session.add_all(
            [
                ModelInstall(
                    id="activation-model",
                    name="Neutral media activation",
                    role=role,
                    engine="comfyui",
                    local_path=str(settings.model_dir / "activation-fixture"),
                    manifest_json={
                        "expected_sha256": {"fixture.safetensors": "ab" * 32},
                        "workflow_template_id": compiled.template.id,
                        "workflow_template_sha256": compiled.template.sha256,
                        "files": ["fixture.safetensors"],
                        "remote_id": "neutral/media",
                        "revision": "neutral",
                        "comfy_paths": {"checkpoints": "."},
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
    try:
        await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
        heartbeat = next(
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "job-heartbeat-activation-claim"
        )
        return _Activation(task, heartbeat, gate)
    except BaseException:
        gate.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


def _counts() -> tuple[bool, int, int]:
    with SessionLocal() as session:
        install = session.get(ModelInstall, "activation-model")
        assert install is not None
        evidence = session.scalar(select(func.count()).select_from(ModelCapabilityEvidence))
        workflows = session.scalar(select(func.count()).select_from(WorkflowRevision))
        assert evidence is not None and workflows is not None
        return install.active, evidence, workflows


@pytest.mark.parametrize("role", ["image", "video"])
@pytest.mark.parametrize("disposition", ["cleared", "replaced"])
async def test_a_displaced_media_probe_preserves_install_evidence_and_workflows(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    role: Literal["image", "video"],
    disposition: str,
) -> None:
    work = await _media_probe(settings, monkeypatch, role)
    try:
        _move_claim(disposition)
        before_job = _state()
        before_counts = _counts()
        await _until(work.heartbeat.done, timeout=PATIENCE_SECONDS)
        work.gate.set()
        await asyncio.wait_for(
            asyncio.gather(work.task, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        assert _counts() == before_counts
        assert _state() == before_job
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)


@pytest.mark.parametrize("role", ["image", "video"])
async def test_an_owned_media_probe_records_evidence_and_workflows(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    role: Literal["image", "video"],
) -> None:
    work = await _media_probe(settings, monkeypatch, role)
    try:
        before = _counts()
        work.gate.set()
        await asyncio.wait_for(work.task, timeout=PATIENCE_SECONDS)
        active, evidence, workflows = _counts()
        assert active and evidence == before[1] + 1 and workflows == before[2] + 1
        assert _state()[0] == JobStatus.COMPLETE.value
    finally:
        work.gate.set()
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)
