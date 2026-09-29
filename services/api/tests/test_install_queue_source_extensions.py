"""Source extension writers retain the completion claim that admitted their work."""

from __future__ import annotations

import asyncio
import sys
import threading
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_offer_packages import _setup
from test_workflow_package_execution_plan import _inputs
from test_workflow_package_import_endpoint import _ui_graph
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import source_runtime as source_runtime

from local_lm import comfy_registry_activation_batches as batches
from local_lm import comfy_registry_installs as registry_installs
from local_lm import comfy_registry_launch_verification as launch_verification
from local_lm import comfy_registry_lifecycle as lifecycle
from local_lm import workflow_source_extension_trust as trust_module
from local_lm import workflow_source_runtime
from local_lm.comfy_registry_downloads import ComfyRegistryArchiveDownloader
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ComfyRegistryInstall, Job, WorkflowInstallOffer
from local_lm.scheduler import ResourceScheduler
from local_lm.workflow_completion_jobs import stage_workflow_completion_job
from local_lm.workflow_package_preparation import PreparationContext
from local_lm.workflow_source_extensions import (
    ExtensionPreparationServices,
    prepare_workflow_source_extensions,
)


@pytest.mark.parametrize(
    "boundary",
    ["prepare", "failure", "trust", "batch-begin", "batch-verified", "batch-rollback", "recovery"],
)
@pytest.mark.parametrize("replacement", ["token", "attempt", "released", "expiry", "none"])
async def test_source_extension_writes_require_the_original_completion_claim(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_runtime: dict[str, Any],
    boundary: str,
    replacement: str,
) -> None:
    services = app.state.services
    services.settings.comfy_executable = Path(sys.executable)
    context = PreparationContext(
        Path(sys.executable), services.settings.custom_node_dir, services.settings.registry_dir
    )
    inputs = _inputs()
    original_probe = inputs["interpreter_probe"]

    async def probe(executable: Path) -> Any:
        environment, tags = await original_probe(executable)
        return environment, tags, ()

    inputs["interpreter_probe"] = probe
    _prepared_inputs, context, offer_id, _saved = await _setup(
        tmp_path,
        package_inputs=[inputs],
        source_payload={
            "name": "Neutral source claim verification",
            "operation": "text_to_image",
            "ui_graph": _ui_graph(),
            "dependencies": {"version": 1, "slots": []},
            "selections": [],
        },
        preparation_context=context,
        runtime_plan=services.processes.runtimes.preflight("comfyui"),
        available_nodes=frozenset(source_runtime),
    )
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        job_id = stage_workflow_completion_job(session, offer).id
        child_id = session.scalar(select(Job.id).where(Job.kind == "registry_prepare"))
        assert child_id is not None
        session.commit()
    for name in inputs["selected"].node_types:
        source_runtime[name] = {
            "python_module": "custom_nodes.example",
            "input": {"required": {}},
            "output": [],
        }

    async def inventory() -> frozenset[str]:
        return frozenset(source_runtime)

    async def validate(_graph: dict[str, Any]) -> list[str]:
        return ["Neutral validation refusal"] if boundary == "batch-rollback" else []

    monkeypatch.setattr(services.processes, "comfy_node_inventory", inventory)
    monkeypatch.setattr(services.engines.media, "validate_workflow", validate)
    entered, release = threading.Event(), threading.Event()
    armed = boundary != "recovery"

    def delayed(original: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(30), "Source extension verification was not released"
        return result

    if boundary == "prepare":
        # Read where it is defined; it is the same function the lifecycle calls.
        original_binding = registry_installs.verify_comfy_registry_wheel_binding
        monkeypatch.setattr(
            lifecycle,
            "verify_comfy_registry_wheel_binding",
            lambda *args, **kwargs: delayed(original_binding, *args, **kwargs),
        )
    elif boundary == "trust":
        original_launch = launch_verification.verify_comfy_registry_launch
        monkeypatch.setattr(
            trust_module,
            "verify_comfy_registry_launch",
            lambda *args, **kwargs: delayed(original_launch, *args, **kwargs),
        )
    elif boundary in {"batch-begin", "batch-verified", "recovery"}:
        original_verification = launch_verification._verify_comfy_registry_launch

        def verify_batch(*args: Any, **kwargs: Any) -> Any:
            result = original_verification(*args, **kwargs)
            prepare_installs = kwargs.get("prepare_installs")
            prefix = "_verified." if boundary == "batch-verified" else "_begin."
            if (
                armed
                and prepare_installs is not None
                and prepare_installs.__qualname__.startswith(prefix)
            ):
                entered.set()
                assert release.wait(30), "Batch verification was not released"
            return result

        monkeypatch.setattr(batches, "_verify_comfy_registry_launch", verify_batch)
    elif boundary == "batch-rollback":
        original_restore = batches._restore_flags

        def restore(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            assert release.wait(30), "Batch rollback was not released"
            return original_restore(*args, **kwargs)

        monkeypatch.setattr(batches, "_restore_flags", restore)

    async def archive_response(_request: httpx.Request) -> httpx.Response:
        if boundary == "failure":
            entered.set()
            async with asyncio.timeout(30):
                while not release.is_set():
                    await asyncio.sleep(0.01)
            return httpx.Response(503)
        return httpx.Response(200, content=inputs["state"]["content"])

    archive = ComfyRegistryArchiveDownloader(transport=httpx.MockTransport(archive_response))
    wheels = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    )

    async def prepare(*args: Any, **kwargs: Any) -> Any:
        return await prepare_workflow_source_extensions(
            *args,
            **kwargs,
            services=ExtensionPreparationServices(
                probe,
                inputs["registry_client"],
                inputs["project_client"],
                inputs["metadata_client"],
                archive,
                wheels,
            ),
        )

    monkeypatch.setattr(workflow_source_runtime, "prepare_workflow_source_extensions", prepare)
    if boundary == "recovery":
        await services.processes.stop("media")
        prepared = await prepare(SessionLocal, offer_id, context=context, media_worker_stopped=True)
        trust_module.trust_workflow_source_extensions(
            SessionLocal, offer_id, context=context, media_worker_stopped=True
        )
        batches._begin(
            SessionLocal,
            offer_id,
            prepared.preparations,
            prepared.execution_plans,
            context.custom_node_root,
            registry_wheel_environment_root(context.state_root),
        )
        armed = True

    runtime_calls: list[str] = []
    original_start = services.processes.start_media
    original_stop = services.processes.stop

    async def start(*args: Any, **kwargs: Any) -> Any:
        runtime_calls.append("start")
        return await original_start(*args, **kwargs)

    async def stop(*args: Any, **kwargs: Any) -> Any:
        runtime_calls.append("stop")
        return await original_stop(*args, **kwargs)

    monkeypatch.setattr(services.processes, "start_media", start)
    monkeypatch.setattr(services.processes, "stop", stop)
    task = asyncio.create_task(
        services.downloads.reconcile_workflow_install_offers(only_offer_id=offer_id)
    )
    try:
        async with asyncio.timeout(30):
            while not entered.is_set():
                assert not task.done(), "Source completion stopped before its write boundary"
                await asyncio.sleep(0.01)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            child = session.get(Job, child_id)
            assert job is not None and job.status == "running" and job.claim_owner is not None
            assert child is not None
            if replacement == "token":
                job.claim_owner = "replacement-claim"
            elif replacement == "attempt":
                job.attempt += 1
            elif replacement == "released":
                job.claim_owner = None
            elif replacement == "expiry":
                job.claim_expires_at = utcnow() - timedelta(seconds=30)
            elif replacement == "child-attempt":
                child.attempt += 1
            child_before = (child.status, child.attempt, deepcopy(child.result_json), child.error)
            installs_before = [
                (row.id, row.trusted, row.active, deepcopy(row.review_json))
                for row in session.scalars(select(ComfyRegistryInstall))
            ]
            session.commit()
        if replacement == "expiry":
            assert ResourceScheduler()._expire_foreign_claims("primary") == [job_id]
        previous_calls = list(runtime_calls)
        release.set()
        try:
            await asyncio.wait_for(task, timeout=30)
        except httpx.HTTPStatusError:
            assert boundary == "failure"
        with SessionLocal() as session:
            child = session.get(Job, child_id)
            offer = session.get(WorkflowInstallOffer, offer_id)
            assert child is not None and offer is not None
            installs = list(session.scalars(select(ComfyRegistryInstall)))
            if replacement != "none":
                assert (child.status, child.attempt, child.result_json, child.error) == child_before
                assert [
                    (row.id, row.trusted, row.active, row.review_json) for row in installs
                ] == installs_before
                if replacement != "child-attempt":
                    assert runtime_calls == previous_calls
            elif boundary == "failure":
                assert child.status == "failed" and not installs
            elif boundary == "batch-rollback":
                assert child.status == "complete" and offer.status == "queued"
                assert len(installs) == 1 and not installs[0].active
            else:
                assert child.status == "complete" and offer.status == "completed"
                assert len(installs) == 1 and installs[0].trusted and installs[0].active
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await services.downloads.close()
        await archive.close()
        await wheels.close()


@pytest.mark.parametrize("boundary", ["prepare", "failure"])
async def test_source_preparation_preserves_a_replaced_child_attempt(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_runtime: dict[str, Any],
    boundary: str,
) -> None:
    await test_source_extension_writes_require_the_original_completion_claim(
        client, app, monkeypatch, tmp_path, source_runtime, boundary, "child-attempt"
    )
