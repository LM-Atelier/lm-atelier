"""Accepted raw sources become executable only after managed-runtime verification."""

from __future__ import annotations

import asyncio
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_downloads import safetensors_bytes
from test_workflow_package_import_endpoint import _object_info, _ui_graph
from test_workflow_package_install_plans import configure_runtime
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm import models
from local_lm.db import SessionLocal
from local_lm.processes import ProcessSupervisor
from local_lm.schemas import WorkerStatus

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def source_runtime(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, reviewed_runtime: None, tmp_path: Path
) -> dict[str, Any]:
    configure_runtime(app.state.services.settings, tmp_path)
    processes: ProcessSupervisor = app.state.services.processes
    original_statuses = processes.statuses
    original_stop = processes.stop
    running = True
    launch_digest: str | None = None

    def statuses() -> list[WorkerStatus]:
        return [
            worker.model_copy(
                update={
                    "running": running,
                    "state": "ready" if running else "stopped",
                    "pid": 43210 if running else None,
                }
            )
            if worker.name == "media"
            else worker
            for worker in original_statuses()
        ]

    async def stop(name: str) -> WorkerStatus:
        nonlocal running, launch_digest
        if name != "media":
            return await original_stop(name)
        running = False
        launch_digest = None
        return next(worker for worker in statuses() if worker.name == "media")

    async def start(*_args: Any, **_kwargs: Any) -> WorkerStatus:
        nonlocal running, launch_digest
        running = True
        scope = _kwargs.get("activation_scope")
        launch_digest = scope.launch_sha256 if scope is not None else None
        return next(worker for worker in statuses() if worker.name == "media")

    monkeypatch.setattr(processes, "statuses", statuses)
    monkeypatch.setattr(processes, "stop", stop)
    monkeypatch.setattr(processes, "start_media", start)
    monkeypatch.setattr(processes, "launch_scope_sha256", lambda _name: launch_digest)
    info = _object_info()
    for value in info.values():
        value["python_module"] = "nodes"

    async def object_info() -> dict[str, Any]:
        return copy.deepcopy(info)

    media = app.state.services.engines.media
    monkeypatch.setattr(media, "object_info", object_info, raising=False)
    monkeypatch.setattr(media, "invalidate_object_info_cache", lambda: None, raising=False)
    monkeypatch.setattr(app.state.services.downloads, "media_adapter", media)
    return info


async def _plan(client: AsyncClient) -> str:
    response = await client.post(
        "/api/workflows/packages/install-plans",
        json={
            "name": "Neutral source installation",
            "operation": "text_to_image",
            "ui_graph": _ui_graph(),
            "dependencies": {"version": 1, "slots": []},
            "selections": [],
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["can_accept"], response.text
    return str(response.json()["id"])


async def _approve(client: AsyncClient, app: FastAPI, plan_id: str) -> str:
    response = await client.post(f"/api/workflow-install-offers/{plan_id}/install")
    assert response.status_code == 202, response.text
    assert [item["kind"] for item in response.json()] == ["workflow_install"]
    with SessionLocal() as session:
        offer = session.scalar(
            select(models.WorkflowInstallOffer).where(
                models.WorkflowInstallOffer.source_plan_id == plan_id
            )
        )
        assert offer is not None
        offer_id = offer.id
    tasks = getattr(app.state.services.downloads, "_offer_tasks", {})
    if offer_id in tasks:
        await tasks[offer_id]
    return offer_id


@pytest.mark.parametrize("moment", ["before-inspection", "after-validation"])
async def test_runtime_drift_cannot_complete_an_accepted_source(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, moment: str
) -> None:
    from local_lm import workflow_source_completion

    plan_id = await _plan(client)
    executable = app.state.services.settings.comfy_executable
    assert executable is not None
    if moment == "before-inspection":
        original = workflow_source_completion._read_source

        def read(identifier: str) -> Any:
            source = original(identifier)
            if source is not None:
                executable.write_bytes(b"changed neutral runtime")
            return source

        monkeypatch.setattr(workflow_source_completion, "_read_source", read)
    else:

        async def validate(_graph: dict[str, Any]) -> list[str]:
            executable.write_bytes(b"changed neutral runtime")
            return []

        monkeypatch.setattr(app.state.services.engines.media, "validate_workflow", validate)
    offer_id = await _approve(client, app, plan_id)
    _assert_unfinished(offer_id)
    assert _state(offer_id)[3] == "workflow-runtime-plan-changed"


def _state(offer_id: str) -> tuple[str, str, str, str | None]:
    with SessionLocal() as session:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert offer is not None
        revision = session.get(models.WorkflowRevision, offer.workflow_revision_id)
        assert revision is not None
        return offer.status, revision.workflow_id, revision.id, offer.completion_error_code


def _assert_unfinished(offer_id: str) -> None:
    status, workflow_id, draft_id, code = _state(offer_id)
    assert status == "queued" and code is not None
    with SessionLocal() as session:
        definition = session.get(models.WorkflowDefinition, workflow_id)
        assert definition is not None and definition.current_revision_id == draft_id
        assert definition.family_id is None
        revisions = list(
            session.scalars(
                select(models.WorkflowRevision).where(
                    models.WorkflowRevision.workflow_id == workflow_id
                )
            )
        )
        assert len(revisions) == 1 and revisions[0].api_graph_json == {}
        assert not revisions[0].trusted
        assert session.get(models.WorkflowRevisionReview, draft_id) is None
        assert not list(
            session.scalars(
                select(models.WorkflowActivation).where(
                    models.WorkflowActivation.workflow_revision_id == draft_id
                )
            )
        )


async def test_one_source_approval_creates_reviewed_revision_and_activation_without_downloads(
    client: AsyncClient, app: FastAPI
) -> None:
    plan_id = await _plan(client)
    offer_id = await _approve(client, app, plan_id)
    status, workflow_id, revision_id, code = _state(offer_id)
    assert status == "completed" and code is None
    with SessionLocal() as session:
        definition = session.get(models.WorkflowDefinition, workflow_id)
        revision = session.get(models.WorkflowRevision, revision_id)
        assert definition is not None and revision is not None
        assert definition.current_revision_id == revision_id and definition.family_id
        assert revision.trusted and revision.api_graph_json["1"]["inputs"]["label"] == "camera"
        assert revision.ui_graph_json == _ui_graph() and revision.version == 2
        review = session.get(models.WorkflowRevisionReview, revision_id)
        assert review is not None and review.state == "approved"
        activation = session.scalar(
            select(models.WorkflowActivation).where(
                models.WorkflowActivation.workflow_revision_id == revision_id
            )
        )
        assert activation is not None and activation.is_active
        preference = session.scalar(
            select(models.WorkflowPreference).where(
                models.WorkflowPreference.workflow_family_id == definition.family_id
            )
        )
        assert preference is not None and not preference.is_default
        draft = session.scalar(
            select(models.WorkflowRevision).where(
                models.WorkflowRevision.workflow_id == workflow_id,
                models.WorkflowRevision.version == 1,
            )
        )
        assert draft is not None and not draft.trusted and draft.api_graph_json == {}
        assert session.get(models.WorkflowRevisionReview, draft.id) is None
        jobs = list(session.scalars(select(models.Job)))
        assert len(jobs) == 1 and jobs[0].kind == "workflow_install"
        assert (
            jobs[0].status == "complete" and jobs[0].result_json["activation_id"] == activation.id
        )
    progress = await client.get(f"/api/workflow-install-offers/{plan_id}/progress")
    assert progress.status_code == 200 and progress.json()["phase"] == "completed"
    assert progress.json()["workflow_revision_id"] == revision_id
    assert await _approve(client, app, plan_id) == offer_id
    assert _state(offer_id) == (status, workflow_id, revision_id, code)


@pytest.mark.parametrize("change", ["source", "draft", "operation", "runtime", "pid", "validation"])
async def test_source_or_runtime_drift_leaves_only_the_untrusted_draft(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
    change: str,
) -> None:
    plan_id = await _plan(client)

    async def validate(graph: dict[str, Any]) -> list[str]:
        if change == "validation":
            return ["Neutral validation refusal"]
        if change == "runtime":
            source_runtime["Source"]["input"]["required"]["label"][1]["default"] = "changed"
        elif change == "pid":
            previous = app.state.services.processes.statuses
            monkeypatch.setattr(
                app.state.services.processes,
                "statuses",
                lambda: [
                    worker.model_copy(update={"pid": 43211}) if worker.name == "media" else worker
                    for worker in previous()
                ],
            )
        else:
            with SessionLocal() as session:
                offer = session.scalar(select(models.WorkflowInstallOffer))
                assert offer is not None
                revision = session.get(models.WorkflowRevision, offer.workflow_revision_id)
                assert revision is not None
                if change == "source":
                    plan = session.get(models.WorkflowPackageInstallPlan, plan_id)
                    assert plan is not None
                    plan.request_json = {**plan.request_json, "name": "Changed source"}
                elif change == "draft":
                    revision.ui_graph_json = {"nodes": [], "links": []}
                else:
                    definition = session.get(models.WorkflowDefinition, revision.workflow_id)
                    assert definition is not None
                    definition.operation = "text_to_video"
                session.commit()
        return []

    monkeypatch.setattr(app.state.services.engines.media, "validate_workflow", validate)
    offer_id = await _approve(client, app, plan_id)
    _assert_unfinished(offer_id)


async def test_late_activation_failure_rolls_back_revision_review_and_family(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import workflow_source_completion
    from local_lm.workflow_offer_completion import (
        WorkflowOfferCompletionError,
    )
    from local_lm.workflow_offer_completion import (
        complete_workflow_install_offer as original,
    )

    def interrupted(*args: Any, **kwargs: Any) -> str | None:
        assert original(*args, **kwargs) is not None
        raise WorkflowOfferCompletionError("workflow-completion-unavailable")

    monkeypatch.setattr(workflow_source_completion, "complete_workflow_install_offer", interrupted)
    offer_id = await _approve(client, app, await _plan(client))
    _assert_unfinished(offer_id)


async def test_completed_source_retry_does_not_restore_a_revoked_review(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_id = await _plan(client)
    offer_id = await _approve(client, app, plan_id)
    status, workflow_id, revision_id, _code = _state(offer_id)
    assert status == "completed"
    url = f"/api/workflows/{workflow_id}/revisions/{revision_id}/review"
    preview = await client.get(url)
    assert preview.status_code == 200, preview.text
    response = await client.post(
        url, json={"action": "revoke", "subject_sha256": preview.json()["subject_sha256"]}
    )
    assert response.status_code == 200, response.text
    started: list[str] = []
    monkeypatch.setattr(app.state.services.downloads, "start_workflow_installation", started.append)
    retry = await client.post(f"/api/workflow-install-offers/{plan_id}/install")
    assert retry.status_code == 202, retry.text
    assert started == []
    with SessionLocal() as session:
        review = session.get(models.WorkflowRevisionReview, revision_id)
        assert review is not None and review.state == "revoked"
        revision = session.get(models.WorkflowRevision, revision_id)
        assert revision is not None and not revision.trusted
        assert (
            len(
                list(
                    session.scalars(
                        select(models.WorkflowRevision).where(
                            models.WorkflowRevision.workflow_id == workflow_id
                        )
                    )
                )
            )
            == 2
        )


async def test_compilation_does_not_hold_a_writer_while_awaiting_runtime(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def validate(graph: dict[str, Any]) -> list[str]:
        def write() -> None:
            with SessionLocal() as session:
                session.add(models.AppSetting(key="neutral-compilation-control", value_json=True))
                session.commit()

        await asyncio.wait_for(asyncio.to_thread(write), timeout=5)
        return []

    monkeypatch.setattr(app.state.services.engines.media, "validate_workflow", validate)
    offer_id = await _approve(client, app, await _plan(client))
    assert _state(offer_id)[0] == "completed"
    with SessionLocal() as session:
        assert session.get(models.AppSetting, "neutral-compilation-control") is not None


async def test_worker_restoration_retries_a_zero_download_source_after_releasing_its_lease(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import worker_startup

    services = app.state.services
    await services.processes.stop("media")
    start_installation = services.downloads.start_workflow_installation
    monkeypatch.setattr(services.downloads, "start_workflow_installation", lambda _offer_id: None)
    offer_id = await _approve(client, app, await _plan(client))
    assert _state(offer_id)[0] == "queued"
    monkeypatch.setattr(services.downloads, "start_workflow_installation", start_installation)
    monkeypatch.setattr(services.settings, "media_engine", "comfyui")
    monkeypatch.setattr(worker_startup, "_media_worker_should_restore", lambda _services: True)

    async def refresh() -> int:
        return 0

    monkeypatch.setattr(services.downloads, "refresh_installed_media_workflows", refresh)
    await asyncio.wait_for(worker_startup.restore_configured_workers(services), timeout=10)
    assert _state(offer_id)[0] == "completed"


async def test_repeated_cancellation_retains_the_primary_lease_until_compilation_finishes(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    acquired = asyncio.Event()

    async def validate(graph: dict[str, Any]) -> list[str]:
        entered.set()
        await release.wait()
        return []

    monkeypatch.setattr(app.state.services.engines.media, "validate_workflow", validate)
    plan_id = await _plan(client)
    response = await client.post(f"/api/workflow-install-offers/{plan_id}/install")
    assert response.status_code == 202, response.text
    await asyncio.wait_for(entered.wait(), timeout=5)
    with SessionLocal() as session:
        offer = session.scalar(select(models.WorkflowInstallOffer))
        assert offer is not None
        offer_id = offer.id
    manager = app.state.services.downloads
    completion = manager._offer_tasks[offer_id]
    manager.start_workflow_installation(offer_id)
    assert manager._offer_tasks[offer_id] is completion

    async def take_lease() -> None:
        async with app.state.services.scheduler.lease("primary"):
            acquired.set()

    waiting = asyncio.create_task(take_lease())
    try:
        completion.cancel()
        await asyncio.sleep(0)
        completion.cancel()
        await asyncio.sleep(0)
        assert not completion.done() and not acquired.is_set()
        release.set()
        await asyncio.gather(completion, return_exceptions=True)
        await asyncio.wait_for(waiting, timeout=5)
        assert acquired.is_set() and _state(offer_id)[0] == "completed"
    finally:
        release.set()
        await asyncio.gather(completion, waiting, return_exceptions=True)


@pytest.mark.parametrize("change", ["none", "bytes", "request", "undeclared"])
async def test_source_completion_requires_its_exact_downloaded_declared_resource(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
    change: str,
) -> None:
    from local_lm.model_manifests import inspect_repository_metadata
    from local_lm.model_planner import persist_install_plan, resolve_install_plan

    content = safetensors_bytes(["lora_unet_block.lora_down.weight"])
    digest = hashlib.sha256(content).hexdigest()
    filename = "detail.safetensors"
    revision = "d" * 40
    planned = resolve_install_plan(
        remote_id="synthetic/neutral-detail",
        revision=revision,
        role="image",
        engine="comfyui",
        selected_files=[{"filename": filename, "size": len(content), "sha256": digest}],
        inspection=inspect_repository_metadata({filename: content}, [filename], role="image"),
        comfy_paths={"loras": "."},
        auxiliary_kind="lora",
    )
    with SessionLocal() as session:
        plan = persist_install_plan(session, planned)
        session.commit()
        download_plan_id = plan.id
    source_runtime["LoraLoader"] = {
        "python_module": "nodes",
        "input": {"required": {"lora_name": [[filename]]}},
        "input_order": {"required": ["lora_name"]},
        "output": [],
    }
    graph = _ui_graph()
    graph["nodes"].append(
        {
            "id": 3,
            "type": "LoraLoader",
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "widgets_values": [filename],
            "properties": {"cnr_id": "comfy-core"},
        }
    )
    source = await client.post(
        "/api/workflows/packages/install-plans",
        json={
            "name": "Neutral installed source",
            "operation": "text_to_image",
            "ui_graph": graph,
            "dependencies": {
                "version": 1,
                "slots": []
                if change == "undeclared"
                else [
                    {
                        "name": "detail",
                        "resource_kind": "model_asset",
                        "required": True,
                        "satisfaction": "all_of",
                        "requirements": [
                            {
                                "key": "detail",
                                "constraints": {"asset_kind": "lora", "sha256": digest},
                            }
                        ],
                    }
                ],
            },
            "selections": [
                {
                    "reference_filename": filename,
                    "install_plan_id": download_plan_id,
                    "artifact_path": filename,
                }
            ],
        },
    )
    assert source.status_code == 201 and source.json()["can_accept"], source.text
    services = app.state.services
    manager = services.downloads
    monkeypatch.setattr(services.engines.settings, "media_engine", "comfyui")
    monkeypatch.setattr(manager, "start", lambda _job: None)
    monkeypatch.setattr(
        manager,
        "_api",
        SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename=filename, size=len(content), lfs={"sha256": digest})
                ],
                sha=revision,
                pipeline_tag=None,
                tags=["lora"],
                gated=False,
            )
        ),
    )

    async def download_file(**kwargs: Any) -> str:
        target = kwargs["staging"] / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    monkeypatch.setattr(manager, "_download_file", download_file)
    accepted = await client.post(f"/api/workflow-install-offers/{source.json()['id']}/install")
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()[0]["id"]
    await asyncio.gather(*list(manager._offer_tasks.values()))
    original_publish = services.events.publish

    async def publish(event: str, identifier: str, data: dict[str, Any]) -> None:
        if event == "download.completed" and identifier == job_id:
            with SessionLocal() as session:
                job = session.get(models.Job, job_id)
                assert job is not None
                asset = session.get(models.ModelAssetInstall, job.result_json["model_asset_id"])
                assert asset is not None
                if change == "bytes":
                    (Path(asset.local_path) / filename).write_bytes(content + b"changed")
                elif change == "request":
                    job.payload_json = {**job.payload_json, "revision": "e" * 40}
                    session.commit()
        await original_publish(event, identifier, data)

    monkeypatch.setattr(services.events, "publish", publish)
    await manager._download(job_id)
    with SessionLocal() as session:
        job = session.get(models.Job, job_id)
        assert job is not None
        assert job.status == "complete", job.error
        offer = session.scalar(select(models.WorkflowInstallOffer))
        assert offer is not None
        offer_id = offer.id
        asset_id = job.result_json["model_asset_id"]
        if change in {"none", "undeclared"}:
            assert offer.status == "completed", offer.completion_error_code
            bindings = list(
                session.scalars(
                    select(models.WorkflowDependencyBinding).where(
                        models.WorkflowDependencyBinding.workflow_revision_id
                        == offer.workflow_revision_id
                    )
                )
            )
            assert any(binding.model_asset_install_id == asset_id for binding in bindings)
            compiled = session.get(models.WorkflowRevision, offer.workflow_revision_id)
            assert compiled is not None
            assert {slot["resource_kind"] for slot in compiled.dependencies_json["slots"]} == {
                "model_asset",
                "runtime",
            }
            if change == "undeclared":
                original = session.get(models.WorkflowPackageInstallPlan, offer.source_plan_id)
                assert original is not None and original.request_json["dependencies"]["slots"] == []
    if change not in {"none", "undeclared"}:
        _assert_unfinished(offer_id)
