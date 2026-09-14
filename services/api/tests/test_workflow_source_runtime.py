"""A source installation owns its temporary worker and restores the prior state."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import _approve, _plan, _state
from test_workflow_source_completion import source_runtime as source_runtime

from local_lm.processes import ProcessSupervisor
from local_lm.schemas import WorkerStatus

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("limit", ["slots", "requirements"])
async def test_source_installation_completes_at_the_dependency_limit(
    client: AsyncClient, app: FastAPI, limit: str
) -> None:
    from test_workflow_package_import_endpoint import _ui_graph
    from test_workflow_package_install_plans import _runtime_declaration

    from local_lm.db import SessionLocal
    from local_lm.models import WorkflowRevision
    from local_lm.workflow_dependencies import parse_workflow_dependency_contract

    response = await client.post(
        "/api/workflows/packages/install-plans",
        json={
            "name": "Neutral workflow at the dependency limit",
            "operation": "text_to_image",
            "ui_graph": _ui_graph(),
            "dependencies": _runtime_declaration(
                63 if limit == "slots" else 511, per_slot=1 if limit == "slots" else 64
            ),
            "selections": [],
        },
    )
    assert response.status_code == 201 and response.json()["can_accept"], response.text
    offer_id = await _approve(client, app, str(response.json()["id"]))
    status, _workflow_id, revision_id, code = _state(offer_id)
    assert status == "completed" and code is None
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        contract = parse_workflow_dependency_contract(revision.dependencies_json)
        if limit == "slots":
            assert len(contract.slots) == 64
        else:
            assert sum(len(slot.requirements) for slot in contract.slots) == 512


@pytest.mark.parametrize("unicode_text", [False, True])
async def test_source_installation_completes_at_the_serialized_dependency_limit(
    client: AsyncClient, app: FastAPI, unicode_text: bool
) -> None:
    from test_workflow_package_import_endpoint import _ui_graph
    from test_workflow_package_install_plans import _sized_optional_declaration

    from local_lm.db import SessionLocal
    from local_lm.models import WorkflowRevision
    from local_lm.workflow_dependencies import (
        MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES,
        canonical_workflow_dependency_json,
        parse_workflow_dependency_contract,
    )

    runtime_slot = {
        "name": "media_runtime_2",
        "resource_kind": "runtime",
        "required": True,
        "satisfaction": "all_of",
        "requirements": [{"key": "installed", "constraints": {"engine": "comfyui"}}],
    }
    declaration = _sized_optional_declaration(
        MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES
        - len(canonical_workflow_dependency_json(runtime_slot))
        - 1,
        unicode_text=unicode_text,
    )
    response = await client.post(
        "/api/workflows/packages/install-plans",
        json={
            "name": "Neutral workflow at the serialized dependency limit",
            "operation": "text_to_image",
            "ui_graph": _ui_graph(),
            "dependencies": declaration,
            "selections": [],
        },
    )
    assert response.status_code == 201 and response.json()["can_accept"], response.text
    offer_id = await _approve(client, app, str(response.json()["id"]))
    status, _workflow_id, revision_id, code = _state(offer_id)
    assert status == "completed" and code is None
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        assert (
            len(canonical_workflow_dependency_json(revision.dependencies_json))
            == MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES
        )
        contract = parse_workflow_dependency_contract(revision.dependencies_json)
        assert len(contract.slots) == 2
        assert revision.dependencies_json["slots"] == [*declaration["slots"], runtime_slot]


@pytest.mark.parametrize("initially_running", [False, True])
@pytest.mark.parametrize("validation_fails", [False, True])
async def test_source_installation_owns_temporary_worker_and_restores_previous_state(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    initially_running: bool,
    validation_fails: bool,
) -> None:
    services = app.state.services
    processes: ProcessSupervisor = services.processes
    plan_id = await _plan(client)
    original = processes.statuses
    original_stop = processes.stop
    original_start = processes.start_media
    running = initially_running
    events: list[str] = []

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
            for worker in original()
        ]

    async def stop(name: str) -> WorkerStatus:
        nonlocal running
        if name != "media":
            return await original_stop(name)
        events.append("stop")
        running = False
        await original_stop(name)
        return next(worker for worker in statuses() if worker.name == "media")

    async def start(*args: Any, **kwargs: Any) -> WorkerStatus:
        nonlocal running
        assert not running
        scope = kwargs.get("activation_scope")
        events.append("start-source" if scope is not None else "restore")
        if scope is not None:
            assert scope.offer_id
            assert scope.models == () and scope.assets == () and scope.registry_packages == ()
        running = True
        await original_start(*args, **kwargs)
        return next(worker for worker in statuses() if worker.name == "media")

    async def validate(_graph: dict[str, Any]) -> list[str]:
        assert running
        events.append("validate")
        return ["The constructed workflow could not be validated."] if validation_fails else []

    monkeypatch.setattr(processes, "statuses", statuses)
    monkeypatch.setattr(processes, "stop", stop)
    monkeypatch.setattr(processes, "start_media", start)
    monkeypatch.setattr(services.engines.media, "validate_workflow", validate)
    offer_id = await _approve(client, app, plan_id)
    assert _state(offer_id)[0] == ("queued" if validation_fails else "completed")
    assert events == (["stop"] if initially_running else []) + [
        "start-source",
        "validate",
        "stop",
    ] + (["restore"] if initially_running else [])
    assert running is initially_running


@pytest.mark.parametrize("failure", ["stop", "restart"])
@pytest.mark.parametrize("validation_fails", [False, True])
async def test_restoration_failure_preserves_only_a_committed_installation(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    validation_fails: bool,
) -> None:
    from local_lm.db import SessionLocal
    from local_lm.models import WorkflowActivation, WorkflowInstallOffer
    from local_lm.workflow_completion_jobs import WorkflowCompletionResult, workflow_completion_job

    services = app.state.services
    processes: ProcessSupervisor = services.processes
    plan_id = await _plan(client)
    original_statuses = processes.statuses
    original_stop = processes.stop
    original_publish = services.events.publish
    original_start = processes.start_media
    running = failure == "restart"
    validated = False
    fault_enabled = True
    published: list[str] = []

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
        nonlocal running
        if name != "media":
            return await original_stop(name)
        if validated and failure == "stop" and fault_enabled:
            raise OSError("Neutral worker stop failure")
        running = False
        await original_stop(name)
        return next(worker for worker in statuses() if worker.name == "media")

    async def start(*_args: Any, **kwargs: Any) -> WorkerStatus:
        nonlocal running
        if kwargs.get("activation_scope") is None and failure == "restart":
            raise OSError("Neutral worker restart failure")
        running = True
        await original_start(*_args, **kwargs)
        return next(worker for worker in statuses() if worker.name == "media")

    async def validate(_graph: dict[str, Any]) -> list[str]:
        nonlocal validated
        validated = True
        return ["The constructed workflow is invalid."] if validation_fails else []

    async def publish(kind: str, *args: Any, **kwargs: Any) -> Any:
        if kind.startswith("workflow.install."):
            published.append(kind)
        return await original_publish(kind, *args, **kwargs)

    monkeypatch.setattr(processes, "statuses", statuses)
    monkeypatch.setattr(processes, "stop", stop)
    monkeypatch.setattr(processes, "start_media", start)
    monkeypatch.setattr(services.engines.media, "validate_workflow", validate)
    monkeypatch.setattr(services.events, "publish", publish)
    try:
        offer_id = await _approve(client, app, plan_id)
    finally:
        fault_enabled = False
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        job = workflow_completion_job(session, offer)
        result = WorkflowCompletionResult.model_validate(job.result_json)
        if validation_fails:
            assert offer.status == "queued" and job.status == "failed"
            assert result.activation_id is None
            assert published == ["workflow.install.attention"]
            assert offer.completion_error_code != "workflow-media-restore-failed"
        else:
            assert offer.status == "completed" and job.status == "complete"
            assert job.error is None and result.activation_id is not None
            activation = session.get(WorkflowActivation, result.activation_id)
            assert activation is not None
            assert activation.workflow_revision_id == offer.workflow_revision_id
            assert offer.completion_error_code == "workflow-media-restore-failed"
            assert published == ["workflow.install.completed"]
    progress = await client.get(f"/api/workflow-install-offers/{offer_id}/progress")
    assert progress.status_code == 200, progress.text
    if not validation_fails:
        assert progress.json()["phase"] == "completed"
        assert progress.json()["attention_code"] == "workflow-media-restore-failed"
    # The test worker also reflects the cleanup failure, rather than claiming restoration.
    assert running is (failure == "stop")


@pytest.mark.parametrize(
    "change", ["missing-runtime", "changed-runtime", "worker-scope", "worker-scope-after-bindings"]
)
async def test_final_activation_must_keep_the_resources_used_for_validation(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from dataclasses import replace

    from test_workflow_source_completion import _assert_unfinished

    from local_lm import workflow_source_completion
    from local_lm.workflow_activations import revalidate_workflow_activation

    processes: ProcessSupervisor = app.state.services.processes
    plan_id = await _plan(client)
    validated = False
    inspected = False

    async def validate(_graph: dict[str, Any]) -> list[str]:
        nonlocal validated
        validated = True
        if change == "worker-scope":
            monkeypatch.setattr(processes, "launch_scope_sha256", lambda _name: "b" * 64)
        return []

    def final_scope(*args: Any, **kwargs: Any) -> Any:
        nonlocal inspected
        inspected = True
        scope = revalidate_workflow_activation(*args, **kwargs)
        if change == "missing-runtime":
            return replace(scope, runtime_keys=(), runtimes=())
        if change == "changed-runtime":
            return replace(scope, runtimes=(replace(scope.runtimes[0], identity_json="{}"),))
        if change == "worker-scope-after-bindings":
            monkeypatch.setattr(processes, "launch_scope_sha256", lambda _name: "b" * 64)
        return scope

    monkeypatch.setattr(app.state.services.engines.media, "validate_workflow", validate)
    monkeypatch.setattr(
        workflow_source_completion, "revalidate_workflow_activation", final_scope, raising=False
    )
    offer_id = await _approve(client, app, plan_id)
    assert validated
    _assert_unfinished(offer_id)
    if change != "worker-scope":
        assert inspected
