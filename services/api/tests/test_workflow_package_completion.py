"""Registry preparation completes through exact policy and runtime verification."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_comfy_registry_lifecycle import _ArchiveDownloader, _resolution, _WheelDownloader
from test_workflow_activation_api import created, url
from test_workflow_package_preparation import _Registry
from test_workflow_package_prepare_endpoint import _workflow
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm import api as api_module
from local_lm import workflow_package_preparation as preparation_module
from local_lm.comfy_registry_activation import review_comfy_registry_install
from local_lm.comfy_registry_lifecycle import prepare_comfy_registry_install
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_registry_wheel_artifacts import current_comfy_registry_wheel_target
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    ComfyRegistryInstall,
    Job,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowRevision,
)
from local_lm.registry_trust_policy import POLICY_ID
from local_lm.workflow_revision_reviews import build_review_snapshot, record_review

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "change",
    [
        "policy",
        "deprecated",
        "warning",
        "refused",
        "bytes",
        "identity",
        "activation-fails",
        "running",
        "lease",
        "cancel",
        "scoped",
        "scope-revoked",
        "scope-missing",
        "restore-fails",
        "cancel-restore",
        "stop-refused",
        "writer",
    ],
)
async def test_preparation_applies_the_policy_and_finishes_runtime_setup(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: str,
) -> None:
    runtime = tmp_path / "runtime"
    custom_nodes = runtime / "custom_nodes"
    custom_nodes.mkdir(parents=True)
    monkeypatch.setattr(settings, "comfy_directory", runtime)
    monkeypatch.setattr(settings, "comfy_executable", Path(sys.executable))
    resolution = _resolution(
        package_id="example-pack",
        warnings=("deprecated_version",)
        if change == "deprecated"
        else ("unknown_registry_notice",)
        if change == "warning"
        else (),
    )
    monkeypatch.setattr(api_module, "ComfyRegistryClient", lambda: _Registry(resolution))
    monkeypatch.setattr(api_module, "ComfyRegistryArchiveDownloader", _ArchiveDownloader)
    monkeypatch.setattr(api_module, "ComfyRegistryWheelDownloader", _WheelDownloader)

    async def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("A package without dependencies requested remote wheel metadata")

    monkeypatch.setattr(
        api_module, "ComfyRegistryWheelProjectClient", lambda: SimpleNamespace(fetch=no_network)
    )
    monkeypatch.setattr(
        api_module, "ComfyRegistryWheelMetadataClient", lambda: SimpleNamespace(fetch=no_network)
    )

    async def probe(_path: Path) -> tuple[dict[str, str], tuple[str, ...]]:
        return current_comfy_registry_wheel_target()

    monkeypatch.setattr(api_module, "probe_comfy_registry_runtime_target", probe)
    processes = app.state.services.processes
    original_statuses = processes.statuses
    media_status = next(status for status in processes.statuses() if status.name == "media")
    original_stop = processes.stop
    scoped = change in {"scoped", "scope-revoked", "restore-fails", "cancel-restore"}
    scope_sha256: str | None = None
    workflow_id = revision_id = ""
    if scoped:
        workflow_id, revision_id, payload = await created(client)
        activated = await client.post(url(workflow_id, revision_id), json=payload)
        assert activated.status_code == 200, activated.text
        with SessionLocal() as session:
            activation = session.get(WorkflowActivation, activated.json()["id"])
            assert activation is not None and activation.is_active
            scope_sha256 = activation.details_json["launch_sha256"]
    elif change == "scope-missing":
        scope_sha256 = "a" * 64
    initially_running = scoped or change in {"running", "lease", "scope-missing", "stop-refused"}
    running = initially_running
    calls: list[str] = []
    launch_scopes: list[Any] = []
    started = asyncio.Event()
    restoring = asyncio.Event()
    finish_restoring = asyncio.Event()
    writer_committed = False

    def independent_write() -> None:
        nonlocal writer_committed
        with SessionLocal() as writer:
            writer.add(Job(kind="chat", status="complete"))
            writer.commit()
        writer_committed = True

    def statuses() -> list[Any]:
        return [
            media_status.model_copy(
                update={"running": running, "state": "ready" if running else "stopped"}
            )
            if worker.name == "media"
            else worker
            for worker in original_statuses()
        ]

    async def stop(name: str) -> Any:
        nonlocal running
        if name != "media":
            return await original_stop(name)
        calls.append("stop")
        if change != "stop-refused":
            running = False
        return statuses()[0]

    async def start(*_args: object, **_kwargs: object) -> Any:
        nonlocal running
        calls.append("start")
        launch_scopes.append(_kwargs.get("activation_scope"))
        started.set()
        if change == "writer":
            await asyncio.to_thread(independent_write)
        if _kwargs.get("activation_scope") is not None:
            restoring.set()
            if change == "restore-fails":
                running = True
                raise RuntimeError("Synthetic restoration failure")
            if change == "cancel-restore":
                await finish_restoring.wait()
        if change == "activation-fails" and calls.count("start") == 1:
            raise RuntimeError("Synthetic runtime startup failure")
        if change == "cancel" and calls.count("start") == 1:
            await asyncio.Event().wait()
        running = True
        return statuses()[0]

    async def inventory() -> frozenset[str]:
        return frozenset({"ExampleNode"})

    monkeypatch.setattr(processes, "statuses", statuses)
    monkeypatch.setattr(processes, "stop", stop)
    monkeypatch.setattr(processes, "start_media", start)
    monkeypatch.setattr(processes, "comfy_node_inventory", inventory)
    monkeypatch.setattr(processes, "launch_scope_sha256", lambda _name: scope_sha256)
    prepare = prepare_comfy_registry_install

    async def prepare_then_change(session: Any, **kwargs: Any) -> Any:
        result = await prepare(session, **kwargs)
        install = session.get(ComfyRegistryInstall, result.install_id)
        assert install is not None
        if change == "refused":
            review_comfy_registry_install(
                session,
                install_id=install.id,
                trusted=False,
                custom_node_root=custom_nodes,
                environment_root=registry_wheel_environment_root(settings.registry_dir),
                media_worker_stopped=not running,
            )
        elif change == "bytes":
            (custom_nodes / install.installed_path / "__init__.py").write_bytes(b"changed")
        elif change == "identity":
            install.archive_sha256 = "b" * 64
            session.commit()
        elif change == "scope-revoked":
            # The HTTP review waits on this same lease. Inject a committed
            # revocation through its owning writer to test restoration drift.
            info = await app.state.services.engines.media.object_info()
            with SessionLocal() as changed_session:
                definition = changed_session.get(WorkflowDefinition, workflow_id)
                revision = changed_session.get(WorkflowRevision, revision_id)
                assert definition is not None and revision is not None
                snapshot = build_review_snapshot(
                    changed_session, definition, revision, object_info=info
                )
                record_review(changed_session, revision, snapshot, approved=False)
                changed_session.commit()
        return result

    monkeypatch.setattr(preparation_module, "prepare_comfy_registry_install", prepare_then_change)

    observed_tasks: dict[str, asyncio.Task[None]] = {}
    original_run = api_module._run_workflow_package_preparation

    async def observed_run(*args: Any, **kwargs: Any) -> None:
        task = asyncio.current_task()
        assert task is not None
        observed_tasks[args[1]] = task
        await original_run(*args, **kwargs)

    monkeypatch.setattr(api_module, "_run_workflow_package_preparation", observed_run)

    async def queue() -> tuple[str, asyncio.Task[None]]:
        response = await client.post(
            "/api/workflows/packages/prepare",
            json={"package_id": "example-pack", "version": "1.2.3", "ui_graph": _workflow()},
        )
        assert response.status_code == 202, response.text
        identity = response.json()["id"]
        return identity, api_module._REGISTRY_PREPARE_TASKS.get(identity) or observed_tasks[
            identity
        ]

    if change == "lease":
        async with app.state.services.scheduler.lease("primary"):
            job_id, task = await queue()
            for _ in range(5):
                await asyncio.sleep(0)
            assert not task.done()
            assert calls == []
            assert running
    else:
        job_id, task = await queue()
    if change == "cancel-restore":
        await asyncio.wait_for(restoring.wait(), 30)
        cancelled = asyncio.create_task(client.post(f"/api/jobs/{job_id}/cancel"))
        acquired = asyncio.Event()

        async def competing_generation() -> None:
            async with app.state.services.scheduler.lease("primary"):
                acquired.set()

        competitor = asyncio.create_task(competing_generation())
        try:
            for _ in range(5):
                await asyncio.sleep(0)
            task.cancel()
            for _ in range(5):
                await asyncio.sleep(0)
            assert not task.done() and not cancelled.done()
            assert not acquired.is_set(), "The primary lease escaped before restoration finished"
        finally:
            finish_restoring.set()
            result = await asyncio.wait_for(cancelled, 30)
            await asyncio.wait_for(competitor, 30)
        assert result.status_code == 200 and result.json()["status"] == "cancelled"
    elif change == "cancel":
        reached = asyncio.create_task(started.wait())
        try:
            done, _pending = await asyncio.wait(
                {task, reached},
                timeout=30,
                return_when=asyncio.FIRST_COMPLETED,
            )
            assert reached in done, "Preparation ended without reaching runtime activation"
            cancel_response = await client.post(f"/api/jobs/{job_id}/cancel")
            assert (
                cancel_response.status_code == 200
                and cancel_response.json()["status"] == "cancelled"
            )
            await asyncio.gather(task, return_exceptions=True)
        finally:
            reached.cancel()
            await asyncio.gather(reached, return_exceptions=True)
    else:
        await asyncio.wait_for(task, 30)

    failed = change in {
        "bytes",
        "identity",
        "activation-fails",
        "scope-revoked",
        "scope-missing",
        "restore-fails",
        "stop-refused",
    }
    paused = change in {"warning", "refused"}
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.status == (
            "cancelled"
            if change in {"cancel", "cancel-restore"}
            else "failed"
            if failed
            else "complete"
        ), job.error
        install = session.scalar(select(ComfyRegistryInstall))
        if change in {"scope-missing", "stop-refused"}:
            assert install is None
            assert "start" not in calls
            assert running
            assert job.payload_json["error_code"] == (
                "media_scope_unavailable" if change == "scope-missing" else "media_worker_running"
            )
            return
        assert install is not None
        if change in {"scope-revoked", "restore-fails", "cancel-restore"}:
            assert install.active and install.trusted, (job.error, job.payload_json)
            if change != "cancel-restore":
                assert job.payload_json["error_code"] == "media_restore_failed"
        elif change == "cancel" or failed:
            assert not install.active
        else:
            assert install.trusted is not paused
            assert install.active is not paused
            outcome = job.payload_json["activation"]
            assert outcome["state"] == ("review_required" if paused else "active")
            if paused:
                assert outcome["reason"] == (
                    "previously_refused" if change == "refused" else "unrecognised_warning"
                )
                assert outcome["explanation"]
                assert "start" not in calls
            else:
                assert install.review_json["trust_authority"] == POLICY_ID
                assert install.review_json["trusted_by_local_user"] is False
                assert outcome["notices"] == (
                    ["deprecated_version"] if change == "deprecated" else []
                )
                assert calls.count("start") >= 1
    assert running is (initially_running and change not in {"scope-revoked", "restore-fails"})
    if scoped and change != "scope-revoked":
        assert launch_scopes[-1] is not None
        assert launch_scopes[-1].launch_sha256 == scope_sha256
    elif change == "scope-revoked":
        assert len(launch_scopes) == 1, "Revoked setup was restarted without its original scope"
    if change == "writer":
        assert writer_committed, "Runtime startup prevented another connection from committing"
