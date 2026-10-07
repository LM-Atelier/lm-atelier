"""Registry preparation keeps installation policy and ownership across its lifecycle."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from sqlalchemy.exc import IntegrityError
from test_install_queue_recovery import control
from test_registry_install_endpoints import _seed_install, _write_install_paths
from test_workflow_package_prepare_endpoint import _workflow

from local_lm import api as api_module
from local_lm.comfy_registry import ComfyNodeResolution
from local_lm.comfy_registry_lifecycle import ComfyRegistryPreparation
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ComfyRegistryInstall, Job
from local_lm.queue_lane_policy import read_lane_policy
from local_lm.scheduler import ResourceScheduler
from local_lm.workflow_package_activation import WorkflowPackageActivation
from local_lm.workflow_package_preparation import WorkflowPackagePreparationError


@pytest.fixture(autouse=True)
def configured_registry(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"")
    monkeypatch.setattr(settings, "comfy_executable", executable)
    monkeypatch.setattr(settings, "comfy_directory", tmp_path / "ComfyUI")


async def accept(client: AsyncClient) -> str:
    response = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "example-pack", "version": "1.2.3", "ui_graph": _workflow()},
    )
    assert response.status_code == 202, response.text
    return str(response.json()["id"])


def state(job_id: str) -> tuple[str, str | None, int, str | None]:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        return job.status, job.claim_owner, job.attempt, job.phase


async def paused(job_id: str) -> None:
    async with asyncio.timeout(10):
        while state(job_id)[3] != "install paused":
            await asyncio.sleep(0.01)
    assert state(job_id)[:3] == ("queued", None, 0)


@pytest.mark.parametrize("restart", [False, True])
async def test_queued_registry_preparation_waits_for_resume_after_restart(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, restart: bool
) -> None:
    entered = asyncio.Event()

    async def refuse(*_args: object, **_kwargs: object) -> None:
        entered.set()
        raise WorkflowPackagePreparationError("interpreter_probe_unavailable", "Probe unavailable")

    monkeypatch.setattr(api_module, "prepare_workflow_package", refuse)
    assert control("pause_after_current", 0).dispatch_state == "paused"
    job_id = await accept(client)
    await paused(job_id)

    async def resume() -> None:
        assert not entered.is_set()
        assert state(job_id)[:3] == ("queued", None, 0)
        with SessionLocal() as session:
            policy = read_lane_policy(session, "install")
            assert policy.dispatch_state == "paused" and policy.running_jobs == 0
        control("resume", policy.revision)
        await app.state.services.scheduler.queue_control_changed("install")
        await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
        async with asyncio.timeout(10):
            while state(job_id)[:3] != ("failed", None, 1):
                await asyncio.sleep(0.01)
        assert state(job_id)[:3] == ("failed", None, 1)

    try:
        if restart:
            await api_module.shutdown_registry_preparations()
            async with app.router.lifespan_context(app):
                await resume()
        else:
            await resume()
    finally:
        await api_module.shutdown_registry_preparations()


@pytest.mark.parametrize("ending", ["failed", "cancelled", "interrupted"])
async def test_retry_keeps_accepted_registry_inputs_and_waits_for_installation_resume(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    entered = asyncio.Event()
    calls: list[tuple[object, object, object]] = []

    async def prepare(*_args: object, **kwargs: object) -> None:
        calls.append((kwargs["package_id"], kwargs["version"], kwargs["node_types"]))
        entered.set()
        if ending == "failed" or len(calls) > 1:
            raise WorkflowPackagePreparationError("probe_unavailable", "Probe unavailable")
        await asyncio.Event().wait()

    monkeypatch.setattr(api_module, "prepare_workflow_package", prepare)
    job_id = await accept(client)
    await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
    if ending == "cancelled":
        cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
    elif ending == "interrupted":
        await api_module.shutdown_registry_preparations()
    else:
        async with asyncio.timeout(10):
            while job_id in api_module._REGISTRY_PREPARE_TASKS:
                await asyncio.sleep(0.01)
    assert state(job_id)[:3] == (ending, None, 1)
    policy = control("pause_after_current", 0)
    try:
        response = await client.post(f"/api/jobs/{job_id}/retry")
        assert response.status_code == 200, response.text
        await paused_retry(job_id)
        assert len(calls) == 1
        assert response.json()["payload_json"] == {
            "package_id": "example-pack",
            "version": "1.2.3",
            "node_types": ["ExampleNode"],
        }
        duplicate = await client.post(f"/api/jobs/{job_id}/retry")
        assert duplicate.status_code == 409
        control("resume", policy.revision)
        await app.state.services.scheduler.queue_control_changed("install")
        async with asyncio.timeout(10):
            while state(job_id)[:3] != ("failed", None, 2):
                await asyncio.sleep(0.01)
        assert calls == [("example-pack", "1.2.3", ("ExampleNode",))] * 2
    finally:
        await api_module.shutdown_registry_preparations()


async def paused_retry(job_id: str) -> None:
    async with asyncio.timeout(10):
        while state(job_id)[3] != "install paused":
            await asyncio.sleep(0.01)
    assert state(job_id)[:3] == ("queued", None, 1)


@pytest.mark.parametrize(
    "damage", ["missing-inputs", "unknown-inputs", "retained-claim", "non-object"]
)
async def test_registry_retry_refuses_invalid_inputs_or_unfinished_cleanup(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    async def prepare(*_args: object, **_kwargs: object) -> None:
        raise WorkflowPackagePreparationError("probe_unavailable", "Probe unavailable")

    monkeypatch.setattr(api_module, "prepare_workflow_package", prepare)
    job_id = await accept(client)
    async with asyncio.timeout(10):
        while state(job_id)[:3] != ("failed", None, 1):
            await asyncio.sleep(0.01)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        payload = dict(job.payload_json)
        if damage == "missing-inputs":
            del payload["node_types"]
        elif damage == "unknown-inputs":
            payload["unrecognized"] = "neutral-marker"
        elif damage == "retained-claim":
            job.claim_owner = "cleaning-up"
        # Deliberately not an object, which the database must refuse.
        job.payload_json = cast(Any, []) if damage == "non-object" else payload
        saved_payload = job.payload_json
        if damage == "non-object":
            with pytest.raises(IntegrityError, match="artifact JSON reference is invalid"):
                session.commit()
            session.rollback()
            assert state(job_id)[0] == "failed"
            return
        session.commit()
    before = state(job_id)
    response = await client.post(f"/api/jobs/{job_id}/retry")
    assert response.status_code == 409, response.text
    assert state(job_id) == before
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None and job.payload_json == saved_payload


@pytest.mark.parametrize("replaced", ["none", "claim", "retry"])
async def test_a_failed_registry_claim_settles_only_its_original_queued_attempt(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, replaced: str
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def fail(*_args: object, **_kwargs: object) -> None:
        entered.set()
        await release.wait()
        raise RuntimeError("Queue unavailable")

    monkeypatch.setattr(app.state.services.scheduler, "_acquire_job", fail)
    job_id = await accept(client)
    await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    if replaced != "none":
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None
            job.status = "running" if replaced == "claim" else "queued"
            job.claim_owner = "replacement-claim" if replaced == "claim" else None
            job.attempt = 1 if replaced == "claim" else 0
            job.queue_ticket = "replacement-ticket"
            job.phase = "Current phase"
            job.error = "Current error"
            session.commit()
    try:
        release.set()
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None
            if replaced != "none":
                assert (job.status, job.claim_owner, job.attempt) == (
                    ("running", "replacement-claim", 1)
                    if replaced == "claim"
                    else ("queued", None, 0)
                )
                assert job.queue_ticket == "replacement-ticket"
                assert job.phase == "Current phase" and job.error == "Current error"
            else:
                assert (job.status, job.claim_owner, job.attempt) == ("failed", None, 0)
                assert job.completed_at is not None
                assert job.error == "Package preparation could not start. Try again."
    finally:
        release.set()
        await api_module.shutdown_registry_preparations()


@pytest.mark.parametrize("outcome", ["complete", "failed"])
async def test_a_failed_registry_release_retains_its_result_until_claim_recovery(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    entered, finish = asyncio.Event(), asyncio.Event()
    scheduler = app.state.services.scheduler

    async def prepare(*_args: object, **_kwargs: object) -> ComfyRegistryPreparation:
        entered.set()
        await finish.wait()
        if outcome == "failed":
            raise WorkflowPackagePreparationError("probe_unavailable", "Probe unavailable")
        return ComfyRegistryPreparation(
            install_id="accepted-install",
            installed_path="accepted-path",
            wheel_environment_path="accepted-environment",
            archive_sha256="a" * 64,
            manifest_sha256="b" * 64,
            wheel_closure_sha256="c" * 64,
            wheel_environment_sha256="d" * 64,
            reused_wheel_environment=False,
        )

    async def fail_release(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Release unavailable")

    monkeypatch.setattr(api_module, "prepare_workflow_package", prepare)
    original_release = scheduler._release_job
    monkeypatch.setattr(scheduler, "_release_job", fail_release)
    job_id = await accept(client)
    await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    assert control("pause_after_current", 0).dispatch_state == "draining"
    try:
        finish.set()
        await asyncio.wait_for(task, timeout=PATIENCE_SECONDS)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.status == outcome
            assert job.claim_owner is not None and job.completed_at is not None
            saved = (dict(job.payload_json), job.error, job.completed_at)
            if outcome == "complete":
                assert saved[0]["preparation"]["install_id"] == "accepted-install"
            else:
                assert saved[0]["error_code"] == "probe_unavailable"
                assert saved[1] == "Probe unavailable"
            assert read_lane_policy(session, "install").dispatch_state == "draining"
        assert not scheduler._lock("primary", 1).locked()
        retry = await client.post(f"/api/jobs/{job_id}/retry")
        assert retry.status_code == 409
        monkeypatch.setattr(scheduler, "_release_job", original_release)
        assert scheduler._expire_foreign_claims("primary") == [job_id]
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.status == outcome and job.claim_owner is None
            assert (job.payload_json, job.error, job.completed_at) == saved
            assert read_lane_policy(session, "install").dispatch_state == "paused"
        if outcome == "failed":
            retry = await client.post(f"/api/jobs/{job_id}/retry")
            assert retry.status_code == 200
            await paused_retry(job_id)
    finally:
        finish.set()
        monkeypatch.setattr(scheduler, "_release_job", original_release)
        await api_module.shutdown_registry_preparations()


@pytest.mark.parametrize("same_time", [False, True])
async def test_an_old_cancel_does_not_cancel_a_retry_accepted_after_cleanup(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, same_time: bool
) -> None:
    entered = asyncio.Event()
    accepted: list[str] = []

    async def prepare(*_args: object, **_kwargs: object) -> None:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(api_module, "prepare_workflow_package", prepare)
    job_id = await accept(client)
    await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
    assert control("pause_after_current", 0).dispatch_state == "draining"
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    with SessionLocal() as session:
        initial = session.get(Job, job_id)
        assert initial is not None and initial.enqueued_at is not None
        accepted_at = initial.enqueued_at

    def retry_after_release(_done: asyncio.Task[None]) -> None:
        if same_time:
            monkeypatch.setattr(api_module, "utcnow", lambda: accepted_at)
        with SessionLocal() as session:
            job = api_module._retry_registry_preparation(session, app.state.services, job_id)
            accepted.append(job.id)

    task.add_done_callback(retry_after_release)
    try:
        response = await client.post(f"/api/jobs/{job_id}/cancel")
        assert response.status_code == 409
        assert accepted == [job_id]
        assert state(job_id)[:3] == ("queued", None, 1)
        await paused_retry(job_id)
    finally:
        await api_module.shutdown_registry_preparations()


@pytest.mark.parametrize("restart", [False, True])
async def test_registry_renewal_preserves_its_install_and_pause_across_restart(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    restart: bool,
) -> None:
    _write_install_paths(settings, environment=False)
    install_id = _seed_install(trusted=True)
    calls: list[object] = []

    async def prepare(*_args: object, **kwargs: object) -> None:
        calls.append(kwargs["renew_install_id"])
        assert kwargs["authorized_workflow"] is None
        assert kwargs["on_prepared"] is None
        raise WorkflowPackagePreparationError("probe_unavailable", "Probe unavailable")

    monkeypatch.setattr(api_module, "prepare_workflow_package", prepare)
    policy = control("pause_after_current", 0)
    response = await client.post(f"/api/workflows/packages/installs/{install_id}/renew")
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    await paused(job_id)
    assert calls == []

    async def resume() -> None:
        assert calls == []
        control("resume", policy.revision)
        await app.state.services.scheduler.queue_control_changed("install")
        async with asyncio.timeout(10):
            while state(job_id)[:3] != ("failed", None, 1):
                await asyncio.sleep(0.01)
        assert calls == [install_id]
        with SessionLocal() as session:
            install = session.get(ComfyRegistryInstall, install_id)
            assert install is not None and install.trusted and not install.active

    try:
        if restart:
            await api_module.shutdown_registry_preparations()
            async with app.router.lifespan_context(app):
                await resume()
        else:
            await resume()
    finally:
        await api_module.shutdown_registry_preparations()


@pytest.mark.parametrize("shutdown", [False, True])
async def test_registry_cancellation_keeps_the_install_claim_until_restoration_finishes(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, shutdown: bool
) -> None:
    entered, restoring, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def preparing(*_args: object, **_kwargs: object) -> None:
        entered.set()
        await asyncio.Event().wait()

    async def stop(_name: str, **_kwargs: object) -> None:
        restoring.set()
        await release.wait()

    monkeypatch.setattr(api_module, "prepare_workflow_package", preparing)
    monkeypatch.setattr(app.state.services.processes, "stop", stop)
    job_id = await accept(client)
    await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
    assert control("pause_after_current", 0).dispatch_state == "draining"
    task = asyncio.create_task(
        api_module.shutdown_registry_preparations()
        if shutdown
        else client.post(f"/api/jobs/{job_id}/cancel")
    )
    try:
        await asyncio.wait_for(restoring.wait(), timeout=PATIENCE_SECONDS)
        assert not task.done()
        with SessionLocal() as session:
            policy = read_lane_policy(session, "install")
            assert policy.dispatch_state == "draining" and policy.running_jobs == 1
        assert state(job_id)[0] == "running" and state(job_id)[1] is not None
        release.set()
        response = await asyncio.wait_for(task, timeout=PATIENCE_SECONDS)
        if not shutdown:
            assert response is not None and response.status_code == 200
        assert state(job_id)[:3] == ("interrupted" if shutdown else "cancelled", None, 1)
        with SessionLocal() as session:
            policy = read_lane_policy(session, "install")
            assert policy.dispatch_state == "paused" and policy.running_jobs == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await api_module.shutdown_registry_preparations()


@pytest.mark.parametrize("damage", ["missing-nodes", "repeated-nodes", "unknown-field"])
async def test_restart_refuses_malformed_saved_registry_inputs_without_starting_work(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    calls: list[bool] = []

    async def prepare(*_args: object, **_kwargs: object) -> None:
        calls.append(True)

    monkeypatch.setattr(api_module, "prepare_workflow_package", prepare)
    control("pause_after_current", 0)
    job_id = await accept(client)
    await paused(job_id)
    await api_module.shutdown_registry_preparations()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        payload = dict(job.payload_json)
        if damage == "missing-nodes":
            del payload["node_types"]
        elif damage == "repeated-nodes":
            payload["node_types"] = ["ExampleNode", "ExampleNode"]
        else:
            payload["unrecognized"] = "neutral-marker"
        job.payload_json = payload
        session.commit()
    async with app.router.lifespan_context(app):
        assert state(job_id)[:3] == ("failed", None, 0)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.payload_json == payload
            assert job.error == "The accepted preparation inputs are unavailable."
            assert read_lane_policy(session, "install").dispatch_state == "paused"
        assert calls == []


@pytest.mark.parametrize("replacement", ["token", "attempt", "terminal"])
@pytest.mark.parametrize("write", ["progress", "success", "refusal", "error"])
async def test_a_departed_registry_attempt_cannot_overwrite_current_job_state(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, replacement: str, write: str
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare(*_args: object, **kwargs: object) -> ComfyRegistryPreparation:
        entered.set()
        await release.wait()
        if write == "progress":
            phase = kwargs["phase"]
            assert callable(phase)
            phase("Obsolete progress", 1, 2)
            raise asyncio.CancelledError
        if write == "refusal":
            raise WorkflowPackagePreparationError("probe_unavailable", "Obsolete refusal")
        if write == "error":
            raise RuntimeError("Obsolete failure")
        return ComfyRegistryPreparation(
            install_id="old-install",
            installed_path="old-path",
            wheel_environment_path="old-environment",
            archive_sha256="a" * 64,
            manifest_sha256="b" * 64,
            wheel_closure_sha256="c" * 64,
            wheel_environment_sha256="d" * 64,
            reused_wheel_environment=False,
        )

    monkeypatch.setattr(api_module, "prepare_workflow_package", prepare)
    job_id = await accept(client)
    await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None and job.claim_owner is not None and job.attempt == 1
        if replacement == "token":
            job.claim_owner = "replacement-claim"
        elif replacement == "attempt":
            job.attempt += 1
        else:
            job.status = "cancelled"
        job.error = "Current error"
        job.phase = "Current phase"
        expected_status = job.status
        expected_payload = dict(job.payload_json)
        session.commit()
    try:
        release.set()
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None
            assert job.status == expected_status
            assert job.phase == "Current phase"
            assert job.error == "Current error"
            assert job.payload_json == expected_payload
    finally:
        release.set()
        await api_module.shutdown_registry_preparations()


@pytest.mark.parametrize("boundary", ["progress", "activation"])
@pytest.mark.parametrize("replacement", ["expiry", "token", "attempt", "none"])
async def test_registry_preparation_cannot_continue_after_losing_its_claim(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, boundary: str, replacement: str
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    continued: list[str] = []
    preparation = ComfyRegistryPreparation(
        "neutral-install",
        "neutral-nodes",
        "neutral-environment",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        False,
    )

    async def activate(*_args: object, **_kwargs: object) -> WorkflowPackageActivation:
        continued.append("activation")
        return WorkflowPackageActivation("review_required", "review_required", "", ())

    async def prepare(*_args: object, **kwargs: object) -> ComfyRegistryPreparation:
        entered.set()
        await release.wait()
        if boundary == "progress":
            phase = kwargs["phase"]
            assert callable(phase)
            phase("Preparing dependencies", None, None)
            continued.append("progress")
        else:
            consumer = kwargs["on_prepared"]
            assert callable(consumer)
            with SessionLocal() as session:
                await consumer(
                    session,
                    preparation,
                    ComfyNodeResolution("example-pack", "1.2.3", ("ExampleNode",)),
                )
        return preparation

    monkeypatch.setattr(api_module, "prepare_workflow_package", prepare)
    monkeypatch.setattr(api_module, "activate_prepared_workflow_package", activate)
    job_id = await accept(client)
    await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
    task = api_module._REGISTRY_PREPARE_TASKS[job_id]
    try:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.status == "running" and job.attempt == 1
            if replacement == "expiry":
                job.claim_expires_at = utcnow() - timedelta(seconds=30)
            elif replacement == "token":
                job.claim_owner = "replacement-claim"
            elif replacement == "attempt":
                job.attempt += 1
            session.commit()
        if replacement == "expiry":
            assert ResourceScheduler()._expire_foreign_claims("primary") == [job_id]
        expected = state(job_id)
        release.set()
        await asyncio.wait_for(task, timeout=PATIENCE_SECONDS)
        if replacement == "none":
            assert continued == [boundary]
            assert state(job_id)[:3] == ("complete", None, 1)
        else:
            assert continued == []
            assert state(job_id)[0] == expected[0]
            assert state(job_id)[2] == expected[2]
    finally:
        release.set()
        await api_module.shutdown_registry_preparations()
