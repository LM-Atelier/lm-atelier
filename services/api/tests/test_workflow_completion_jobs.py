"""An accepted workflow keeps one completion job through cancellation, retry and restart."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import _plan
from test_workflow_source_completion import source_runtime as source_runtime

from local_lm import models
from local_lm.db import SessionLocal

pytestmark = pytest.mark.asyncio


async def _accept(client: AsyncClient) -> tuple[str, str, str]:
    plan_id = await _plan(client)
    response = await client.post(f"/api/workflow-install-offers/{plan_id}/install")
    assert response.status_code == 202, response.text
    assert len(response.json()) == 1 and response.json()[0]["kind"] == "workflow_install"
    job_id = response.json()[0]["id"]
    with SessionLocal() as session:
        offer = session.scalar(
            select(models.WorkflowInstallOffer).where(
                models.WorkflowInstallOffer.source_plan_id == plan_id
            )
        )
        assert offer is not None and offer.completion_job_id == job_id
        return plan_id, offer.id, job_id


async def _settled(app: FastAPI, offer_id: str) -> None:
    manager = app.state.services.downloads
    for _ in range(10):
        task = manager._offer_tasks.get(offer_id)
        if task is None:
            return
        await asyncio.wait_for(asyncio.shield(task), timeout=30)
        await asyncio.sleep(0)
    pytest.fail("Workflow installation did not finish its scheduled attempts.")


def _state(offer_id: str) -> tuple[str, str, str, dict[str, Any], int]:
    with SessionLocal() as session:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert offer is not None
        job = session.get(models.Job, offer.completion_job_id)
        assert job is not None
        return offer.status, job.status, job.id, job.result_json, job.attempt


async def test_completion_and_its_runtime_result_commit_with_the_activation(
    client: AsyncClient,
    app: FastAPI,
) -> None:
    plan_id, offer_id, job_id = await _accept(client)
    await _settled(app, offer_id)
    offer_status, status, identifier, result, attempt = _state(offer_id)
    assert offer_status == "completed" and status == "complete" and identifier == job_id
    assert attempt == 1 and result["runtime_plan"]["operation"] == "reuse_configured"
    with SessionLocal() as session:
        activation = session.get(models.WorkflowActivation, result["activation_id"])
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert activation is not None and offer is not None
        assert activation.workflow_revision_id == offer.workflow_revision_id
    again = await client.post(f"/api/workflow-install-offers/{plan_id}/install")
    assert again.status_code == 202 and [item["id"] for item in again.json()] == [job_id]
    assert _state(offer_id) == (offer_status, status, identifier, result, attempt)


async def test_failed_validation_retries_the_same_job_and_keeps_its_runtime_result(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reject = True

    async def validate(_graph: dict[str, Any]) -> list[str]:
        return ["Neutral validation failure"] if reject else []

    monkeypatch.setattr(app.state.services.engines.media, "validate_workflow", validate)
    _plan_id, offer_id, job_id = await _accept(client)
    await _settled(app, offer_id)
    before = _state(offer_id)
    assert before[0:3] == ("queued", "failed", job_id)
    assert before[3]["activation_id"] is None and before[3]["runtime_plan"]
    reject = False
    retried = await client.post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 200 and retried.json()["id"] == job_id, retried.text
    await _settled(app, offer_id)
    after = _state(offer_id)
    assert after[0:3] == ("completed", "complete", job_id)
    assert after[3]["runtime_plan"] == before[3]["runtime_plan"] and after[4] > before[4]


async def test_cancelling_queued_installation_requires_explicit_retry(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = app.state.services.downloads
    with monkeypatch.context() as patch:
        patch.setattr(manager, "start_workflow_installation", lambda _id: None)
        _plan_id, offer_id, job_id = await _accept(client)
    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled", (
        cancelled.text
    )
    await manager.reconcile_workflow_install_offers(only_offer_id=offer_id)
    assert _state(offer_id)[0:3] == ("queued", "cancelled", job_id)
    retry = await client.post(f"/api/jobs/{job_id}/retry")
    assert retry.status_code == 200, retry.text
    await _settled(app, offer_id)
    assert _state(offer_id)[0:3] == ("completed", "complete", job_id)


async def test_cancel_and_immediate_retry_cannot_let_the_old_attempt_commit(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def validate(_graph: dict[str, Any]) -> list[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        return []

    monkeypatch.setattr(app.state.services.engines.media, "validate_workflow", validate)
    try:
        _plan_id, offer_id, job_id = await _accept(client)
        await asyncio.wait_for(entered.wait(), timeout=30)
        before = _state(offer_id)
        assert (await client.post(f"/api/jobs/{job_id}/cancel")).status_code == 200
        assert (await client.post(f"/api/jobs/{job_id}/retry")).status_code == 200
        assert _state(offer_id)[0:2] == ("queued", "queued")
    finally:
        release.set()
    await _settled(app, offer_id)
    assert calls == 2
    after = _state(offer_id)
    assert after[0:3] == ("completed", "complete", job_id) and after[4] > before[4]
    with SessionLocal() as session:
        assert len(list(session.scalars(select(models.WorkflowActivation)))) == 1


@pytest.mark.parametrize("operation", ["cancel", "retry"])
async def test_changed_completion_job_payload_refuses_control_before_any_work(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(app.state.services.downloads, "start_workflow_installation", lambda _id: None)
        _plan_id, offer_id, job_id = await _accept(client)
    with SessionLocal() as session:
        job = session.get(models.Job, job_id)
        assert job is not None
        job.status = "failed" if operation == "retry" else "queued"
        job.payload_json = {**job.payload_json, "source_plan_id": "another-source"}
        session.commit()
    response = await client.post(f"/api/jobs/{job_id}/{operation}")
    assert response.status_code == 409, response.text
    assert _state(offer_id)[0] == "queued"
    assert offer_id not in app.state.services.downloads._offer_tasks


async def test_interrupted_completion_resumes_its_existing_job_after_restart(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = app.state.services.downloads
    with monkeypatch.context() as patch:
        patch.setattr(manager, "start_workflow_installation", lambda _id: None)
        _plan_id, offer_id, job_id = await _accept(client)
    with SessionLocal() as session:
        job = session.get(models.Job, job_id)
        assert job is not None
        job.status = "running"
        session.commit()
    app.state.services.orchestrator.recover_interrupted()
    assert _state(offer_id)[1] == "interrupted"
    manager.recover_interrupted()
    await asyncio.wait_for(manager._offer_recovery_task, timeout=30)
    assert _state(offer_id)[0:3] == ("completed", "complete", job_id)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_restart_does_not_resume_explicitly_stopped_installation(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    manager = app.state.services.downloads
    with monkeypatch.context() as patch:
        patch.setattr(manager, "start_workflow_installation", lambda _id: None)
        _plan_id, offer_id, job_id = await _accept(client)
    with SessionLocal() as session:
        job = session.get(models.Job, job_id)
        assert job is not None
        job.status = status
        session.commit()
    manager.recover_interrupted()
    await asyncio.wait_for(manager._offer_recovery_task, timeout=30)
    assert _state(offer_id)[0:3] == ("queued", status, job_id)


async def test_completion_job_creation_rolls_back_with_the_approval(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import workflow_package_acceptance
    from local_lm.workflow_completion_jobs import stage_workflow_completion_job

    plan_id = await _plan(client)
    original = stage_workflow_completion_job

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        original(*args, **kwargs)
        raise ValueError("Neutral transaction interruption")

    monkeypatch.setattr(workflow_package_acceptance, "stage_workflow_completion_job", interrupted)
    response = await client.post(f"/api/workflow-install-offers/{plan_id}/install")
    assert response.status_code == 409, response.text
    with SessionLocal() as session:
        assert list(session.scalars(select(models.WorkflowInstallOffer))) == []
        assert list(session.scalars(select(models.Job))) == []
