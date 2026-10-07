from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy import delete, select, text
from test_comfy_registry_launch_verification import _add_active
from test_comfy_registry_reviewed_batches import _watch_files
from test_registry_reviewed_activation_endpoints import package as package

from local_lm import comfy_registry_archives as archives
from local_lm import comfy_registry_installs as installs
from local_lm import workflow_review_runtime as reviews
from local_lm.adapters.base import MediaEvent
from local_lm.comfy_registry_target_verification import ComfyRegistryVerificationTarget
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    ComfyRegistryInstall,
    ComfyRegistrySourceArtifactReview,
    Job,
    Run,
    WorkflowRevision,
)
from local_lm.scheduler import JobClaim
from local_lm.schemas import TurnRequest
from local_lm.workflow_revision_reviews import WorkflowReviewError


@pytest.fixture
async def workflow(
    package: dict[str, Any],
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[dict[str, Any]]:
    response = await client.post(package["url"] + "/review", json={"trusted": True})
    assert response.status_code == 200, response.text
    response = await client.post(package["url"] + "/activate")
    assert response.status_code == 200, response.text
    services = app.state.services
    original_statuses = services.processes.statuses
    monkeypatch.setattr(
        services.processes,
        "statuses",
        lambda: [
            worker.model_copy(
                update={"managed": True, "running": True, "state": "ready", "pid": 43210}
            )
            if worker.name == "media"
            else worker
            for worker in original_statuses()
        ],
    )
    selected = package["inputs"]["selected"]
    kind = selected.node_types[0]

    async def info() -> dict[str, Any]:
        return {
            kind: {"python_module": "custom_nodes.neutral", "input": {"required": {}}, "output": []}
        }

    monkeypatch.setattr(services.engines.media, "object_info", info, raising=False)
    response = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral reviewed source workflow",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": {"1": {"class_type": kind, "inputs": {}}},
            "dependencies": {
                "registry_packages": [
                    {
                        "package_id": selected.package_id,
                        "package_version": selected.declared_version,
                    }
                ]
            },
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    url = f"/api/workflows/{body['id']}/revisions/{body['current_revision_id']}/review"
    preview = await client.get(url)
    assert preview.status_code == 200 and preview.json()["can_approve"], preview.text
    yield dict(
        package=package,
        revision=body["current_revision_id"],
        url=url,
        subject=preview.json()["subject_sha256"],
        services=services,
    )


async def approve(client: AsyncClient, workflow: dict[str, Any]) -> Any:
    return await client.post(
        workflow["url"], json={"action": "approve", "subject_sha256": workflow["subject"]}
    )


def revoke() -> None:
    with SessionLocal() as session:
        session.execute(delete(ComfyRegistrySourceArtifactReview))
        session.commit()


def trusted(workflow: dict[str, Any]) -> bool:
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, workflow["revision"])
        assert revision is not None
        return revision.trusted


@pytest.mark.parametrize("package", ["active", "inactive", "empty"], indirect=True)
async def test_revision_review_and_dispatch_verify_current_selected_source_packages(
    workflow: dict[str, Any],
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _watch_files(monkeypatch)
    response = await approve(client, workflow)
    assert response.status_code == 200, response.text
    assert trusted(workflow)
    services = workflow["services"]
    verified = await reviews.verify_workflow_review_runtime(
        settings, services.processes, services.engines.media, SessionLocal, workflow["revision"]
    )
    assert verified is not None and verified.packages is not None and seen
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, workflow["revision"])
        assert revision is not None
        reviews.revalidate_workflow_review_runtime(session, services.processes, revision, verified)
    with SessionLocal() as session:
        install = session.get(ComfyRegistryInstall, workflow["package"]["preparation"].install_id)
        assert install is not None
        install.trusted = False
        session.commit()
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, workflow["revision"])
        assert revision is not None
        with pytest.raises(WorkflowReviewError):
            reviews.revalidate_workflow_review_runtime(
                session, services.processes, revision, verified
            )


async def test_revision_review_excludes_unselected_active_packages(
    workflow: dict[str, Any],
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        selected = session.get(ComfyRegistryInstall, workflow["package"]["preparation"].install_id)
        assert selected is not None
        _add_active(session, selected)
        other = session.get(ComfyRegistryInstall, "other-registry-install")
        assert other is not None
        other.node_types_json = ["OtherNeutralNode"]
        session.commit()
    response = await approve(client, workflow)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("boundary", ["during-info", "before-commit"])
async def test_revision_approval_refuses_revocation_after_physical_verification(
    workflow: dict[str, Any],
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    original = reviews.VerifiedReviewedPackages.refresh

    async def refresh(self: reviews.VerifiedReviewedPackages) -> reviews.VerifiedReviewedPackages:
        if boundary == "during-info":
            revoke()
        result = await original(self)
        if boundary == "before-commit":
            revoke()
        return result

    monkeypatch.setattr(reviews.VerifiedReviewedPackages, "refresh", refresh)
    response = await approve(client, workflow)
    assert response.status_code == 409, response.text
    assert not trusted(workflow)


async def test_revision_approval_refreshes_files_after_runtime_inspection(
    workflow: dict[str, Any],
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = workflow["services"].engines.media
    original = media.object_info
    calls = 0

    async def info() -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            path = next(workflow["package"]["context"].custom_node_root.glob("*/__init__.py"))
            path.write_text("raise RuntimeError('Changed neutral code')", encoding="utf-8")
        return await original()

    monkeypatch.setattr(media, "object_info", info)
    response = await approve(client, workflow)
    assert response.status_code == 409, response.text
    assert not trusted(workflow)


async def test_revision_verification_refuses_a_changed_selected_package_snapshot(
    workflow: dict[str, Any],
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ComfyRegistryVerificationTarget.verify

    async def verify(self: ComfyRegistryVerificationTarget, *args: Any, **kwargs: Any) -> Any:
        with SessionLocal() as session:
            install = session.get(
                ComfyRegistryInstall, workflow["package"]["preparation"].install_id
            )
            assert install is not None
            install.review_json = {**install.review_json, "reviewed_at": "changed-neutral-review"}
            session.commit()
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ComfyRegistryVerificationTarget, "verify", verify)
    response = await approve(client, workflow)
    assert response.status_code == 409 and not trusted(workflow)


async def test_cancelled_revision_verification_drains_its_file_worker_before_releasing_the_lease(
    workflow: dict[str, Any],
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    acquired = asyncio.Event()
    original = archives.verify_staged_comfy_registry_archive

    def paused(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        try:
            assert release.wait(30)
            return original(*args, **kwargs)
        finally:
            finished.set()

    async def reached() -> bool:
        return entered.is_set()

    async def take_lease() -> None:
        async with workflow["services"].scheduler.lease("primary"):
            acquired.set()

    monkeypatch.setattr(installs, "verify_staged_comfy_registry_archive", paused)
    task = asyncio.create_task(approve(client, workflow))
    waiter = None
    try:
        await wait_until(reached, bool, what="workflow review file verification")
        waiter = asyncio.create_task(take_lease())
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not acquired.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await waiter
        assert finished.is_set() and not trusted(workflow)
    finally:
        release.set()
        if entered.is_set():
            await asyncio.to_thread(finished.wait, 30)
        await asyncio.gather(
            task, *([waiter] if waiter is not None else []), return_exceptions=True
        )


@pytest.mark.parametrize("change", ["none", "files", "authority"])
async def test_generation_refreshes_review_after_request_preparation_without_holding_a_writer(
    workflow: dict[str, Any],
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    response = await approve(client, workflow)
    assert response.status_code == 200, response.text
    services = app.state.services
    orchestrator = services.orchestrator
    calls = []
    original = orchestrator._geometry_binding_confirmed

    def geometry(*args: Any, **kwargs: Any) -> Any:
        with SessionLocal() as writer:
            writer.connection().exec_driver_sql("PRAGMA busy_timeout=100")
            writer.execute(text("UPDATE comfy_registry_installs SET active = active WHERE 0"))
            writer.commit()
        if change == "files":
            path = next(workflow["package"]["context"].custom_node_root.glob("*/__init__.py"))
            path.write_text("raise RuntimeError('Changed neutral code')", encoding="utf-8")
        return original(*args, **kwargs)

    original_refresh = reviews.VerifiedReviewedPackages.refresh
    refresh_calls = 0

    async def refresh(self: reviews.VerifiedReviewedPackages) -> reviews.VerifiedReviewedPackages:
        nonlocal refresh_calls
        result = await original_refresh(self)
        refresh_calls += 1
        if change == "authority" and refresh_calls == 2:
            revoke()
        return result

    async def generate(request: Any) -> Any:
        calls.append(request)
        yield MediaEvent(type="cancelled")

    monkeypatch.setattr(orchestrator, "_geometry_binding_confirmed", geometry)
    monkeypatch.setattr(reviews.VerifiedReviewedPackages, "refresh", refresh)
    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)
    chat = (await client.post("/api/chats", json={"title": "Neutral dispatch review"})).json()
    async with services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(text="A neutral picture.", mode="image"),
                freeze_context=False,
                activate_branch=False,
            )
            run_id = accepted.run.id
            run = session.get(Run, run_id)
            assert run is not None
            run.workflow_revision_id = workflow["revision"]
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            claim = JobClaim(token="neutral-review-dispatch", attempt=1)
            job.status = "running"
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id = job.id
            session.commit()
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None and run.workflow_revision_id == workflow["revision"]
        if change == "none":
            await orchestrator._execute_media(job_id, run_id, claim)
            assert len(calls) == 1
            assert calls[0].workflow == {
                "1": {
                    "class_type": workflow["package"]["inputs"]["selected"].node_types[0],
                    "inputs": {},
                }
            }
            assert refresh_calls == 2
        else:
            with pytest.raises(WorkflowReviewError):
                await orchestrator._execute_media(job_id, run_id, claim)
            assert not calls
