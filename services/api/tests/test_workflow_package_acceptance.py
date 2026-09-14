"""Source approval commits the draft, accepted offer and exact jobs together."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_workflow_package_install_plans import _payload, configure_runtime

from local_lm import models
from local_lm.db import SessionLocal
from local_lm.schemas import DownloadRequest

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def runtime_inventory(app: FastAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_runtime(app.state.services.settings, tmp_path)
    monkeypatch.setattr(
        app.state.services.downloads, "start_workflow_installation", lambda _id: None, raising=False
    )

    async def object_info() -> dict[str, Any]:
        return {"LoraLoader": {}, "EmptyLatentImage": {}}

    monkeypatch.setattr(app.state.services.engines.media, "object_info", object_info, raising=False)


@pytest.fixture
def started(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    jobs: list[str] = []
    monkeypatch.setattr(app.state.services.downloads, "start", jobs.append)
    return jobs


async def _plan(client: AsyncClient, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = await client.post(
        "/api/workflows/packages/install-plans", json=payload if payload is not None else _payload()
    )
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


def _url(identifier: str) -> str:
    return f"/api/workflow-install-offers/{identifier}/install"


def _identities() -> dict[str, set[str]]:
    with SessionLocal() as session:
        return {
            model.__tablename__: set(session.scalars(select(model.id)))
            for model in (
                models.WorkflowDefinition,
                models.WorkflowRevision,
                models.WorkflowInstallOffer,
                models.WorkflowInstallOfferDownload,
                models.Job,
            )
        }


async def test_source_approval_is_committed_before_any_download_starts(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = await _plan(client)
    observed: list[str] = []

    def start(job_id: str) -> None:
        with SessionLocal() as session:
            job = session.get(models.Job, job_id)
            link = session.scalar(
                select(models.WorkflowInstallOfferDownload).where(
                    models.WorkflowInstallOfferDownload.job_id == job_id
                )
            )
            assert job is not None and link is not None
            offer = session.get(models.WorkflowInstallOffer, link.offer_id)
            assert offer is not None and offer.status == "queued"
            assert offer.source_plan_id == plan["id"]
            assert offer.offer_sha256 == link.offer_sha256 == plan["plan_sha256"]
            assert job.payload_json == link.request_json == plan["download_requests"][0]
            revision = session.get(models.WorkflowRevision, offer.workflow_revision_id)
            assert revision is not None and not revision.trusted and revision.api_graph_json == {}
            definition = session.get(models.WorkflowDefinition, revision.workflow_id)
            assert definition is not None and definition.current_revision_id == revision.id
            observed.append(job_id)

    monkeypatch.setattr(app.state.services.downloads, "start", start)
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 202, response.text
    assert observed == [response.json()[0]["id"]]
    progress = await client.get(f"/api/workflow-install-offers/{plan['id']}/progress")
    assert progress.status_code == 200, progress.text
    assert progress.json()["phase"] == "downloading"
    assert progress.json()["total_downloads"] == 1


async def test_overlapping_approvals_share_one_offer_and_job(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, started: list[str]
) -> None:
    plan = await _plan(client)
    arrived = 0
    both = asyncio.Event()

    async def object_info() -> dict[str, Any]:
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both.set()
        await asyncio.wait_for(both.wait(), timeout=5)
        return {"LoraLoader": {}, "EmptyLatentImage": {}}

    monkeypatch.setattr(app.state.services.engines.media, "object_info", object_info)
    responses = await asyncio.gather(client.post(_url(plan["id"])), client.post(_url(plan["id"])))
    assert [response.status_code for response in responses] == [202, 202]
    first, second = responses[0].json()[0], responses[1].json()[0]
    # SQLite reloads the same UTC timestamps without their timezone suffix.
    for key in ("created_at", "updated_at", "enqueued_at"):
        first[key] = first[key].removesuffix("Z")
        second[key] = second[key].removesuffix("Z")
    assert first == second
    with SessionLocal() as session:
        assert len(list(session.scalars(select(models.WorkflowInstallOffer)))) == 1
        assert len(list(session.scalars(select(models.WorkflowInstallOfferDownload)))) == 1
        assert sorted(item.kind for item in session.scalars(select(models.Job))) == [
            "download",
            "workflow_install",
        ]
    assert set(started) == {responses[0].json()[0]["id"]}


async def test_source_approval_preserves_an_existing_paused_download(
    client: AsyncClient, app: FastAPI, started: list[str]
) -> None:
    plan = await _plan(client)
    with SessionLocal() as session:
        job = app.state.services.downloads.stage(
            session, DownloadRequest.model_validate(plan["download_requests"][0])
        )
        job.status = "paused"
        session.commit()
        job_id = job.id
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 202, response.text
    assert response.json()[0]["id"] == job_id
    assert response.json()[0]["status"] == "paused"
    assert started == []
    repeated = (await client.post(_url(plan["id"]))).json()
    original = response.json()
    for items in (original, repeated):
        for item in items:
            for field in ("created_at", "updated_at", "enqueued_at"):
                if item[field] is not None:
                    item[field] = item[field].removesuffix("Z")
    assert repeated == original
    assert started == []


@pytest.mark.parametrize("status", ["complete", "failed", "cancelled"])
async def test_repeated_approval_does_not_redownload_a_terminal_job(
    client: AsyncClient, started: list[str], status: str
) -> None:
    plan = await _plan(client)
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 202, response.text
    job_id = response.json()[0]["id"]
    with SessionLocal() as session:
        job = session.get(models.Job, job_id)
        install = session.get(models.InstallPlan, "plan_lora")
        assert job is not None and install is not None
        job.status = status
        install.status = "activated"
        session.commit()
    started.clear()
    retry = await client.post(_url(plan["id"]))
    assert retry.status_code == 202, retry.text
    assert retry.json()[0]["id"] == job_id and retry.json()[0]["status"] == status
    assert started == []


@pytest.mark.parametrize("change", ["source", "artifact", "runtime"])
async def test_changed_preflight_refuses_without_creating_installation_work(
    client: AsyncClient, started: list[str], change: str
) -> None:
    plan = await _plan(client)
    before = _identities()
    with SessionLocal() as session:
        if change == "source":
            record = session.get(models.WorkflowPackageInstallPlan, plan["id"])
            assert record is not None
            record.request_json = {**record.request_json, "name": "Changed source"}
        else:
            install = session.get(models.InstallPlan, "plan_lora")
            assert install is not None
            if change == "artifact":
                artifacts = copy.deepcopy(install.artifacts_json)
                artifacts[0]["sha256"] = "b" * 64
                install.artifacts_json = artifacts
            else:
                install.runtime_contract_json = {
                    **install.runtime_contract_json,
                    "default_settings": {"steps": 5},
                }
        session.commit()
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "workflow-package-install-plan-changed"
    assert _identities() == before and started == []


async def test_failed_job_staging_rolls_back_the_source_draft_and_offer(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, started: list[str]
) -> None:
    plan = await _plan(client)
    before = _identities()
    manager = app.state.services.downloads
    original = manager.stage

    def interrupted(session: Session, request: DownloadRequest) -> models.Job:
        original(session, request)
        session.flush()
        raise RuntimeError("Neutral interrupted acceptance")

    monkeypatch.setattr(manager, "stage", interrupted)
    with pytest.raises(RuntimeError, match="Neutral interrupted acceptance"):
        await client.post(_url(plan["id"]))
    assert _identities() == before and started == []


async def test_core_only_source_acceptance_does_not_invent_a_download(
    client: AsyncClient, started: list[str]
) -> None:
    payload = _payload()
    payload["selections"] = []
    payload["ui_graph"]["nodes"][0]["type"] = "EmptyLatentImage"
    payload["ui_graph"]["nodes"][0]["widgets_values"] = [512, 512, 1]
    plan = await _plan(client, payload)
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 202, response.text
    assert [item["kind"] for item in response.json()] == ["workflow_install"]
    assert started == []
    progress = await client.get(f"/api/workflow-install-offers/{plan['id']}/progress")
    assert progress.status_code == 200, progress.text
    assert progress.json()["total_downloads"] == 0
    assert progress.json()["phase"] == "verifying"
    retry = await client.post(_url(progress.json()["id"]))
    assert retry.status_code == 202, retry.text


async def test_unresolved_extension_plan_refuses_before_draft_creation(
    client: AsyncClient, started: list[str]
) -> None:
    payload = _payload()
    payload["ui_graph"]["nodes"][0]["properties"] = {"cnr_id": "neutral-package", "ver": "1.2.3"}
    plan = await _plan(client, payload)
    assert "extension-plan-unavailable" in plan["blockers"]
    before = _identities()
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "workflow-package-install-plan-incomplete"
    assert _identities() == before and started == []


async def test_acceptance_can_select_an_operation_for_an_unaccepted_source_draft(
    client: AsyncClient, started: list[str]
) -> None:
    payload = _payload()
    draft_payload = {key: payload[key] for key in ("name", "operation", "ui_graph")}
    draft = await client.post("/api/workflows/packages/drafts", json=draft_payload)
    assert draft.status_code == 201, draft.text
    payload["operation"] = "image_to_image"
    plan = await _plan(client, payload)
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 202, response.text
    with SessionLocal() as session:
        definition = session.get(models.WorkflowDefinition, draft.json()["id"])
        assert definition is not None and definition.operation == "image_to_image"
    assert len(started) == 1


async def test_a_second_plan_cannot_overwrite_an_accepted_source_draft(
    client: AsyncClient, started: list[str]
) -> None:
    payload = _payload()
    first = await _plan(client, payload)
    response = await client.post(_url(first["id"]))
    assert response.status_code == 202, response.text
    before = _identities()
    second = await _plan(
        client, {**payload, "name": "Different installation", "operation": "image_to_image"}
    )
    started.clear()
    refused = await client.post(_url(second["id"]))
    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "workflow-package-installation-in-progress"
    assert _identities() == before and started == []


@pytest.mark.parametrize("change", ["job", "link", "draft", "offer"])
async def test_damaged_acceptance_is_not_hidden_by_a_retry(
    client: AsyncClient, started: list[str], change: str
) -> None:
    plan = await _plan(client)
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 202, response.text
    with SessionLocal() as session:
        offer = session.scalar(select(models.WorkflowInstallOffer))
        assert offer is not None
        if change == "job":
            job = session.get(models.Job, response.json()[0]["id"])
            assert job is not None
            job.payload_json = {**job.payload_json, "revision": "303"}
        elif change == "link":
            link = session.scalar(select(models.WorkflowInstallOfferDownload))
            assert link is not None
            link.request_sha256 = "b" * 64
        elif change == "offer":
            offer.workflow_artifact_sha256 = "b" * 64
        else:
            revision = session.get(models.WorkflowRevision, offer.workflow_revision_id)
            assert revision is not None
            revision.ui_graph_json = {"nodes": [], "links": []}
        session.commit()
    before = _identities()
    started.clear()
    retry = await client.post(_url(plan["id"]))
    assert retry.status_code == 409, retry.text
    assert retry.json()["code"] == "workflow-package-installation-changed"
    assert _identities() == before and started == []


@pytest.mark.parametrize("change", ["artifact", "runtime"])
async def test_repeated_acceptance_refuses_a_changed_download_plan(
    client: AsyncClient, started: list[str], change: str
) -> None:
    plan = await _plan(client)
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 202, response.text
    with SessionLocal() as session:
        install = session.get(models.InstallPlan, "plan_lora")
        assert install is not None
        if change == "artifact":
            artifacts = copy.deepcopy(install.artifacts_json)
            artifacts[0]["sha256"] = "b" * 64
            install.artifacts_json = artifacts
        else:
            install.runtime_contract_json = {
                **install.runtime_contract_json,
                "default_settings": {"steps": 12},
            }
        session.commit()
    started.clear()
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "workflow-package-install-plan-changed"
    assert started == []


async def test_repeated_acceptance_reuses_the_active_download_worker(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = await _plan(client)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []
    manager = app.state.services.downloads

    async def download(job_id: str) -> None:
        calls.append(job_id)
        entered.set()
        await release.wait()

    monkeypatch.setattr(manager, "_download", download)
    try:
        first = await client.post(_url(plan["id"]))
        assert first.status_code == 202, first.text
        await asyncio.wait_for(entered.wait(), timeout=5)
        active = manager._tasks[first.json()[0]["id"]]
        second = await client.post(_url(plan["id"]))
        assert second.status_code == 202, second.text
        assert second.json()[0]["id"] == first.json()[0]["id"]
        assert manager._tasks[first.json()[0]["id"]] is active
        assert calls == [first.json()[0]["id"]]
    finally:
        release.set()
        await asyncio.gather(*list(manager._tasks.values()))


@pytest.mark.parametrize("download_status", ["failed", "cancelled"])
async def test_failed_dependency_makes_installation_retryable_without_new_jobs(
    client: AsyncClient,
    app: FastAPI,
    started: list[str],
    download_status: str,
) -> None:
    plan = await _plan(client)
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 202, response.text
    jobs = {item["kind"]: item["id"] for item in response.json()}
    assert set(jobs) == {"download", "workflow_install"}
    with SessionLocal() as session:
        offer = session.scalar(
            select(models.WorkflowInstallOffer).where(
                models.WorkflowInstallOffer.source_plan_id == plan["id"]
            )
        )
        download = session.get(models.Job, jobs["download"])
        assert offer is not None and download is not None
        offer_id = offer.id
        download.status = download_status
        session.commit()
    before = _identities()
    await app.state.services.downloads.reconcile_workflow_install_offers(only_offer_id=offer_id)
    with SessionLocal() as session:
        completion = session.get(models.Job, jobs["workflow_install"])
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert completion is not None and completion.status == "failed"
        assert offer is not None and offer.completion_error_code == "workflow-download-failed"
    started.clear()
    retry = await client.post(f"/api/jobs/{jobs['workflow_install']}/retry")
    assert retry.status_code == 200, retry.text
    assert retry.json()["id"] == jobs["workflow_install"]
    assert retry.json()["status"] == "queued"
    assert started == [jobs["download"]]
    assert _identities() == before
    with SessionLocal() as session:
        download = session.get(models.Job, jobs["download"])
        assert download is not None and download.status == "queued"


@pytest.mark.parametrize("retry_before_cancellation", [False, True])
async def test_restart_finishes_saved_cancellation_before_starting_downloads(
    client: AsyncClient,
    app: FastAPI,
    started: list[str],
    monkeypatch: pytest.MonkeyPatch,
    retry_before_cancellation: bool,
) -> None:
    plan = await _plan(client)
    response = await client.post(_url(plan["id"]))
    assert response.status_code == 202, response.text
    jobs = {item["kind"]: item["id"] for item in response.json()}
    manager = app.state.services.downloads

    async def stop_before_child_cancel(_job_id: str, **_kwargs: Any) -> bool:
        raise OSError("Neutral shutdown interruption")

    with monkeypatch.context() as patch:
        patch.setattr(manager, "cancel", stop_before_child_cancel)
        with pytest.raises(OSError, match="Neutral shutdown interruption"):
            await client.post(f"/api/jobs/{jobs['workflow_install']}/cancel")
    with SessionLocal() as session:
        completion = session.get(models.Job, jobs["workflow_install"])
        download = session.get(models.Job, jobs["download"])
        assert completion is not None and completion.status == "cancelled"
        assert download is not None and download.status == "queued"
    before = _identities()
    started.clear()
    cancel = manager.cancel

    async def finish_cancel(job_id: str, *, cancelled_offer_id: str | None = None) -> bool:
        if retry_before_cancellation:
            retry = await client.post(f"/api/jobs/{jobs['workflow_install']}/retry")
            assert retry.status_code == 200, retry.text
        return await cancel(job_id, cancelled_offer_id=cancelled_offer_id)

    monkeypatch.setattr(manager, "cancel", finish_cancel)
    manager.recover_interrupted()
    assert manager._offer_recovery_task is not None
    await asyncio.wait_for(manager._offer_recovery_task, timeout=30)
    assert started == ([jobs["download"]] if retry_before_cancellation else [])
    assert _identities() == before
    with SessionLocal() as session:
        completion = session.get(models.Job, jobs["workflow_install"])
        download = session.get(models.Job, jobs["download"])
        expected = "queued" if retry_before_cancellation else "cancelled"
        assert completion is not None and completion.status == expected
        assert download is not None and download.status == expected
    if not retry_before_cancellation:
        retry = await client.post(f"/api/jobs/{jobs['workflow_install']}/retry")
        assert retry.status_code == 200, retry.text
    assert started == [jobs["download"]]
    assert _identities() == before


@pytest.mark.parametrize(
    "operation", ["cancel", "restart", "retry-before-cancel", "retry-during-cleanup", "direct"]
)
async def test_cancelled_installation_preserves_another_installations_shared_download(
    client: AsyncClient,
    app: FastAPI,
    started: list[str],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    payload = _payload()
    first_plan = await _plan(client, payload)
    first = await client.post(_url(first_plan["id"]))
    assert first.status_code == 202, first.text
    payload["ui_graph"]["nodes"][0]["widgets_values"].append(0.5)
    second_plan = await _plan(client, payload)
    second = await client.post(_url(second_plan["id"]))
    assert second.status_code == 202, second.text
    left = {item["kind"]: item["id"] for item in first.json()}
    right = {item["kind"]: item["id"] for item in second.json()}
    assert left["workflow_install"] != right["workflow_install"]
    assert left["download"] == right["download"]
    manager = app.state.services.downloads
    resumed: list[str] = []
    if operation == "retry-during-cleanup":
        stopped = await client.post(f"/api/jobs/{right['workflow_install']}/cancel")
        assert stopped.status_code == 200, stopped.text
        monkeypatch.setattr(manager, "start", type(manager).start.__get__(manager))

        async def download(identifier: str) -> None:
            resumed.append(identifier)

        monkeypatch.setattr(manager, "_download", download)
        stop_task = manager._stop_task

        async def retry_while_stopping(identifier: str) -> None:
            retry = await client.post(f"/api/jobs/{right['workflow_install']}/retry")
            assert retry.status_code == 200, retry.text
            await asyncio.sleep(0)
            assert resumed == []
            await stop_task(identifier)

        monkeypatch.setattr(manager, "_stop_task", retry_while_stopping)
    with SessionLocal() as session:
        offer = session.scalar(
            select(models.WorkflowInstallOffer).where(
                models.WorkflowInstallOffer.completion_job_id == left["workflow_install"]
            )
        )
        assert offer is not None
        offer_id = offer.id
    if operation == "restart":

        async def interrupted(_job_id: str, **_kwargs: Any) -> bool:
            raise OSError("Neutral shutdown interruption")

        with monkeypatch.context() as patch:
            patch.setattr(manager, "cancel", interrupted)
            with pytest.raises(OSError, match="Neutral shutdown interruption"):
                await client.post(f"/api/jobs/{left['workflow_install']}/cancel")
        started.clear()
        manager.recover_interrupted()
        assert manager._offer_recovery_task is not None
        await asyncio.wait_for(manager._offer_recovery_task, timeout=30)
        assert started == [left["download"]]
    else:
        target = left["download"] if operation == "direct" else left["workflow_install"]
        response = await client.post(f"/api/jobs/{target}/cancel")
        assert response.status_code == 200, response.text
        if operation == "retry-during-cleanup":
            await asyncio.sleep(0)
            assert resumed == [left["download"]]
            monkeypatch.setattr(manager, "_stop_task", stop_task)
        if operation == "retry-before-cancel":
            retry = await client.post(f"/api/jobs/{left['workflow_install']}/retry")
            assert retry.status_code == 200, retry.text
            assert not await manager.cancel(left["download"], cancelled_offer_id=offer_id)
    with SessionLocal() as session:
        download = session.get(models.Job, left["download"])
        other = session.get(models.Job, right["workflow_install"])
        assert download is not None and other is not None
        assert download.status == ("cancelled" if operation == "direct" else "queued")
        assert other.status == "queued"
