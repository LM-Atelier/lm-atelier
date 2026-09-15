"""An accepted offer and its exact jobs must survive together before workers start."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import MetaData, Table, func, select
from test_workflow_install_offer_api import DIGEST, REFERENCE, _graph, _offer_payload
from test_workflow_revision_review import _GRAPH
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm import api as api_module
from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.model_planner import INSTALL_RESOLVER_VERSION
from local_lm.models import InstallPlan, Job, WorkflowInstallOffer
from local_lm.schemas import DownloadRequest
from local_lm.workflow_install_offers import revalidate_workflow_install_offer

pytestmark = pytest.mark.asyncio


async def _created_offer(app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> str:
    # Only the remote install plan and runtime metadata are fixture data.
    # Workflow identity, review and offer acceptance use production endpoints.
    with SessionLocal() as session:
        session.add(
            InstallPlan(
                id="accepted-plan",
                provider="civitai",
                remote_id="101",
                revision="202",
                role="image",
                engine="comfyui",
                plan_hash=hashlib.sha256(b"accepted-plan").hexdigest(),
                resolver_version=INSTALL_RESOLVER_VERSION,
                compatibility="supported",
                artifacts_json=[
                    {
                        "path": REFERENCE,
                        "kind": "lora",
                        "target_folder": "loras",
                        "size_bytes": 17,
                        "sha256": DIGEST,
                        "required": True,
                        "reuse": "download",
                        "source_version_id": "202",
                        "source_file_id": "301",
                    }
                ],
                runtime_contract_json={
                    "auxiliary_kind": "lora",
                    "comfy_paths": {"loras": "styles"},
                },
                activation_probe_json={},
                status="planned",
            )
        )
        session.commit()
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral download acceptance",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _GRAPH,
            "ui_graph": _graph(),
            "dependencies": {"version": 1, "slots": []},
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()["id"]
    revision = created.json()["current_revision_id"]
    review_url = f"/api/workflows/{workflow}/revisions/{revision}/review"
    preview = await client.get(review_url)
    assert preview.status_code == 200, preview.text
    approved = await client.post(
        review_url,
        json={
            "action": "approve",
            "subject_sha256": preview.json()["subject_sha256"],
        },
    )
    assert approved.status_code == 200 and approved.json()["trusted"], approved.text
    original_info = app.state.services.engines.media.object_info

    async def object_info() -> dict[str, Any]:
        return {**await original_info(), "LoraLoader": {}}

    monkeypatch.setattr(app.state.services.engines.media, "object_info", object_info)
    offer = await client.post(
        f"/api/workflows/{workflow}/revisions/{revision}/install-offers",
        json=_offer_payload("accepted-plan"),
    )
    assert offer.status_code == 201, offer.text
    return str(offer.json()["id"])


def _links(offer_id: str) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        table = Table(
            "workflow_install_offer_downloads", MetaData(), autoload_with=session.get_bind()
        )
        return [
            dict(row)
            for row in session.execute(select(table).where(table.c.offer_id == offer_id)).mappings()
        ]


async def test_worker_start_observes_committed_offer_and_exact_download_link(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    offer_id = await _created_offer(app, client, monkeypatch)
    started: list[str] = []

    def start(_manager: DownloadManager, job_id: str) -> None:
        with SessionLocal() as session:
            offer = session.get(WorkflowInstallOffer, offer_id)
            assert offer is not None and offer.status == "queued"
            assert offer.queued_at is not None and offer.completed_at is None
            job = session.get(Job, job_id)
            assert job is not None and job.status == "queued"
            links = _links(offer_id)
            assert len(links) == 1 and links[0]["job_id"] == job_id
            accepted = links[0]["request_json"]
            assert accepted == job.payload_json
            assert accepted["install_plan_id"] == "accepted-plan"
            assert accepted["expected_sha256"] == {REFERENCE: DIGEST}
            assert (
                links[0]["request_sha256"]
                == hashlib.sha256(
                    json.dumps(
                        accepted, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                    ).encode()
                ).hexdigest()
            )
        started.append(job_id)

    monkeypatch.setattr(DownloadManager, "start", start)
    response = await client.post(f"/api/workflow-install-offers/{offer_id}/install")
    assert response.status_code == 202, response.text
    assert started == [response.json()[0]["id"]]
    assert len(_links(offer_id)) == 1
    repeated = await client.post(f"/api/workflow-install-offers/{offer_id}/install")
    assert repeated.status_code == 422
    assert len(started) == 1 and len(_links(offer_id)) == 1
    app.state.services.downloads.recover_interrupted()
    assert started == [response.json()[0]["id"]] * 2
    assert len(_links(offer_id)) == 1
    accepted = _links(offer_id)[0]["request_json"]
    with SessionLocal() as session:
        job = session.get(Job, started[0])
        assert job is not None
        session.delete(job)
        session.commit()
    retained = _links(offer_id)
    assert len(retained) == 1 and retained[0]["job_id"] is None
    assert retained[0]["request_json"] == accepted


async def test_failed_offer_acceptance_leaves_no_jobs_or_links(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    offer_id = await _created_offer(app, client, monkeypatch)
    started: list[str] = []
    monkeypatch.setattr(DownloadManager, "start", lambda _manager, job_id: started.append(job_id))

    def refuse(_offer: WorkflowInstallOffer) -> None:
        raise RuntimeError("Neutral acceptance interruption")

    monkeypatch.setattr(api_module, "mark_workflow_install_offer_queued", refuse)
    with pytest.raises(RuntimeError, match="Neutral acceptance interruption"):
        await client.post(f"/api/workflow-install-offers/{offer_id}/install")
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Job)) == 0
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None and offer.status == "ready" and offer.queued_at is None
    assert started == []
    assert _links(offer_id) == []


async def test_accepted_offer_links_existing_paused_download_without_resuming_it(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    offer_id = await _created_offer(app, client, monkeypatch)
    started: list[str] = []
    monkeypatch.setattr(DownloadManager, "start", lambda _manager, job_id: started.append(job_id))
    with SessionLocal() as session:
        _, requests = revalidate_workflow_install_offer(
            session,
            offer_id,
            available_node_types={"LoraLoader"},
            available_asset_filenames=set(),
            installed_package_versions={},
        )
        # The ordinary producer creates the same request before offer acceptance.
        existing = app.state.services.downloads.create(
            session, DownloadRequest.model_validate(requests[0])
        )
        existing.status = "paused"
        existing_id = existing.id
        session.commit()
    started.clear()
    response = await client.post(f"/api/workflow-install-offers/{offer_id}/install")
    assert response.status_code == 202, response.text
    assert [(job["id"], job["status"]) for job in response.json()] == [(existing_id, "paused")]
    assert started == []
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Job)) == 1
    app.state.services.downloads.recover_interrupted()
    assert started == []
    assert [link["job_id"] for link in _links(offer_id)] == [existing_id]
