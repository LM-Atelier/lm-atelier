from __future__ import annotations

import hashlib
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy.orm import Session

from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.model_planner import INSTALL_RESOLVER_VERSION, workflow_artifact_contract
from local_lm.models import (
    InstallPlan,
    Job,
    WorkflowDefinition,
    WorkflowInstallOffer,
    WorkflowRevision,
)
from local_lm.schemas import DownloadRequest
from local_lm.workflow_dependencies import (
    parse_workflow_dependency_contract,
    workflow_dependency_contract_sha256,
)

pytestmark = pytest.mark.asyncio

REFERENCE = "styles/detail.safetensors"
DIGEST = "a" * 64


def _graph() -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "LoraLoader",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [REFERENCE],
                "properties": {"cnr_id": "comfy-core", "version": "0.28.0"},
            }
        ],
        "links": [],
    }


def _seed() -> tuple[str, str, str]:
    contract = parse_workflow_dependency_contract({"version": 1, "slots": []})
    with SessionLocal() as session:
        definition = WorkflowDefinition(
            id="workflow_offer_api",
            name="Offer API workflow",
            operation="text_to_image",
        )
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            id="wfrev_offer_api",
            workflow_id=definition.id,
            version=1,
            engine="comfyui",
            ui_graph_json=_graph(),
            api_graph_json={},
            input_schema_json={},
            dependencies_json={},
            artifact_sha256=workflow_artifact_contract(
                operation=definition.operation,
                engine="comfyui",
                api_graph={},
                input_schema={},
                dependencies={},
            ),
            dependency_contract_sha256=workflow_dependency_contract_sha256(contract),
            trusted=True,
        )
        plan = InstallPlan(
            id="plan_offer_api",
            provider="civitai",
            remote_id="101",
            revision="202",
            role="image",
            engine="comfyui",
            plan_hash=hashlib.sha256(b"plan_offer_api").hexdigest(),
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
        session.add_all([revision, plan])
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        return definition.id, revision.id, plan.id


def _offer_payload(plan_id: str) -> dict[str, object]:
    return {
        "selections": [
            {
                "reference_filename": REFERENCE,
                "install_plan_id": plan_id,
                "artifact_path": REFERENCE,
            }
        ]
    }


async def _install_node_inventory(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def object_info() -> dict[str, object]:
        return {"LoraLoader": {}}

    monkeypatch.setattr(
        app.state.services.engines.media,
        "object_info",
        object_info,
        raising=False,
    )


async def test_create_and_queue_offer_uses_only_the_opaque_offer_id(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_id, revision_id, plan_id = _seed()
    await _install_node_inventory(app, monkeypatch)
    captured: list[DownloadRequest] = []

    def create(
        _manager: DownloadManager,
        session: Session,
        request: DownloadRequest,
    ) -> Job:
        captured.append(request)
        job = Job(
            id="job_offer_api",
            kind="download",
            status="queued",
            payload_json=request.model_dump(mode="json"),
        )
        session.add(job)
        session.flush()
        return job

    monkeypatch.setattr(DownloadManager, "create", create)
    created = await client.post(
        f"/api/workflows/{workflow_id}/revisions/{revision_id}/install-offers",
        json=_offer_payload(plan_id),
    )

    assert created.status_code == 201
    body = created.json()
    assert body["workflow_revision_id"] == revision_id
    assert body["status"] == "ready"
    assert body["plan_count"] == 1
    assert body["total_bytes"] == 17
    assert body["assets"][0]["sha256"] == DIGEST
    assert "selections" not in body

    queued = await client.post(
        f"/api/workflow-install-offers/{body['id']}/install",
    )

    assert queued.status_code == 202
    assert len(queued.json()) == 1
    assert len(captured) == 1
    assert captured[0].install_plan_id == plan_id
    assert captured[0].expected_sha256 == {REFERENCE: DIGEST}
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, body["id"])
        assert offer is not None
        assert offer.status == "queued"
        assert offer.queued_at is not None


async def test_offer_creation_forbids_browser_graph_and_digest_claims(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_id, revision_id, plan_id = _seed()
    await _install_node_inventory(app, monkeypatch)
    response = await client.post(
        f"/api/workflows/{workflow_id}/revisions/{revision_id}/install-offers",
        json={
            **_offer_payload(plan_id),
            "ui_graph": _graph(),
            "binding_plan_sha256": "f" * 64,
        },
    )

    assert response.status_code == 422


async def test_stale_offer_refuses_before_any_download_job_starts(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_id, revision_id, plan_id = _seed()
    await _install_node_inventory(app, monkeypatch)
    created = await client.post(
        f"/api/workflows/{workflow_id}/revisions/{revision_id}/install-offers",
        json=_offer_payload(plan_id),
    )
    assert created.status_code == 201
    offer_id = created.json()["id"]
    with SessionLocal() as session:
        plan = session.get(InstallPlan, plan_id)
        assert plan is not None
        plan.status = "failed"
        session.commit()

    def must_not_create(*_args: object, **_kwargs: object) -> Job:
        raise AssertionError("a stale offer must not create a download")

    monkeypatch.setattr(DownloadManager, "create", must_not_create)
    refused = await client.post(f"/api/workflow-install-offers/{offer_id}/install")

    assert refused.status_code == 422
    assert refused.json()["code"] == "install_plan_not_pending"
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        assert offer.status == "invalidated"
        assert offer.invalidation_code == "install_plan_not_pending"


async def test_offer_review_reports_media_inventory_unavailability(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_id, revision_id, plan_id = _seed()

    async def unavailable() -> dict[str, object]:
        raise ConnectionError("worker stopped")

    monkeypatch.setattr(
        app.state.services.engines.media,
        "object_info",
        unavailable,
        raising=False,
    )
    response = await client.post(
        f"/api/workflows/{workflow_id}/revisions/{revision_id}/install-offers",
        json=_offer_payload(plan_id),
    )

    assert response.status_code == 503
    assert response.json()["code"] == "media-runtime-unavailable"
