from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_workflow_install_offer_api import (
    REFERENCE,
    _install_node_inventory,
    _offer_payload,
    _seed,
)

from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.downloads import DownloadManager
from local_lm.model_planner import workflow_artifact_contract
from local_lm.models import (
    InstallPlan,
    Job,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowInstallOffer,
    WorkflowRevision,
)

pytestmark = pytest.mark.asyncio
FAMILY = "wffamily_offer_projection"


async def _reviewed_offer(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str, str, dict[str, Any]]:
    workflow_id, revision_id, plan_id = _seed()
    monkeypatch.setattr(app.state.services.settings, "media_engine", "comfyui")
    await _install_node_inventory(app, monkeypatch)
    with SessionLocal() as session:
        family = WorkflowFamily(id=FAMILY, name="Detail workflow", enabled=True, archived=False)
        session.add(family)
        definition = session.get(WorkflowDefinition, workflow_id)
        revision = session.get(WorkflowRevision, revision_id)
        assert definition is not None and revision is not None
        definition.family_id = FAMILY
        definition.variant_key = "image"
        revision.api_graph_json = {
            "1": {"class_type": "LoraLoader", "inputs": {"lora_name": REFERENCE}}
        }
        revision.artifact_sha256 = workflow_artifact_contract(
            operation=definition.operation,
            engine=revision.engine,
            api_graph=revision.api_graph_json,
            input_schema=revision.input_schema_json,
            dependencies=revision.dependencies_json,
        )
        session.commit()
    response = await client.post(
        f"/api/workflows/{workflow_id}/revisions/{revision_id}/install-offers",
        json=_offer_payload(plan_id),
    )
    assert response.status_code == 201
    return workflow_id, revision_id, plan_id, dict(response.json())


async def _variant(client: AsyncClient) -> dict[str, Any]:
    response = await client.get("/api/workflow-families?include_archived=true")
    assert response.status_code == 200
    family = next(item for item in response.json() if item["id"] == FAMILY)
    assert len(family["variants"]) == 1
    return dict(family["variants"][0])


async def test_family_projects_the_exact_reviewed_files_and_cost(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, revision_id, _, reviewed = await _reviewed_offer(app, client, monkeypatch)
    variant = await _variant(client)
    assert variant["readiness"] == "setup_required"
    assert variant["setup_resolution"] == "reviewed_download_available"
    assert variant["install_offer"] == reviewed
    assert variant["install_offer"]["workflow_revision_id"] == revision_id
    assert variant["install_offer"]["total_bytes"] == 17
    assert variant["install_offer"]["assets"][0]["reference_filename"] == REFERENCE


@pytest.mark.parametrize(
    "scenario",
    [
        "absent",
        "queued",
        "completed",
        "invalidated",
        "expired",
        "ambiguous",
        "different_artifact",
        "different_contract",
        "changed_execution",
        "changed_dependency_contract",
        "not_current",
        "new_revision",
        "untrusted",
        "disabled",
        "archived",
        "unexecutable",
        "different_engine",
        "ready",
        "terminal_metadata",
        "malformed_assets",
    ],
)
async def test_family_never_projects_a_stale_ambiguous_or_ineligible_offer(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    workflow_id, revision_id, _, reviewed = await _reviewed_offer(app, client, monkeypatch)
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, reviewed["id"])
        revision = session.get(WorkflowRevision, revision_id)
        definition = session.get(WorkflowDefinition, workflow_id)
        family = session.get(WorkflowFamily, FAMILY)
        assert (
            offer is not None
            and revision is not None
            and definition is not None
            and family is not None
        )
        if scenario == "absent":
            session.delete(offer)
        elif scenario in {"queued", "completed", "invalidated", "expired"}:
            offer.status = scenario
        elif scenario == "ambiguous":
            values = {
                column.name: getattr(offer, column.name)
                for column in WorkflowInstallOffer.__table__.columns
                if column.name not in {"id", "created_at", "updated_at"}
            }
            session.add(WorkflowInstallOffer(id="wfoffer_ambiguous", **values))
        elif scenario == "different_artifact":
            offer.workflow_artifact_sha256 = "b" * 64
        elif scenario == "different_contract":
            offer.dependency_contract_sha256 = "b" * 64
        elif scenario == "changed_execution":
            revision.api_graph_json = {"changed": {"class_type": "LoraLoader", "inputs": {}}}
        elif scenario == "changed_dependency_contract":
            revision.dependency_contract_sha256 = "b" * 64
            offer.dependency_contract_sha256 = "b" * 64
        elif scenario == "not_current":
            definition.current_revision_id = None
        elif scenario == "new_revision":
            values = {
                column.name: getattr(revision, column.name)
                for column in WorkflowRevision.__table__.columns
                if column.name not in {"id", "version", "created_at", "updated_at"}
            }
            current = WorkflowRevision(id="wfrev_new_current", version=2, **values)
            session.add(current)
            session.flush()
            definition.current_revision_id = current.id
        elif scenario == "untrusted":
            revision.trusted = False
        elif scenario == "disabled":
            family.enabled = False
        elif scenario == "archived":
            family.archived = True
        elif scenario == "unexecutable":
            revision.api_graph_json = {}
        elif scenario == "different_engine":
            revision.engine = "mock"
        elif scenario == "ready":
            revision.dependency_contract_sha256 = None
        elif scenario == "terminal_metadata":
            offer.queued_at = utcnow()
        elif scenario == "malformed_assets":
            offer.assets_json = [{}]
        else:
            raise AssertionError("unhandled fixture")
        session.commit()
    variant = await _variant(client)
    assert variant["install_offer"] is None
    assert variant["setup_resolution"] == (
        "attention_required" if variant["readiness"] == "setup_required" else None
    )


async def test_family_offer_projection_does_not_contact_inventory_or_start_jobs(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, reviewed = await _reviewed_offer(app, client, monkeypatch)

    async def refuse_inventory() -> dict[str, object]:
        raise AssertionError("GET contacted the runtime")

    def refuse_job(*_args: object, **_kwargs: object) -> Job:
        raise AssertionError("GET created a job")

    monkeypatch.setattr(app.state.services.engines.media, "object_info", refuse_inventory)
    monkeypatch.setattr(DownloadManager, "create", refuse_job)
    variant = await _variant(client)
    assert variant["install_offer"] == reviewed
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, reviewed["id"])
        assert offer is not None and offer.status == "ready"
        assert offer.queued_at is None and offer.invalidated_at is None
        assert session.scalar(select(func.count()).select_from(Job)) == 0


async def test_projected_offer_is_revalidated_after_the_user_opens_the_review(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, plan_id, reviewed = await _reviewed_offer(app, client, monkeypatch)
    variant = await _variant(client)
    assert variant["install_offer"]["id"] == reviewed["id"]
    with SessionLocal() as session:
        plan = session.get(InstallPlan, plan_id)
        assert plan is not None
        plan.plan_hash = "b" * 64
        session.commit()

    def refuse_job(*_args: object, **_kwargs: object) -> Job:
        raise AssertionError("stale review created a job")

    monkeypatch.setattr(DownloadManager, "create", refuse_job)
    result = await client.post(f"/api/workflow-install-offers/{reviewed['id']}/install")
    assert result.status_code == 422
    fresh = await _variant(client)
    assert fresh["install_offer"] is None
    assert fresh["setup_resolution"] == "attention_required"
