"""Activation API boundaries for reviewed revisions and stored dependency snapshots.

Fixtures create and approve workflows through the API, then exercise activation
with explicit dependency snapshots and neutral installed files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from test_workflow_activations import _asset
from test_workflow_revision_review import _GRAPH
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.models import (
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowRevision,
    WorkflowRevisionReview,
)
from local_lm.workflow_activations import WorkflowActivationLaunchScope, activate_workflow_revision
from local_lm.workflow_bindings import WorkflowBindingSelection
from local_lm.workflow_dependencies import (
    parse_workflow_dependency_contract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_sha256,
)

pytestmark = pytest.mark.asyncio


async def created(
    client: AsyncClient, dependencies: dict[str, Any] | None = None
) -> tuple[str, str, dict[str, Any]]:
    declaration = dependencies if dependencies is not None else {"version": 1, "slots": []}
    result = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral activation workflow",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _GRAPH,
            "dependencies": declaration,
        },
    )
    assert result.status_code == 201, result.text
    workflow = result.json()["id"]
    revision = result.json()["current_revision_id"]
    contract = parse_workflow_dependency_contract(declaration)
    digest = workflow_dependency_contract_sha256(contract)
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None
        # Creation stores the declared contract itself; nothing is seeded here.
        assert row.dependency_contract_sha256 == digest
        stored = session.scalars(
            select(WorkflowDependencySlot)
            .where(WorkflowDependencySlot.workflow_revision_id == revision)
            .order_by(WorkflowDependencySlot.ordinal)
        ).all()
        # The declared names in canonical order, read from the fixture itself
        # rather than from the parsed contract the stored rows were built from.
        declared_names = sorted(item["name"] for item in declaration["slots"])
        slot_digests = {slot.name: workflow_dependency_slot_sha256(slot) for slot in contract.slots}
        assert [(item.ordinal, item.name, item.contract_sha256) for item in stored] == [
            (ordinal, name, slot_digests[name]) for ordinal, name in enumerate(declared_names)
        ]
        artifact = row.artifact_sha256
    review_url = f"/api/workflows/{workflow}/revisions/{revision}/review"
    preview = await client.get(review_url)
    assert preview.status_code == 200, preview.text
    approval = await client.post(
        review_url, json={"action": "approve", "subject_sha256": preview.json()["subject_sha256"]}
    )
    assert approval.status_code == 200, approval.text
    assert approval.json()["trusted"]
    return (
        workflow,
        revision,
        {
            "workflow_artifact_sha256": artifact,
            "dependency_contract_sha256": digest,
            "selections": [],
        },
    )


def url(workflow: str, revision: str) -> str:
    return f"/api/workflows/{workflow}/revisions/{revision}/activation"


def counts() -> tuple[int, int]:
    with SessionLocal() as session:
        return (
            session.scalar(select(func.count()).select_from(WorkflowActivation)) or 0,
            session.scalar(select(func.count()).select_from(WorkflowDependencyBinding)) or 0,
        )


async def test_activation_subject_is_read_only_and_exact(client: AsyncClient) -> None:
    workflow, revision, payload = await created(client)
    before = counts()
    response = await client.get(url(workflow, revision))
    assert response.status_code == 200, response.text
    assert response.json() == {
        "workflow_revision_id": revision,
        "workflow_artifact_sha256": payload["workflow_artifact_sha256"],
        "dependency_contract_sha256": payload["dependency_contract_sha256"],
        "slots": [],
    }
    assert counts() == before == (0, 0)


async def test_exact_activation_is_created_and_reused(client: AsyncClient) -> None:
    workflow, revision, payload = await created(client)
    first = await client.post(url(workflow, revision), json=payload)
    assert first.status_code == 200, first.text
    result = first.json()
    assert result["workflow_revision_id"] == revision
    assert result["dependency_contract_sha256"] == payload["dependency_contract_sha256"]
    assert result["state"] == "ready" and result["is_active"] is True
    assert "models" not in result and "base_path" not in result
    with SessionLocal() as session:
        row = session.get(WorkflowActivation, result["id"])
        assert row is not None and row.is_active and row.state == "ready"
        assert row.binding_sha256 == result["binding_sha256"]
    repeated = await client.post(url(workflow, revision), json=payload)
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["id"] == result["id"]
    assert counts() == (1, 0)


@pytest.mark.parametrize(
    "change",
    ["untrusted", "revoked", "graph", "declaration", "artifact", "contract", "current", "archived"],
)
async def test_changed_or_unavailable_revision_cannot_activate(
    client: AsyncClient, change: str
) -> None:
    workflow, revision, payload = await created(client)
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        definition = session.get(WorkflowDefinition, workflow)
        assert row is not None and definition is not None
        if change == "untrusted":
            row.trusted = False
        elif change == "revoked":
            review = session.get(WorkflowRevisionReview, revision)
            assert review is not None
            review.state = "revoked"
        elif change == "graph":
            row.api_graph_json = {"1": {"class_type": "EmptyLatentImage", "inputs": {}}}
        elif change == "declaration":
            row.dependencies_json = {"version": 1, "slots": [], "changed": True}
        elif change == "artifact":
            payload["workflow_artifact_sha256"] = "b" * 64
        elif change == "contract":
            payload["dependency_contract_sha256"] = "b" * 64
        elif change == "current":
            definition.current_revision_id = None
        else:
            family = WorkflowFamily(name="Archived neutral family", archived=True)
            session.add(family)
            session.flush()
            definition.family_id = family.id
        session.commit()
    result = await client.post(url(workflow, revision), json=payload)
    assert result.status_code == 409, result.text
    assert result.json()["code"] == "workflow-activation-unavailable"
    assert counts() == (0, 0)


async def test_activation_cannot_target_another_definition(client: AsyncClient) -> None:
    workflow, revision, payload = await created(client)
    other, _, _ = await created(client)
    result = await client.post(url(other, revision), json=payload)
    assert result.status_code == 404, result.text
    assert workflow != other
    assert counts() == (0, 0)


async def test_activation_refuses_unknown_selection_without_writes(client: AsyncClient) -> None:
    workflow, revision, payload = await created(client)
    payload["selections"] = [
        {
            "slot_name": "missing",
            "requirement_key": "default",
            "local_kind": "model_asset",
            "local_id": "missing-asset",
        }
    ]
    result = await client.post(url(workflow, revision), json=payload)
    assert result.status_code == 409, result.text
    assert counts() == (0, 0)


async def test_late_activation_refusal_rolls_back_the_new_snapshot(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import workflow_activation_requests as activation_requests

    workflow, revision, payload = await created(client)
    original = activate_workflow_revision

    def revoke_after_binding(
        session: Session,
        selected: WorkflowRevision | str,
        selections: list[WorkflowBindingSelection],
        **kwargs: Any,
    ) -> WorkflowActivationLaunchScope:
        result = original(session, selected, selections, **kwargs)
        row = session.get(WorkflowRevision, revision)
        assert row is not None
        row.trusted = False
        session.flush()
        return result

    monkeypatch.setattr(activation_requests, "activate_workflow_revision", revoke_after_binding)
    result = await client.post(url(workflow, revision), json=payload)
    assert result.status_code == 409, result.text
    assert counts() == (0, 0)
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None and row.trusted is True


@pytest.mark.parametrize(
    "condition", ["ready", "changed_bytes", "changed_identity", "missing_selection"]
)
async def test_activation_binds_and_verifies_the_selected_asset(
    client: AsyncClient, tmp_path: Path, condition: str
) -> None:
    workflow, revision, payload = await created(
        client,
        {
            "version": 1,
            "slots": [
                {
                    "name": "style",
                    "resource_kind": "model_asset",
                    "required": True,
                    "satisfaction": "all_of",
                    "requirements": [{"key": "selected", "constraints": {}}],
                }
            ],
        },
    )
    with SessionLocal() as session:
        asset = _asset(
            session, tmp_path / "neutral-asset", suffix="selected", content=b"neutral weights"
        )
        asset_id = asset.id
        session.commit()
    payload["selections"] = [
        {
            "slot_name": "style",
            "requirement_key": "selected",
            "local_kind": "model_asset",
            "local_id": asset_id,
        }
    ]
    if condition == "changed_bytes":
        (tmp_path / "neutral-asset/styles/style.safetensors").write_bytes(b"changed weights")
    elif condition == "changed_identity":
        payload["selections"][0]["recorded_resource_identity_sha256"] = "a" * 64
    elif condition == "missing_selection":
        payload["selections"] = []
    response = await client.post(url(workflow, revision), json=payload)
    if condition != "ready":
        assert response.status_code == 409, response.text
        assert counts() == (0, 0)
        return
    assert response.status_code == 200, response.text
    assert counts() == (1, 1)
    with SessionLocal() as session:
        binding = session.scalar(
            select(WorkflowDependencyBinding).where(
                WorkflowDependencyBinding.workflow_activation_id == response.json()["id"]
            )
        )
        assert binding is not None
        assert binding.model_asset_install_id == asset_id
        assert binding.requirement_key == "selected"
        assert binding.resource_identity_json == {
            "kind": "model_asset",
            "asset_kind": "lora",
            "runtime_reference": "styles/style.safetensors",
            "sha256": hashlib.sha256(b"neutral weights").hexdigest(),
        }
