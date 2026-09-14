"""Activation preparation selects exact installed matches without granting readiness."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import event, func, select
from sqlalchemy.orm import Session
from test_workflow_activations import _asset, _custom_node, _model, _profile, _registry, _slot
from test_workflow_revision_review import _GRAPH
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.models import (
    ComfyRegistryInstall,
    CustomNodeInstall,
    ModelAssetInstall,
    WorkflowActivation,
    WorkflowDependencyBinding,
    WorkflowRevision,
)
from local_lm.schemas import RuntimeStatus
from local_lm.workflow_activations import WorkflowRuntimeMaterializer
from local_lm.workflow_bindings import WorkflowActivationResolution, WorkflowBindingSelection
from local_lm.workflow_dependencies import WorkflowDependencyContract

pytestmark = pytest.mark.asyncio


async def created(
    client: AsyncClient, slots: list[dict[str, object]], *, approve: bool = True
) -> tuple[str, str]:
    response = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral dependency preparation",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _GRAPH,
            "dependencies": {"version": 1, "slots": slots},
        },
    )
    assert response.status_code == 201, response.text
    workflow, revision = response.json()["id"], response.json()["current_revision_id"]
    if approve:
        url = f"/api/workflows/{workflow}/revisions/{revision}/review"
        preview = await client.get(url)
        assert preview.status_code == 200, preview.text
        result = await client.post(
            url, json={"action": "approve", "subject_sha256": preview.json()["subject_sha256"]}
        )
        assert result.status_code == 200 and result.json()["trusted"], result.text
    return workflow, revision


def url(workflow: str, revision: str) -> str:
    return f"/api/workflows/{workflow}/revisions/{revision}/activation/prepare"


def counts() -> tuple[int, int]:
    with SessionLocal() as session:
        return (
            session.scalar(select(func.count()).select_from(WorkflowActivation)) or 0,
            session.scalar(select(func.count()).select_from(WorkflowDependencyBinding)) or 0,
        )


async def test_preparation_is_read_only_and_its_exact_selection_can_be_activated(
    client: AsyncClient, tmp_path: Path
) -> None:
    content = b"neutral selected weights"
    digest = hashlib.sha256(content).hexdigest()
    workflow, revision = await created(
        client,
        [
            _slot(
                "style",
                "model_asset",
                requirements=[{"key": "exact", "constraints": {"sha256": digest}}],
            )
        ],
    )
    with SessionLocal() as session:
        asset = _asset(session, tmp_path / "selected", suffix="selected", content=content)
        selected = asset.id
        _asset(session, tmp_path / "other", suffix="other", content=b"other weights")
        session.commit()
        engine = session.get_bind()
    writes: list[str] = []

    def record(_conn: Any, _cursor: Any, statement: str, *_args: Any) -> None:
        if statement.lstrip().split(maxsplit=1)[0].upper() in {
            "INSERT",
            "UPDATE",
            "DELETE",
            "REPLACE",
            "CREATE",
            "DROP",
            "ALTER",
        }:
            writes.append(statement.split(maxsplit=1)[0])

    event.listen(engine, "before_cursor_execute", record)
    try:
        response = await client.get(url(workflow, revision))
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert response.status_code == 200, response.text
    body = response.json()
    assert writes == [] and counts() == (0, 0)
    assert body["state"] == "prepared"
    assert body["workflow_revision_id"] == revision
    assert body["issues"] == []
    assert len(body["selections"]) == 1
    choice = body["selections"][0]
    assert (
        choice["slot_name"],
        choice["requirement_key"],
        choice["local_kind"],
        choice["local_id"],
    ) == ("style", "exact", "model_asset", selected)
    assert len(choice["recorded_resource_identity_sha256"]) == 64
    assert body["slots"][0]["choices"][0]["name"] == "Asset selected"
    assert str(tmp_path) not in response.text
    payload = {
        key: body[key]
        for key in ("workflow_artifact_sha256", "dependency_contract_sha256", "selections")
    }
    activated = await client.post(url(workflow, revision).removesuffix("/prepare"), json=payload)
    assert activated.status_code == 200, activated.text
    assert counts() == (1, 1)


@pytest.mark.parametrize("satisfaction", ["all_of", "any_of"])
async def test_multiple_matching_installs_require_a_choice(
    client: AsyncClient, tmp_path: Path, satisfaction: str
) -> None:
    workflow, revision = await created(
        client, [_slot("style", "model_asset", satisfaction=satisfaction)]
    )
    with SessionLocal() as session:
        for name in ("first", "second"):
            _asset(session, tmp_path / name, suffix=name)
        session.commit()
    response = await client.get(url(workflow, revision))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "needs_attention" and body["selections"] is None
    assert body["issues"] == [{"code": "ambiguous_dependency_binding", "slot_name": "style"}]
    assert {item["selection"]["local_id"] for item in body["slots"][0]["choices"]} == {
        "asset_first",
        "asset_second",
    }
    assert counts() == (0, 0)


@pytest.mark.parametrize("satisfaction", ["all_of", "any_of"])
async def test_missing_required_match_cannot_prepare(
    client: AsyncClient, satisfaction: str
) -> None:
    workflow, revision = await created(
        client, [_slot("style", "model_asset", satisfaction=satisfaction)]
    )
    response = await client.get(url(workflow, revision))
    assert response.status_code == 200, response.text
    assert response.json()["selections"] is None
    assert response.json()["issues"] == [
        {"code": "missing_required_dependency", "slot_name": "style"}
    ]
    assert counts() == (0, 0)


async def test_optional_resources_are_available_without_being_implicitly_enabled(
    client: AsyncClient, tmp_path: Path
) -> None:
    workflow, revision = await created(client, [_slot("style", "model_asset", required=False)])
    with SessionLocal() as session:
        _asset(session, tmp_path / "optional", suffix="optional")
        session.commit()
    response = await client.get(url(workflow, revision))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "prepared" and body["selections"] == []
    assert len(body["slots"][0]["choices"]) == 1


@pytest.mark.parametrize("satisfaction", ["all_of", "any_of"])
async def test_multiple_requirements_preserve_slot_satisfaction(
    client: AsyncClient, tmp_path: Path, satisfaction: str
) -> None:
    workflow, revision = await created(
        client,
        [
            _slot(
                "style",
                "model_asset",
                satisfaction=satisfaction,
                requirements=[
                    {
                        "key": name,
                        "constraints": {"sha256": hashlib.sha256(name.encode()).hexdigest()},
                    }
                    for name in ("first", "second")
                ],
            )
        ],
    )
    with SessionLocal() as session:
        for name in ("first", "second"):
            _asset(session, tmp_path / name, suffix=name, content=name.encode())
        session.commit()
    response = await client.get(url(workflow, revision))
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["slots"][0]["choices"]) == 2
    if satisfaction == "all_of":
        assert body["state"] == "prepared"
        assert {item["requirement_key"] for item in body["selections"]} == {"first", "second"}
    else:
        assert body["state"] == "needs_attention" and body["selections"] is None
        assert body["issues"][0]["code"] == "ambiguous_dependency_binding"


@pytest.mark.parametrize("kind", ["model_install", "model_profile"])
async def test_model_and_profile_candidates_use_their_own_resource_kind(
    client: AsyncClient, tmp_path: Path, kind: str
) -> None:
    workflow, revision = await created(client, [_slot("model", kind)])
    with SessionLocal() as session:
        install = _model(session, tmp_path / "model", suffix="selected")
        profile = _profile(session, install, suffix="selected")
        expected = install.id if kind == "model_install" else profile.id
        session.commit()
    response = await client.get(url(workflow, revision))
    assert response.status_code == 200, response.text
    choice = response.json()["selections"][0]
    assert choice["local_kind"] == kind and choice["local_id"] == expected


async def test_untrusted_revision_cannot_prepare(client: AsyncClient) -> None:
    workflow, revision = await created(client, [], approve=False)
    response = await client.get(url(workflow, revision))
    assert response.status_code == 409
    assert response.json()["code"] == "workflow-activation-unavailable"
    assert counts() == (0, 0)


@pytest.mark.parametrize("change", ["bytes", "identity", "approval"])
async def test_preparation_never_replaces_final_verification(
    client: AsyncClient, tmp_path: Path, change: str
) -> None:
    workflow, revision = await created(client, [_slot("style", "model_asset")])
    root = tmp_path / "selected"
    with SessionLocal() as session:
        asset = _asset(session, root, suffix="selected")
        asset_id = asset.id
        session.commit()
    response = await client.get(url(workflow, revision))
    assert response.status_code == 200, response.text
    body = response.json()
    if change == "bytes":
        (root / "styles/style.safetensors").write_bytes(b"changed")
    elif change == "identity":
        with SessionLocal() as session:
            current = session.get(ModelAssetInstall, asset_id)
            assert current is not None
            changed = b"new consistent weights"
            (root / "styles/style.safetensors").write_bytes(changed)
            current.manifest_json = {
                **current.manifest_json,
                "sha256": hashlib.sha256(changed).hexdigest(),
            }
            session.commit()
    else:
        review_url = f"/api/workflows/{workflow}/revisions/{revision}/review"
        review = await client.get(review_url)
        revoked = await client.post(
            review_url, json={"action": "revoke", "subject_sha256": review.json()["subject_sha256"]}
        )
        assert revoked.status_code == 200, revoked.text
    payload = {
        key: body[key]
        for key in ("workflow_artifact_sha256", "dependency_contract_sha256", "selections")
    }
    activated = await client.post(url(workflow, revision).removesuffix("/prepare"), json=payload)
    assert activated.status_code == 409, activated.text
    assert counts() == (0, 0)


async def test_explicit_empty_contract_prepares_without_inventing_dependencies(
    client: AsyncClient,
) -> None:
    workflow, revision = await created(client, [])
    response = await client.get(url(workflow, revision))
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "prepared"
    assert response.json()["selections"] == []
    assert response.json()["slots"] == []


@pytest.mark.parametrize("kind", ["custom_node", "registry_package"])
@pytest.mark.parametrize("condition", ["verified", "inactive", "untrusted"])
async def test_extension_choices_preserve_existing_trust_and_activity_requirements(
    client: AsyncClient, tmp_path: Path, kind: str, condition: str
) -> None:
    workflow, revision = await created(client, [_slot("extension", kind)])
    with SessionLocal() as session:
        extension: CustomNodeInstall | ComfyRegistryInstall
        if kind == "custom_node":
            extension = _custom_node(
                session,
                tmp_path / "nodes",
                suffix="example",
                security_json={"node_types": ["ExampleNode"]},
            )
        else:
            extension = _registry(
                session,
                tmp_path / "nodes",
                tmp_path / "environments",
                suffix="example",
                node_types=["ExampleNode"],
            )
        selected_id = extension.id
        if condition == "inactive":
            extension.active = False
        if condition == "untrusted":
            extension.trusted = False
        session.commit()
    response = await client.get(url(workflow, revision))
    assert response.status_code == 200, response.text
    body = response.json()
    if condition == "verified":
        assert body["state"] == "prepared"
        assert body["selections"][0]["local_kind"] == kind
        assert body["selections"][0]["local_id"] == selected_id
    else:
        assert body["selections"] is None
        assert body["slots"][0]["choices"] == []
    assert counts() == (0, 0)


@pytest.mark.parametrize("condition", ["matching", "different_release", "missing"])
async def test_runtime_choice_is_bound_to_current_supported_runtime(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, condition: str
) -> None:
    workflow, revision = await created(
        client,
        [
            _slot(
                "runtime",
                "runtime",
                requirements=[
                    {
                        "key": "comfy",
                        "constraints": {"engine": "comfyui", "runtime_build": "fixture-release"},
                    }
                ],
            )
        ],
    )

    def status(_engine: str) -> RuntimeStatus:
        return RuntimeStatus(
            engine="comfyui",
            release="different" if condition == "different_release" else "fixture-release",
            state="missing" if condition == "missing" else "ready",
            supported=True,
            managed=True,
            distribution="neutral",
            license="neutral",
        )

    monkeypatch.setattr(app.state.services.processes.runtimes, "status", status)
    response = await client.get(url(workflow, revision))
    assert response.status_code == 200, response.text
    body = response.json()
    if condition == "matching":
        assert body["state"] == "prepared"
        assert body["selections"][0]["local_kind"] == "runtime"
        assert body["selections"][0]["local_id"] == "comfyui"
    else:
        assert body["state"] == "needs_attention" and body["selections"] is None
    assert counts() == (0, 0)


async def test_approval_revoked_during_preparation_refuses_the_result(
    client: AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_lm.workflow_activation_preparation as preparation
    from local_lm.workflow_activations import resolve_workflow_dependencies

    workflow, revision = await created(client, [_slot("style", "model_asset")])
    with SessionLocal() as session:
        _asset(session, tmp_path / "selected", suffix="selected")
        session.commit()
    original = resolve_workflow_dependencies

    def revoke(
        session: Session,
        contract: WorkflowDependencyContract,
        selections: Sequence[WorkflowBindingSelection],
        *,
        runtime_materializer: WorkflowRuntimeMaterializer | None,
    ) -> WorkflowActivationResolution:
        result = original(session, contract, selections, runtime_materializer=runtime_materializer)
        with SessionLocal() as other:
            row = other.get(WorkflowRevision, revision)
            assert row is not None
            row.trusted = False
            other.commit()
        return result

    monkeypatch.setattr(preparation, "resolve_workflow_dependencies", revoke)
    response = await client.get(url(workflow, revision))
    assert response.status_code == 409, response.text
    assert counts() == (0, 0)


@pytest.mark.parametrize("versioned", [False, True])
async def test_revision_details_expose_activation_contract_only_when_declared(
    client: AsyncClient, versioned: bool
) -> None:
    response = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral contract projection",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _GRAPH,
            "dependencies": {"version": 1, "slots": []} if versioned else {},
        },
    )
    assert response.status_code == 201, response.text
    revision = next(
        item
        for item in response.json()["revisions"]
        if item["id"] == response.json()["current_revision_id"]
    )
    expected = hashlib.sha256(b'{"slots":[],"version":1}').hexdigest() if versioned else None
    assert revision["dependency_contract_sha256"] == expected
