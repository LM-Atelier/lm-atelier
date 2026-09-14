"""Explicit import declarations survive compilation and exact draft retries."""

from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from test_workflow_package_import_endpoint import _object_info, _ui_graph, _wire_runtime
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm import api as api_module
from local_lm.db import SessionLocal
from local_lm.models import WorkflowDefinition, WorkflowDependencySlot, WorkflowRevision
from local_lm.revision_dependency_contract import persist_dependency_contract
from local_lm.workflow_dependencies import (
    parse_workflow_dependency_contract,
    workflow_dependency_contract_payload,
    workflow_dependency_contract_sha256,
)

pytestmark = pytest.mark.asyncio


def _declaration() -> dict[str, Any]:
    return {
        "version": 1,
        "slots": [
            {
                "name": name,
                "resource_kind": "registry_package",
                "required": False,
                "satisfaction": "all_of",
                "requirements": [{"key": "package", "constraints": {"package_id": name}}],
            }
            for name in ("zeta", "alpha")
        ],
    }


def _runtime(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    info = _object_info()
    for node in info.values():
        node["python_module"] = "nodes"
    _wire_runtime(app, monkeypatch, runtime_object_info=info)


async def _request(client: AsyncClient, draft: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "Neutral imported workflow",
        "operation": "text_to_image",
        "ui_graph": _ui_graph(),
    }
    if draft:
        created = await client.post("/api/workflows/packages/drafts", json=payload)
        assert created.status_code == 201, created.text
        payload.update(
            draft_workflow_id=created.json()["id"],
            draft_revision_id=created.json()["current_revision_id"],
        )
    return payload


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("declaration", [None, {"version": 1, "slots": []}, _declaration()])
async def test_import_persists_only_the_explicit_declaration(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    draft: bool,
    declaration: dict[str, Any] | None,
) -> None:
    _runtime(app, monkeypatch)
    payload = await _request(client, draft)
    if declaration is not None:
        payload["dependencies"] = declaration
    response = await client.post("/api/workflows/packages/import", json=payload)
    assert response.status_code == 201, response.text
    workflow_id = response.json()["id"]
    revision_id = response.json()["current_revision_id"]
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None and not revision.trusted
        rows = list(
            session.scalars(
                select(WorkflowDependencySlot)
                .where(WorkflowDependencySlot.workflow_revision_id == revision_id)
                .order_by(WorkflowDependencySlot.ordinal)
            )
        )
        if declaration is None:
            assert revision.dependencies_json == {}
            assert revision.dependency_contract_sha256 is None and rows == []
        else:
            parsed = parse_workflow_dependency_contract(declaration)
            assert revision.dependencies_json == workflow_dependency_contract_payload(parsed)
            assert revision.dependency_contract_sha256 == workflow_dependency_contract_sha256(
                parsed
            )
            assert [(row.ordinal, row.name) for row in rows] == [
                (i, name)
                for i, name in enumerate(sorted(slot["name"] for slot in declaration["slots"]))
            ]
        artifact = revision.artifact_sha256
        digest = revision.dependency_contract_sha256
    if declaration == {"version": 1, "slots": []}:
        review_url = f"/api/workflows/{workflow_id}/revisions/{revision_id}/review"
        preview = await client.get(review_url)
        assert preview.status_code == 200, preview.text
        approved = await client.post(
            review_url,
            json={"action": "approve", "subject_sha256": preview.json()["subject_sha256"]},
        )
        assert approved.status_code == 200 and approved.json()["trusted"], approved.text
        activated = await client.post(
            f"/api/workflows/{workflow_id}/revisions/{revision_id}/activation",
            json={
                "workflow_artifact_sha256": artifact,
                "dependency_contract_sha256": digest,
                "selections": [],
            },
        )
        assert activated.status_code == 200 and activated.json()["state"] == "ready", activated.text


@pytest.mark.parametrize(
    "declaration",
    [
        {"version": 2, "slots": []},
        {"version": 1, "slots": [{"name": "incomplete"}]},
        {"legacy_model": "unspecified"},
    ],
)
async def test_invalid_declarations_refuse_before_runtime_reads_or_writes(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, declaration: dict[str, Any]
) -> None:
    calls: list[str] = []

    async def unavailable() -> dict[str, Any]:
        calls.append("runtime")
        raise RuntimeError("Unexpected runtime access")

    monkeypatch.setattr(app.state.services.engines.media, "object_info", unavailable)
    with SessionLocal() as session:
        before = session.scalar(select(func.count()).select_from(WorkflowDefinition))
    payload = await _request(client, False)
    result = await client.post(
        "/api/workflows/packages/import", json={**payload, "dependencies": declaration}
    )
    assert result.status_code == 422, result.text
    assert calls == []
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(WorkflowDefinition)) == before


async def test_finalized_draft_reuses_an_equivalent_declaration_and_refuses_a_changed_one(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime(app, monkeypatch)
    payload = {**await _request(client, True), "dependencies": _declaration()}
    imported = await client.post("/api/workflows/packages/import", json=payload)
    assert imported.status_code == 201, imported.text
    revision_id = imported.json()["current_revision_id"]
    equivalent = deepcopy(payload)
    equivalent["dependencies"]["slots"].reverse()
    repeated = await client.post("/api/workflows/packages/import", json=equivalent)
    assert repeated.status_code == 201, repeated.text
    assert repeated.json()["current_revision_id"] == revision_id
    changed = deepcopy(payload)
    changed["dependencies"]["slots"][0]["required"] = True
    refused = await client.post("/api/workflows/packages/import", json=changed)
    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "workflow-package-draft-already-finalized"
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        assert revision.dependencies_json == workflow_dependency_contract_payload(
            parse_workflow_dependency_contract(payload["dependencies"])
        )
        assert (
            session.scalar(
                select(func.count())
                .select_from(WorkflowRevision)
                .where(WorkflowRevision.workflow_id == imported.json()["id"])
            )
            == 2
        )


@pytest.mark.parametrize("draft", [False, True])
async def test_a_failed_contract_write_leaves_no_partial_import(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, draft: bool
) -> None:
    _runtime(app, monkeypatch)
    payload = {**await _request(client, draft), "dependencies": _declaration()}

    def counts() -> tuple[int | None, ...]:
        with SessionLocal() as session:
            return tuple(
                session.scalar(select(func.count()).select_from(model))
                for model in (WorkflowDefinition, WorkflowRevision, WorkflowDependencySlot)
            )

    before = counts()
    persist = persist_dependency_contract

    def interrupted(session: Session, revision: WorkflowRevision) -> None:
        persist(session, revision)
        assert revision.dependency_contract_sha256 is not None
        raise RuntimeError("Synthetic declaration write failure")

    monkeypatch.setattr(api_module, "persist_dependency_contract", interrupted)
    from local_lm import workflow_revision_writes

    monkeypatch.setattr(workflow_revision_writes, "persist_dependency_contract", interrupted)
    with pytest.raises(RuntimeError, match="Synthetic declaration write failure"):
        await client.post("/api/workflows/packages/import", json=payload)
    assert counts() == before
    if draft:
        with SessionLocal() as session:
            definition = session.get(WorkflowDefinition, payload["draft_workflow_id"])
            assert definition is not None
            assert definition.current_revision_id == payload["draft_revision_id"]
