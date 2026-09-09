from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select

from local_lm.db import SessionLocal
from local_lm.models import WorkflowDefinition, WorkflowRevision


def _payload() -> dict[str, object]:
    return {
        "name": "Neutral workflow",
        "operation": "text_to_image",
        "engine": "mock",
        "api_graph": {"1": {"class_type": "MockImage", "inputs": {}}},
    }


@pytest.mark.parametrize("trusted", [False, True])
async def test_generic_creation_refuses_trust_before_writing(
    client: AsyncClient, trusted: bool
) -> None:
    with SessionLocal() as session:
        definitions = session.scalar(select(func.count()).select_from(WorkflowDefinition))
        revisions = session.scalar(select(func.count()).select_from(WorkflowRevision))
    response = await client.post("/api/workflows", json={**_payload(), "trusted": trusted})
    assert response.status_code == 422
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(WorkflowDefinition)) == definitions
        assert session.scalar(select(func.count()).select_from(WorkflowRevision)) == revisions


@pytest.mark.parametrize("trusted", [False, True])
async def test_generic_revision_refuses_trust_before_writing(
    client: AsyncClient, trusted: bool
) -> None:
    created = await client.post("/api/workflows", json=_payload())
    assert created.status_code == 201
    workflow = created.json()
    with SessionLocal() as session:
        count = session.scalar(select(func.count()).select_from(WorkflowRevision))
    response = await client.post(
        f"/api/workflows/{workflow['id']}/revisions",
        json={"api_graph": {"2": {"class_type": "MockImage"}}, "trusted": trusted},
    )
    assert response.status_code == 422
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(WorkflowRevision)) == count
        definition = session.get(WorkflowDefinition, workflow["id"])
        assert definition is not None
        assert definition.current_revision_id == workflow["current_revision_id"]


@pytest.mark.parametrize("trusted", [False, True])
async def test_clone_and_restore_preserve_only_stored_source_trust(
    client: AsyncClient, trusted: bool
) -> None:
    created = await client.post("/api/workflows", json=_payload())
    assert created.status_code == 201
    workflow = created.json()
    with SessionLocal() as session:
        source = session.get(WorkflowRevision, workflow["current_revision_id"])
        assert source is not None
        source.trusted = trusted
        session.commit()
    clone = await client.post(f"/api/workflows/{workflow['id']}/clone", json={})
    assert clone.status_code == 201, clone.json()
    cloned = clone.json()
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, cloned["current_revision_id"])
        assert revision is not None and revision.trusted is trusted
    fresh = await client.post(
        f"/api/workflows/{workflow['id']}/revisions",
        json={"api_graph": {"2": {"class_type": "MockImage"}}},
    )
    assert fresh.status_code == 201 and fresh.json()["trusted"] is False
    restored = await client.post(
        f"/api/workflows/{workflow['id']}/revisions/{workflow['current_revision_id']}/restore"
    )
    assert restored.status_code == 201, restored.json()
    assert restored.json()["trusted"] is trusted


async def test_query_parameters_cannot_mint_workflow_trust(client: AsyncClient) -> None:
    created = await client.post("/api/workflows?trusted=true", json=_payload())
    assert created.status_code == 201
    workflow = created.json()
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, workflow["current_revision_id"])
        assert revision is not None and revision.trusted is False
    revised = await client.post(
        f"/api/workflows/{workflow['id']}/revisions?trusted=true",
        json={"api_graph": {"2": {"class_type": "MockImage"}}},
    )
    assert revised.status_code == 201 and revised.json()["trusted"] is False
