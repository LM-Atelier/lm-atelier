from __future__ import annotations

import json

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


async def test_creation_names_the_field_when_a_number_cannot_be_stored(
    client: AsyncClient,
) -> None:
    """A refusal that says which field, rather than a server error.

    A whole number past what a JSON round trip preserves used to pass every
    bound the settings enforce, and the first thing to touch it converted it -
    so the answer was a server error with nothing in it the caller could act
    on, from a request that had named exactly one bad field.
    """
    with SessionLocal() as session:
        definitions = session.scalar(select(func.count()).select_from(WorkflowDefinition))

    response = await client.post(
        "/api/workflows",
        json={
            **_payload(),
            "input_schema": {
                "type": "object",
                "properties": {"seed": {"type": "integer", "minimum": 10**400}},
            },
        },
    )

    assert response.status_code == 422, response.text
    assert "numbers must be finite" in response.text
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(WorkflowDefinition)) == definitions


async def test_a_new_revision_is_refused_the_same_way(client: AsyncClient) -> None:
    """The second door into the same row, which validates the same three things."""
    created = await client.post("/api/workflows", json=_payload())
    assert created.status_code == 201, created.text
    workflow = created.json()
    with SessionLocal() as session:
        revisions = session.scalar(select(func.count()).select_from(WorkflowRevision))

    response = await client.post(
        f"/api/workflows/{workflow['id']}/revisions",
        json={
            "api_graph": {"2": {"class_type": "MockImage"}},
            "input_schema": {
                "type": "object",
                "properties": {"steps": {"type": "integer", "maximum": 10**400}},
            },
        },
    )

    assert response.status_code == 422, response.text
    assert "numbers must be finite" in response.text
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(WorkflowRevision)) == revisions


async def test_a_wide_declared_range_with_an_ordinary_value_is_accepted(
    client: AsyncClient,
) -> None:
    """A control may offer a range far wider than the engine's own.

    The first version of the refusal above judged the DECLARED bounds by how
    many digits they had, and refused this workflow although the value it
    actually uses is forty-two. A refusal that makes a working workflow
    unusable is worse than the crash it was written to prevent.
    """
    response = await client.post(
        "/api/workflows",
        json={
            **_payload(),
            "input_schema": {
                "type": "object",
                "properties": {
                    "custom_count": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10**20,
                        "default": 42,
                    }
                },
            },
        },
    )

    assert response.status_code == 201, response.text


async def test_the_same_bound_written_two_ways_is_answered_the_same(
    client: AsyncClient,
) -> None:
    """A browser writes a large bound without an exponent, so its type changes.

    `JSON.stringify` renders ten to the twentieth as its digits, and Python then
    reads a whole number where the same schema left as a fraction. The two are
    the same number, so a workflow must not become acceptable or unacceptable by
    making a round trip through the page that edits it.
    """
    written_out = json.loads(
        '{"type":"object","properties":{"custom_count":'
        '{"type":"integer","minimum":0,"maximum":100000000000000000000,"default":42}}}'
    )
    assert written_out["properties"]["custom_count"]["maximum"] == 1e20

    for schema in (
        {
            "type": "object",
            "properties": {
                "custom_count": {"type": "integer", "minimum": 0, "maximum": 1e20, "default": 42}
            },
        },
        written_out,
    ):
        response = await client.post("/api/workflows", json={**_payload(), "input_schema": schema})
        assert response.status_code == 201, response.text
