"""A revision stored without its artifact identity gets one when the workspace starts.

The value written must be the one creation would have written for the same
content, so these compare against a revision the real creation route made from
identical content rather than against the reconcile's own arithmetic.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import select
from test_workflow_revision_review import _GRAPH
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.models import WorkflowDefinition, WorkflowRevision
from local_lm.workflow_artifact_identity import reconcile_missing_artifact_identity

pytestmark = pytest.mark.asyncio


async def _created(client: AsyncClient, name: str, dependencies: dict[str, Any]) -> tuple[str, str]:
    response = await client.post(
        "/api/workflows",
        json={
            "name": name,
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _GRAPH,
            "dependencies": dependencies,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"], response.json()["current_revision_id"]


def _forget_identity(revision_id: str) -> None:
    """Put a revision in the state an older project import left it in."""

    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.artifact_sha256 = None
        session.commit()


def _identity(revision_id: str) -> str | None:
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        return revision.artifact_sha256


async def test_starting_the_workspace_records_the_identity_creation_would_have(
    app: FastAPI,
) -> None:
    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        client.headers["x-local-lm-csrf"] = (await client.post("/api/session")).json()["csrf_token"]
        _, reference = await _created(client, "Reference copy", {})
        _, forgotten = await _created(client, "Imported copy", {})
    expected = _identity(reference)
    assert expected is not None
    _forget_identity(forgotten)

    async with app.router.lifespan_context(app):
        pass

    assert _identity(forgotten) == expected


async def test_an_identity_already_stored_is_never_rewritten(client: AsyncClient) -> None:
    _, revision_id = await _created(client, "Kept identity", {})
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.artifact_sha256 = "a" * 64
        session.commit()

    with SessionLocal() as session:
        reconcile_missing_artifact_identity(session)
        session.commit()

    assert _identity(revision_id) == "a" * 64
    # Every revision now has one, so a second pass has nothing left to write.
    with SessionLocal() as session:
        missing = session.scalars(
            select(WorkflowRevision.id).where(WorkflowRevision.artifact_sha256.is_(None))
        ).all()
        assert missing == []
        assert reconcile_missing_artifact_identity(session) == 0


async def test_a_reviewed_revision_stays_reviewed_and_becomes_activatable(
    client: AsyncClient,
) -> None:
    """The case that mattered: reviewed, but refused activation for want of an identity."""

    workflow_id, revision_id = await _created(client, "Reviewed copy", {"version": 1, "slots": []})
    review_url = f"/api/workflows/{workflow_id}/revisions/{revision_id}/review"
    review = await client.get(review_url)
    approved = await client.post(
        review_url,
        json={"action": "approve", "subject_sha256": review.json()["subject_sha256"]},
    )
    assert approved.status_code == 200 and approved.json()["trusted"], approved.text
    expected = _identity(revision_id)
    _forget_identity(revision_id)
    activation_url = f"/api/workflows/{workflow_id}/revisions/{revision_id}/activation"
    assert (await client.get(activation_url)).status_code == 409

    with SessionLocal() as session:
        reconcile_missing_artifact_identity(session)
        session.commit()

    assert _identity(revision_id) == expected
    after = await client.get(review_url)
    assert after.status_code == 200 and after.json()["trusted"], after.text
    subject = await client.get(activation_url)
    assert subject.status_code == 200, subject.text
    assert subject.json()["workflow_artifact_sha256"] == expected


async def test_the_identity_follows_the_operation_and_what_the_workflow_needs(
    client: AsyncClient,
) -> None:
    """Operation and execution dependencies are part of what runs, so both are covered."""

    response = await client.post(
        "/api/workflows",
        json={
            "name": "Video copy",
            "operation": "text_to_video",
            "engine": "mock",
            "api_graph": {"loader": {"class_type": "PortableLoader"}},
            "dependencies": {"custom_nodes": [{"name": "portable-node-pack"}]},
        },
    )
    assert response.status_code == 201, response.text
    revision_id = response.json()["current_revision_id"]
    expected = _identity(revision_id)
    _forget_identity(revision_id)

    with SessionLocal() as session:
        reconcile_missing_artifact_identity(session)
        session.commit()
        definition = session.scalar(
            select(WorkflowDefinition).where(WorkflowDefinition.name == "Video copy")
        )
        assert definition is not None and definition.operation == "text_to_video"

    assert _identity(revision_id) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_graph_json", ["not", "a", "graph"]),
        ("dependencies_json", "not a mapping"),
        ("input_schema_json", {"properties": {"steps": {"default": float("nan")}}}),
    ],
)
async def test_a_row_that_cannot_be_read_is_left_alone_and_the_rest_are_repaired(
    app: FastAPI, field: str, value: object
) -> None:
    """Starting the workspace must not fail over one damaged row."""

    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        client.headers["x-local-lm-csrf"] = (await client.post("/api/session")).json()["csrf_token"]
        _, damaged = await _created(client, "Damaged copy", {})
        _, healthy = await _created(client, "Healthy copy", {})
    expected = _identity(healthy)
    _forget_identity(healthy)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, damaged)
        assert revision is not None
        setattr(revision, field, value)
        revision.artifact_sha256 = None
        session.commit()

    async with app.router.lifespan_context(app):
        pass

    assert _identity(damaged) is None
    assert _identity(healthy) == expected
