from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select

from local_lm.db import SessionLocal
from local_lm.models import Artifact, Job, Message, Run, WorkflowRevision, WorkPlan


def _graph() -> dict[str, object]:
    return {
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": "${width}", "height": "${height}", "batch_size": 1},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {"latent_image": ["latent", 0]},
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sampler", 0]},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0]},
        },
    }


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "width": {
                "type": "integer",
                "default": 1024,
                "minimum": 64,
                "maximum": 2048,
                "multipleOf": 64,
            },
            "height": {
                "type": "integer",
                "default": 768,
                "minimum": 64,
                "maximum": 2048,
                "multipleOf": 64,
            },
        },
    }


def _generation_row_counts() -> tuple[int, ...]:
    with SessionLocal() as session:
        return tuple(
            int(session.scalar(select(func.count()).select_from(model)) or 0)
            for model in (Message, WorkPlan, Run, Job, Artifact)
        )


def _trust(revision_id: str) -> None:
    """Mark a revision trusted the way the application does.

    WorkflowCreate no longer accepts a trusted flag - create_workflow persists
    untrusted and a separate path grants it - so the geometry proof, which only
    answers for a trusted revision, needs the grant applied directly here.
    """

    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.trusted = True
        session.commit()


async def test_revision_geometry_capability_is_read_only_and_revision_bound(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Exact geometry",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _graph(),
            "input_schema": _schema(),
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()
    revision_id = workflow["revisions"][0]["id"]
    _trust(revision_id)
    before = (await client.get("/api/workflows")).json()
    generation_rows_before = _generation_row_counts()

    response = await client.get(f"/api/workflow-revisions/{revision_id}/output-geometry")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["available"] is True
    assert body["reason"] is None
    assert body["workflow_id"] == workflow["id"]
    assert body["revision_id"] == revision_id
    assert len(body["artifact_sha256"]) == 64
    assert body["operation"] == "text_to_image"
    assert body["engine"] == "comfyui"
    assert body["size_modes"] == ["exact"]
    assert body["width"]["node_id"] == "latent"
    assert body["width"]["default"] == 1024
    assert body["height"]["default"] == 768
    assert body["save_node_ids"] == ["save"]
    assert body["graph_binding_verified"] is True
    assert body["request_authorized"] is False
    assert body["capability"]["allowed_preset_ids"] == []
    assert body["capability"]["combinations"][0]["size_mode"] == "exact"

    after = (await client.get("/api/workflows")).json()
    assert after == before
    assert _generation_row_counts() == generation_rows_before


async def test_revision_geometry_capability_refuses_without_leaking_graph_details(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Unsupported private graph",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": {
                "private-transform": {
                    "class_type": "PrivateImageScaleNode",
                    "inputs": {"private-secret": "do-not-echo"},
                }
            },
            "input_schema": _schema(),
        },
    )
    assert created.status_code == 201, created.text
    revision_id = created.json()["revisions"][0]["id"]
    _trust(revision_id)

    response = await client.get(f"/api/workflow-revisions/{revision_id}/output-geometry")

    assert response.status_code == 200
    assert response.json() == {
        "version": 1,
        "available": False,
        "reason": "unsupported_workflow_geometry",
        "revision_id": None,
        "workflow_id": None,
        "artifact_sha256": None,
        "operation": None,
        "engine": None,
        "size_modes": [],
        "width": None,
        "height": None,
        "latent_node_id": None,
        "sampler_node_ids": [],
        "decode_node_ids": [],
        "save_node_ids": [],
        "capability": None,
        "graph_binding_verified": False,
        "request_authorized": False,
    }
    assert "private" not in response.text.lower()


async def test_revision_geometry_capability_returns_404_for_unknown_revision(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/workflow-revisions/does-not-exist/output-geometry")

    assert response.status_code == 404
    assert response.json()["code"] == "workflow-revision-not-found"


def _request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "mode": "image",
        "size_mode": "exact",
        "width": 1024,
        "height": 768,
    }
    request.update(overrides)
    return request


async def _trusted_revision(
    client: AsyncClient,
    name: str,
    *,
    api_graph: dict[str, object] | None = None,
    input_schema: dict[str, object] | None = None,
) -> tuple[str, str]:
    created = await client.post(
        "/api/workflows",
        json={
            "name": name,
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": api_graph if api_graph is not None else _graph(),
            "input_schema": input_schema if input_schema is not None else _schema(),
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()
    revision_id = workflow["revisions"][0]["id"]
    _trust(revision_id)
    return workflow["id"], revision_id


async def test_revision_geometry_request_resolves_read_only_and_revision_bound(
    client: AsyncClient,
) -> None:
    workflow_id, revision_id = await _trusted_revision(client, "Resolvable geometry")
    capability = await client.get(f"/api/workflow-revisions/{revision_id}/output-geometry")
    assert capability.status_code == 200, capability.text
    before = (await client.get("/api/workflows")).json()
    generation_rows_before = _generation_row_counts()

    response = await client.post(
        f"/api/workflow-revisions/{revision_id}/output-geometry/resolve",
        json=_request(width=1536, height=512),
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "version": 1,
        "workflow_id": workflow_id,
        "revision_id": revision_id,
        "artifact_sha256": capability.json()["artifact_sha256"],
        "operation": "text_to_image",
        "engine": "comfyui",
        "mode": "image",
        "size_mode": "exact",
        "width": 1536,
        "height": 512,
        "graph_binding_verified": True,
        "request_authorized": False,
    }

    after = (await client.get("/api/workflows")).json()
    assert after == before
    assert _generation_row_counts() == generation_rows_before


@pytest.mark.parametrize(
    "body",
    [
        _request(width=True),
        _request(height=1024.0),
        _request(width="1024"),
        _request(width=1000),
        _request(height=32),
        _request(width=4096),
        _request(mode="video"),
        _request(size_mode="preset"),
        _request(size_mode="workflow_native"),
        _request(request_authorized=True),
        _request(graph_binding_verified=True),
        _request(revision_id="private-caller-supplied-id"),
        {"mode": "image", "size_mode": "exact", "width": 1024},
        {},
    ],
)
async def test_revision_geometry_request_refuses_invalid_and_forged_requests(
    client: AsyncClient, body: dict[str, object]
) -> None:
    _, revision_id = await _trusted_revision(client, "Refusing geometry")

    response = await client.post(
        f"/api/workflow-revisions/{revision_id}/output-geometry/resolve", json=body
    )

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "workflow-geometry-request-invalid"
    assert "private" not in response.text.lower()


async def test_revision_geometry_request_refuses_an_unsupported_workflow_the_same_way(
    client: AsyncClient,
) -> None:
    _, revision_id = await _trusted_revision(
        client,
        "Unsupported private graph for resolution",
        api_graph={
            "private-transform": {
                "class_type": "PrivateImageScaleNode",
                "inputs": {"private-secret": "do-not-echo"},
            }
        },
    )

    response = await client.post(
        f"/api/workflow-revisions/{revision_id}/output-geometry/resolve", json=_request()
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "workflow-geometry-request-invalid",
        "detail": ("Output geometry request is invalid or unsupported for this workflow revision"),
    }
    assert "private" not in response.text.lower()


async def test_revision_geometry_request_refuses_an_untrusted_revision(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Untrusted geometry",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _graph(),
            "input_schema": _schema(),
        },
    )
    assert created.status_code == 201, created.text
    revision_id = created.json()["revisions"][0]["id"]

    response = await client.post(
        f"/api/workflow-revisions/{revision_id}/output-geometry/resolve", json=_request()
    )

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-geometry-request-invalid"


async def test_revision_geometry_requests_do_not_carry_across_revisions(
    client: AsyncClient,
) -> None:
    narrow_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "width": {
                "type": "integer",
                "default": 512,
                "minimum": 64,
                "maximum": 768,
                "multipleOf": 64,
            },
            "height": {
                "type": "integer",
                "default": 512,
                "minimum": 64,
                "maximum": 768,
                "multipleOf": 64,
            },
        },
    }
    wide_id, wide_revision = await _trusted_revision(client, "Wide geometry")
    narrow_id, narrow_revision = await _trusted_revision(
        client, "Narrow geometry", input_schema=narrow_schema
    )
    assert wide_id != narrow_id

    wide = await client.post(
        f"/api/workflow-revisions/{wide_revision}/output-geometry/resolve",
        json=_request(width=1024, height=1024),
    )
    narrow = await client.post(
        f"/api/workflow-revisions/{narrow_revision}/output-geometry/resolve",
        json=_request(width=1024, height=1024),
    )

    assert wide.status_code == 200, wide.text
    assert wide.json()["workflow_id"] == wide_id
    assert wide.json()["revision_id"] == wide_revision
    assert narrow.status_code == 422
    assert narrow.json()["code"] == "workflow-geometry-request-invalid"


async def test_revision_geometry_request_returns_404_for_unknown_revision(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/workflow-revisions/does-not-exist/output-geometry/resolve", json=_request()
    )

    assert response.status_code == 404
    assert response.json()["code"] == "workflow-revision-not-found"
