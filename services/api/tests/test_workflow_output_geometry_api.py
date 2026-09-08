from __future__ import annotations

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
