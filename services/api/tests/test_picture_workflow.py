"""A picture's own workflow is read only to be reviewed, and is imported only as untrusted."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import func, select
from test_exif_text_metadata import ascii_tag, exif_block, jpeg_with_exif, webp
from test_external_generation_metadata import _damaged_png, _png, _uploaded
from test_workflow_package_import_endpoint import _ui_graph, _wire_runtime

from local_lm.db import SessionLocal
from local_lm.models import Artifact, Job, WorkflowDefinition, WorkflowRevision
from local_lm.picture_workflow import read_picture_workflow

_SETTINGS = "a ceramic cup on a wooden table\nSteps: 20, Sampler: Euler a, Seed: 12345"


def _route(artifact_id: str) -> str:
    return f"/api/artifacts/{artifact_id}/embedded-workflow"


def _counts() -> dict[str, int]:
    with SessionLocal() as session:
        return {
            model.__name__: session.scalar(select(func.count()).select_from(model)) or 0
            for model in (Artifact, Job, WorkflowDefinition, WorkflowRevision)
        }


async def test_a_picture_s_workflow_is_reviewed_and_imported_only_as_untrusted(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _ui_graph()
    artifact_id = await _uploaded(
        client, _png(("parameters", _SETTINGS), ("workflow", json.dumps(graph)))
    )
    _wire_runtime(app, monkeypatch)
    before = _counts()

    read = await client.get(_route(artifact_id))

    assert read.status_code == 200, read.text
    assert read.headers["cache-control"] == "no-store"
    assert read.json() == {"ui_graph": graph}
    analysis = await client.post("/api/workflows/packages/analyze", json=read.json())
    assert analysis.status_code == 200, analysis.text
    assert analysis.json()["ready"] is True
    # The review so far has added no workflow, file or job.
    assert _counts() == before

    choice = {
        "ui_graph": read.json()["ui_graph"],
        "name": "Camera save",
        "operation": "text_to_image",
    }
    draft = await client.post("/api/workflows/packages/drafts", json=choice)
    assert draft.status_code == 201, draft.text
    imported = await client.post(
        "/api/workflows/packages/import",
        json={
            **choice,
            "draft_workflow_id": draft.json()["id"],
            "draft_revision_id": draft.json()["current_revision_id"],
        },
    )

    assert imported.status_code == 201, imported.text
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, imported.json()["current_revision_id"])
        assert revision is not None
        assert revision.trusted is False
    assert _counts()["Job"] == before["Job"]


@pytest.mark.parametrize("name", ["workflow", "Workflow"])
def test_a_jpeg_or_webp_carries_its_workflow_in_a_named_text_tag(name: str) -> None:
    graph = _ui_graph()
    block = exif_block([ascii_tag(0x010F, f"{name}:{json.dumps(graph)}")])

    assert read_picture_workflow(webp((b"EXIF", block))) == graph
    assert read_picture_workflow(jpeg_with_exif(block)) == graph


def _nested(depth: int) -> str:
    return '{"nodes": [], "extra": ' + '{"a": ' * depth + "1" + "}" * depth + "}"


@pytest.mark.parametrize(
    "payload",
    [
        _png(("parameters", _SETTINGS)),
        _png(("workflow", "not a graph")),
        _png(("workflow", '{"nodes": [], "nodes": []}')),
        # A graph written to be run, which the review does not read.
        _png(("workflow", json.dumps({"3": {"class_type": "KSampler", "inputs": {}}}))),
        _png(("workflow", "[]")),
        _png(("workflow", '{"nodes": [], "extra": NaN}')),
        _png(("workflow", _nested(40))),
        _png(("workflow", json.dumps(_ui_graph())), compressed=("workflow",)),
        _png(("workflow", json.dumps(_ui_graph())), ("workflow", json.dumps(_ui_graph()))),
        b"GIF89a" + b"\x00" * 32,
    ],
    ids=[
        "none",
        "not-json",
        "repeated-key",
        "run-graph",
        "not-an-object",
        "not-a-number",
        "too-deep",
        "compressed",
        "two",
        "another-format",
    ],
)
def test_text_the_review_could_not_take_is_no_workflow(payload: bytes) -> None:
    assert read_picture_workflow(payload) is None


def test_a_picture_whose_text_is_damaged_is_refused() -> None:
    with pytest.raises(ValueError):
        read_picture_workflow(_damaged_png())


async def test_the_route_says_when_there_is_no_workflow_to_review(client: AsyncClient) -> None:
    without = await _uploaded(client, _png(("parameters", _SETTINGS)))
    video = await _uploaded(client, b"\x00\x00\x00\x18ftypmp42", "video/mp4")
    damaged = await _uploaded(client, _damaged_png())

    for artifact_id in (without, video):
        response = await client.get(_route(artifact_id))
        assert response.status_code == 404, response.text
        assert response.json()["code"] == "picture-workflow-missing"
    unreadable = await client.get(_route(damaged))
    assert unreadable.status_code == 422
    assert unreadable.json()["code"] == "generation-settings-unreadable"
    unknown = await client.get(_route("sha256:" + "0" * 64))
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "artifact-not-found"


async def test_the_settings_still_list_the_workflow_as_left_out(client: AsyncClient) -> None:
    artifact_id = await _uploaded(
        client, _png(("parameters", _SETTINGS), ("workflow", json.dumps(_ui_graph())))
    )

    body: dict[str, Any] = (
        await client.get(f"/api/artifacts/{artifact_id}/generation-settings")
    ).json()

    assert {"name": "workflow", "reason": "workflow_graph"} in body["ignored"]
