"""Source-fit previews use selected stored bytes without creating work."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_source_fit_acceptance import prepared_turn
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.models import (
    Artifact,
    Job,
    Message,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
    WorkPlan,
)


def counts() -> tuple[int, ...]:
    with SessionLocal() as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model)) or 0
            for model in (Artifact, Message, WorkPlan, Run, Job)
        )


async def test_preview_is_read_only_and_matches_the_accepted_canvas(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    base = f"/api/workflow-revisions/{request['workflow_revision_id']}/source-fit"
    capability = await client.get(base)
    assert capability.status_code == 200, capability.text
    assert capability.json()["available"] is True
    assert capability.json()["modes"] == ["extend"]
    assert capability.json()["request_authorized"] is False
    before = counts()
    intent = {"mode": "extend", "width": 5, "height": 6}
    response = await client.post(
        base + "/preview",
        json={"source_artifact_id": request["input_artifact_ids"][0], "source_fit": intent},
    )
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["source"] == {"width": 2, "height": 3}
    assert preview["canvas"] == {"width": 5, "height": 6}
    assert preview["margins"] == {"left": 1, "top": 1, "right": 2, "bottom": 2}
    assert preview["source_rectangle"] == {"x": 1, "y": 1, "width": 2, "height": 3}
    assert preview["request_authorized"] is False
    assert preview["version"] == 1
    assert counts() == before
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat_id}/turns", json={**request, "source_fit": intent}
        )
        assert accepted.status_code == 202, accepted.text
        with SessionLocal() as session:
            run = session.get(Run, accepted.json()["run"]["id"])
            assert run is not None
            snapshot = accepted_context(session, run)
            assert snapshot is not None and snapshot.source_fit is not None
            recipe = snapshot.source_fit
            assert recipe.canvas_width == preview["canvas"]["width"]
            assert recipe.canvas_height == preview["canvas"]["height"]
            assert recipe.image.width == preview["source"]["width"]
            assert recipe.image.height == preview["source"]["height"]
            assert snapshot.workflow is not None
            recipe.route(snapshot.workflow.api_graph_json)
            assert recipe.margins() == {"top": 1, "right": 2, "bottom": 2, "left": 1}


@pytest.mark.parametrize("change", ["untrusted", "graph", "digest", "operation", "two_outputs"])
async def test_preview_rechecks_revision_and_refuses_without_creating_work(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    _, request = await prepared_turn(app, client, monkeypatch)
    revision_id = request["workflow_revision_id"]
    base = f"/api/workflow-revisions/{revision_id}/source-fit"
    first = await client.get(base)
    assert first.status_code == 200, first.text
    assert first.json()["available"] is True
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        if change == "untrusted":
            revision.trusted = False
        elif change == "digest":
            revision.artifact_sha256 = "0" * 64
        elif change == "operation":
            definition = session.get(WorkflowDefinition, revision.workflow_id)
            assert definition is not None
            definition.operation = "text_to_image"
        elif change == "two_outputs":
            revision.api_graph_json = {
                **revision.api_graph_json,
                "another_save": revision.api_graph_json["save"],
            }
        else:
            graph = dict(revision.api_graph_json)
            graph["composite"] = {
                **graph["composite"],
                "inputs": {**graph["composite"]["inputs"], "mask": ["pad", 0]},
            }
            revision.api_graph_json = graph
        session.commit()
    before = counts()
    capability = await client.get(base)
    assert capability.status_code == 200, capability.text
    assert capability.json()["available"] is False
    assert capability.json()["modes"] == []
    response = await client.post(
        base + "/preview",
        json={
            "source_artifact_id": request["input_artifact_ids"][0],
            "source_fit": {"mode": "extend", "width": 4, "height": 5},
        },
    )
    assert response.status_code == 422, response.text
    assert counts() == before


@pytest.mark.parametrize("change", ["crop", "bool", "float", "mode", "source_dimensions"])
async def test_preview_refuses_client_geometry_without_creating_work(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    _, request = await prepared_turn(app, client, monkeypatch)
    fit: dict[str, Any] = {"mode": "extend", "width": 4, "height": 5}
    body: dict[str, Any] = {
        "source_artifact_id": request["input_artifact_ids"][0],
        "source_fit": fit,
    }
    if change == "crop":
        fit["width"] = 1
    elif change == "bool":
        fit["width"] = True
    elif change == "float":
        fit["height"] = 5.0
    elif change == "mode":
        fit["mode"] = "crop"
    else:
        body["source"] = {"width": 1, "height": 1}
    before = counts()
    response = await client.post(
        f"/api/workflow-revisions/{request['workflow_revision_id']}/source-fit/preview",
        json=body,
    )
    assert response.status_code == 422, response.text
    assert counts() == before


async def test_preview_rechecks_source_bytes_after_a_successful_preview(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = await prepared_turn(app, client, monkeypatch)
    source_id = request["input_artifact_ids"][0]
    url = f"/api/workflow-revisions/{request['workflow_revision_id']}/source-fit/preview"
    body = {
        "source_artifact_id": source_id,
        "source_fit": {"mode": "extend", "width": 4, "height": 5},
    }
    first = await client.post(url, json=body)
    assert first.status_code == 200, first.text
    with SessionLocal() as session:
        source = session.get(Artifact, source_id)
        assert source is not None
        app.state.services.artifacts.resolve(source).write_bytes(b"changed neutral fixture")
    before = counts()
    response = await client.post(url, json=body)
    assert response.status_code == 422, response.text
    assert counts() == before


async def test_preview_missing_revision_or_source_returns_not_found(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = await prepared_turn(app, client, monkeypatch)
    missing = await client.get("/api/workflow-revisions/missing/source-fit")
    assert missing.status_code == 404, missing.text
    before = counts()
    response = await client.post(
        f"/api/workflow-revisions/{request['workflow_revision_id']}/source-fit/preview",
        json={
            "source_artifact_id": "sha256:" + "0" * 64,
            "source_fit": {"mode": "extend", "width": 4, "height": 5},
        },
    )
    assert response.status_code == 404, response.text
    assert counts() == before


async def test_preview_helper_keeps_records_unchanged_and_returns_exact_geometry(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm.schemas import SourceFitRequest
    from local_lm.source_fit_preview import preview_source_fit, source_fit_capability

    _, request = await prepared_turn(app, client, monkeypatch)
    before = counts()

    def refuse_write(*args: object, **kwargs: object) -> None:
        raise AssertionError("a preview must not ingest prepared media")

    monkeypatch.setattr(app.state.services.artifacts, "ingest_bytes", refuse_write)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, request["workflow_revision_id"])
        source = session.get(Artifact, request["input_artifact_ids"][0])
        assert revision is not None and source is not None
        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        assert source_fit_capability(definition, revision).available is True
        preview = preview_source_fit(
            definition,
            revision,
            app.state.services.artifacts,
            source,
            SourceFitRequest(mode="extend", width=5, height=6),
        )
        assert preview.source.model_dump() == {"width": 2, "height": 3}
        assert preview.canvas.model_dump() == {"width": 5, "height": 6}
        assert preview.margins.model_dump() == {"left": 1, "top": 1, "right": 2, "bottom": 2}
        assert preview.source_rectangle.model_dump() == {"x": 1, "y": 1, "width": 2, "height": 3}
        assert preview.request_authorized is False
        assert not session.new and not session.dirty and not session.deleted
    assert counts() == before


async def test_preview_helper_refuses_trusted_unsupported_graph_before_reading_source(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm.schemas import SourceFitRequest
    from local_lm.source_fit_preview import preview_source_fit, source_fit_capability

    _, request = await prepared_turn(app, client, monkeypatch, broken_graph=True)

    def refuse_read(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("unsupported graph must refuse before image decoding")

    monkeypatch.setattr(app.state.services.artifacts, "verified_bytes", refuse_read)
    before = counts()
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, request["workflow_revision_id"])
        source = session.get(Artifact, request["input_artifact_ids"][0])
        assert revision is not None and source is not None
        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        assert revision.trusted is True
        assert source_fit_capability(definition, revision).available is False
        with pytest.raises(ValueError, match="^source_fit_workflow_unsupported$"):
            preview_source_fit(
                definition,
                revision,
                app.state.services.artifacts,
                source,
                SourceFitRequest(mode="extend", width=5, height=6),
            )
    assert counts() == before


async def test_preview_http_refuses_a_trusted_unsupported_graph_without_writes(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = await prepared_turn(app, client, monkeypatch, broken_graph=True)
    base = f"/api/workflow-revisions/{request['workflow_revision_id']}/source-fit"
    before = counts()
    capability = await client.get(base)
    assert capability.status_code == 200, capability.text
    assert capability.json()["available"] is False
    assert capability.json()["modes"] == []
    response = await client.post(
        base + "/preview",
        json={
            "source_artifact_id": request["input_artifact_ids"][0],
            "source_fit": {"mode": "extend", "width": 4, "height": 5},
        },
    )
    assert response.status_code == 422, response.text
    assert counts() == before
