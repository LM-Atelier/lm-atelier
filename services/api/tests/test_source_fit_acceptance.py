"""Source-fit intent is accepted through the ordinary HTTP turn boundary."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_source_fit_image import png
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_geometry import (
    composited_graph,
    inpaint_conditioning_graph,
    object_info,
)

from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.models import ModelProfile, Run, RunContextSnapshot


async def prepared_turn(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    *,
    broken_graph: bool = False,
    runtime_sampler: bool = False,
    conditioning: bool = False,
) -> tuple[str, dict[str, Any]]:
    # The conditioning shape is the one the official outpaint template uses.
    graph = inpaint_conditioning_graph() if conditioning else composited_graph()

    async def describe() -> dict[str, Any]:
        info = object_info(graph)
        if runtime_sampler:
            # The runtime describes these controls, so saving the workflow maps them.
            info["SaveImage"]["output_node"] = True
            info["KSampler"]["input"]["required"].update(
                {
                    "seed": ["INT", {"default": 0, "min": 0, "max": 2**32 - 1}],
                    "steps": ["INT", {"default": 20, "min": 1, "max": 100}],
                    "cfg": ["FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}],
                    "sampler_name": [["euler", "heun"], {}],
                    "scheduler": [["normal", "karras"], {}],
                    "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}],
                }
            )
        return info

    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    monkeypatch.setattr(app.state.services.engines.media, "object_info", describe, raising=False)
    with SessionLocal() as session:
        profile = ModelProfile(name="Source canvas fixture", role="image", engine="comfyui")
        session.add(profile)
        session.flush()
        profile_id = profile.id
        session.commit()
    graph["positive"]["inputs"]["text"] = "${prompt}"
    graph["sample"]["inputs"]["steps"] = "${steps}"
    if broken_graph:
        graph["composite"]["inputs"]["mask"] = ["pad", 0]
    properties = {
        "prompt": {"type": "string"},
        "input_image": {"type": "string"},
        "checkpoint": {"type": "string", "default": "fixture.safetensors"},
        "steps": {"type": "integer", "minimum": 1, "maximum": 100, "default": 2},
    }
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Source canvas fixture",
            "operation": "image_to_image",
            "engine": "comfyui",
            "api_graph": graph,
            "input_schema": {"type": "object", "properties": properties},
        },
    )
    assert created.status_code == 201, created.text
    definition = created.json()
    revision_id = definition["current_revision_id"]
    url = f"/api/workflows/{definition['id']}/revisions/{revision_id}/review"
    preview = await client.get(url)
    assert preview.status_code == 200, preview.text
    reviewed = await client.post(
        url,
        json={
            "action": "approve",
            "subject_sha256": preview.json()["subject_sha256"],
        },
    )
    assert reviewed.status_code == 200, reviewed.text
    assert reviewed.json()["trusted"] is True
    uploaded = await client.post(
        "/api/artifacts",
        files={
            "file": ("grid.png", png(orientation=6), "image/png"),
        },
    )
    assert uploaded.status_code == 201
    chat = await client.post("/api/chats", json={"title": "Source canvas fixture"})
    assert chat.status_code == 201
    return chat.json()["id"], {
        "text": "Extend the neutral color grid",
        "mode": "image",
        "input_artifact_ids": [uploaded.json()["id"]],
        "workflow_revision_id": revision_id,
        "profile_id": profile_id,
        "settings": {"steps": 2},
    }


async def test_ordinary_source_turn_still_queues_without_a_fit(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(f"/api/chats/{chat_id}/turns", json=request)
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            run = session.get(Run, response.json()["run"]["id"])
            assert run is not None
            snapshot = accepted_context(session, run)
            assert snapshot is None or snapshot.source_fit is None


async def test_source_fit_is_frozen_before_queue_start_and_estimates_the_requested_canvas(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    async with app.state.services.scheduler.lease("primary"):
        request.update(
            source_fit={"mode": "extend", "width": 4, "height": 5},
            idempotency_key="source-fit-acceptance",
        )
        response = await client.post(f"/api/chats/{chat_id}/turns", json=request)
        assert response.status_code == 202, response.text
        replay = await client.post(f"/api/chats/{chat_id}/turns", json=request)
        assert replay.status_code == 202, replay.text
        assert replay.json()["run"]["id"] == response.json()["run"]["id"]
        with SessionLocal() as session:
            run = session.get(Run, response.json()["run"]["id"])
            assert run is not None
            snapshot = accepted_context(session, run)
            assert snapshot is not None
            assert snapshot.source_fit is not None
            assert (snapshot.source_fit.image.width, snapshot.source_fit.image.height) == (2, 3)
            assert (snapshot.source_fit.canvas_width, snapshot.source_fit.canvas_height) == (4, 5)
            assert snapshot.source_fit.image.source_artifact_id == request["input_artifact_ids"][0]
            assert run.provenance_json["media_plan_estimate"]["work_units_per_output"] == 40
            assert session.scalar(select(func.count()).select_from(RunContextSnapshot)) == 1


@pytest.mark.parametrize(
    "bad_request", ["crop", "bool", "float", "string", "unknown_field", "graph"]
)
async def test_source_fit_refuses_an_invalid_request_without_queuing_any_work(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    bad_request: str,
) -> None:
    chat_id, request = await prepared_turn(
        app, client, monkeypatch, broken_graph=bad_request == "graph"
    )
    async with app.state.services.scheduler.lease("primary"):
        fit: dict[str, Any] = {"mode": "extend", "width": 4, "height": 5}
        if bad_request == "crop":
            fit["width"] = 1
        elif bad_request == "bool":
            fit["width"] = True
        elif bad_request == "float":
            fit["width"] = 4.0
        elif bad_request == "string":
            fit["height"] = "5"
        elif bad_request == "unknown_field":
            fit["source_artifact_id"] = request["input_artifact_ids"][0]
        request["source_fit"] = fit
        response = await client.post(f"/api/chats/{chat_id}/turns", json=request)
        assert response.status_code == 422, response.text
        with SessionLocal() as session:
            assert (
                session.scalar(select(func.count()).select_from(Run).where(Run.chat_id == chat_id))
                == 0
            )


@pytest.mark.parametrize(
    ("count", "budget", "status"),
    [(1, 39, 422), (1, 40, 202), (2, 79, 422), (2, 80, 202)],
)
async def test_source_canvas_budget_counts_every_requested_output(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    budget: int,
    status: int,
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    monkeypatch.setattr(app.state.services.engines.settings, "max_media_plan_work_units", budget)
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                **request,
                "source_fit": {"mode": "extend", "width": 4, "height": 5},
                "output_count": count,
            },
        )
        assert response.status_code == status, response.text
        with SessionLocal() as session:
            runs = list(session.scalars(select(Run).where(Run.chat_id == chat_id)))
            assert len(runs) == (count if status == 202 else 0)
            for run in runs:
                estimate = run.provenance_json["media_plan_estimate"]
                assert estimate["work_units_per_output"] == 40
                assert estimate["work_units"] == 40 * count
                snapshot = accepted_context(session, run)
                assert snapshot is not None and snapshot.source_fit is not None
                assert (snapshot.source_fit.canvas_width, snapshot.source_fit.canvas_height) == (
                    4,
                    5,
                )
