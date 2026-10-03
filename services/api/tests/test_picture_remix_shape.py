"""A picture whose own size a workflow cannot make is offered its shape at a size it makes."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from test_picture_remix import PROMPT, _new_chat, _png, _preview, _queue, _runs, _uploaded
from test_workflow_output_geometry_api import _graph, _schema, _trust
from test_workflow_output_geometry_api import geometry_runtime as geometry_runtime

from local_lm.db import SessionLocal
from local_lm.models import ModelProfile


def _settings(size: str) -> str:
    return f"{PROMPT}\nSteps: 20, Sampler: Euler a, Seed: 12345, Size: {size}"


async def _picture_workflow(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str]:
    """A trusted workflow whose sizes its own geometry proof covers, and a model for it.

    Its sizes run from 64 to 2048 on a 64 grid, and it makes 1024 x 768 by default.
    """

    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    # Work starts once the lease is let go; the engine's own runtime is never set up here.
    monkeypatch.setattr(app.state.services.processes, "start_media", AsyncMock())
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Shaped workflow",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _graph(),
            "input_schema": _schema(),
        },
    )
    assert created.status_code == 201, created.text
    revision_id: str = created.json()["current_revision_id"]
    _trust(revision_id)
    with SessionLocal() as session:
        profile = ModelProfile(name="Shaped model", role="image", engine="comfyui")
        session.add(profile)
        session.commit()
        return revision_id, profile.id


def _sizes(body: dict[str, Any]) -> dict[str, tuple[str, str | None]]:
    return {
        claim["key"]: (claim["state"], claim["reason"])
        for claim in body["claims"]
        if claim["key"] in {"width", "height"}
    }


@pytest.mark.parametrize(
    ("size", "shape"),
    [
        # Off the 64 grid: 2:3 at the size nearest the default area is 768 x 1152.
        ("1000x1500", {"width": 768, "height": 1152}),
        # Past the workflow's largest size: the same shape, at the same size.
        ("4000x6000", {"width": 768, "height": 1152}),
        # 3:2 on the 64 grid is 192c x 128c; c = 6 is nearest the default area.
        ("1500x1000", {"width": 1152, "height": 768}),
    ],
)
async def test_a_size_the_workflow_cannot_make_is_offered_as_its_shape(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    size: str,
    shape: dict[str, int],
) -> None:
    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch)
    artifact_id = await _uploaded(client, _png(("parameters", _settings(size))))

    plain = (await _preview(client, artifact_id, revision_id, profile_id)).json()
    shaped = (await _preview(client, artifact_id, revision_id, profile_id, ["shape"])).json()

    assert plain["ready"] is True, plain
    assert {state for state, _reason in _sizes(plain).values()} == {"incompatible"}
    assert plain["shape"] == shape
    assert plain["shape_applied"] is False
    assert {key: plain["resolved"]["settings"][key] for key in shape} == {
        "width": 1024,
        "height": 768,
    }
    assert shaped["shape_applied"] is True
    assert {key: shaped["resolved"]["settings"][key] for key in shape} == shape
    assert shaped["review_digest"] != plain["review_digest"]


@pytest.mark.parametrize(
    "size",
    [
        # Made exactly, so the picture's own size is applied, not a shape.
        "1024x1536",
        # A ratio the 64 grid cannot make exactly at any size it allows.
        "1001x1500",
    ],
)
async def test_no_shape_is_offered_where_none_is_needed_or_none_is_exact(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, size: str
) -> None:
    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch)
    artifact_id = await _uploaded(client, _png(("parameters", _settings(size))))

    body = (await _preview(client, artifact_id, revision_id, profile_id)).json()
    refused = await _preview(client, artifact_id, revision_id, profile_id, ["shape"])

    assert body["shape"] is None
    assert refused.status_code == 422
    assert refused.json()["code"] == "remix-choice-invalid"


async def test_a_shape_is_applied_alone(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch)
    artifact_id = await _uploaded(client, _png(("parameters", _settings("1000x1500"))))

    both = await _preview(client, artifact_id, revision_id, profile_id, ["shape", "width"])

    assert both.status_code == 422
    assert both.json()["code"] == "remix-choice-invalid"


async def test_a_remix_in_the_pictures_shape_is_queued_as_it_was_previewed(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch)
    artifact_id = await _uploaded(client, _png(("parameters", _settings("1000x1500"))))
    preview = (await _preview(client, artifact_id, revision_id, profile_id, ["shape"])).json()
    chat_id = await _new_chat(client)

    async with app.state.services.scheduler.lease("primary"):
        response = await _queue(client, chat_id, artifact_id, preview, ["shape"])

        assert response.status_code == 202, response.text
        [run] = _runs(chat_id)
        assert run.workflow_revision_id == revision_id
        assert (run.settings_json["width"], run.settings_json["height"]) == (768, 1152)
        assert run.settings_json == preview["resolved"]["settings"]
        receipt = run.provenance_json["remix"]
        assert receipt["shape_applied"] is True
        assert not any(claim["applied"] for claim in receipt["claims"])
        # Which choices, never a claim's value: no claim entry carries one.
        assert all(set(claim) == {"key", "state", "applied"} for claim in receipt["claims"])
