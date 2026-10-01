"""A comparison's shape comes to each choice's size as that choice's workflow makes it."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_default_output_shape import _resolved
from test_workflow_output_geometry_api import _graph, _schema, _trust
from test_workflow_output_geometry_api import geometry_runtime as geometry_runtime

from local_lm.db import SessionLocal
from local_lm.models import ModelProfile

PREFLIGHT = "/api/generation-experiments/preflight"


async def _picture_workflow(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, *, seeded: bool
) -> tuple[str, str]:
    """A trusted picture workflow the geometry proof covers, and a model for it.

    ``seeded`` maps the sampler's seed to a setting; without it the workflow
    declares a seed it cannot be given.
    """

    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    monkeypatch.setattr(app.state.services.processes, "start_media", AsyncMock())
    graph: dict[str, Any] = _graph()
    schema: dict[str, Any] = _schema()
    if seeded:
        graph["sampler"]["inputs"]["seed"] = "${seed}"
        schema["properties"]["seed"] = {"type": "integer", "minimum": 0, "default": 0}
        described = app.state.services.engines.media.object_info

        async def object_info() -> dict[str, Any]:
            nodes: dict[str, Any] = await described()
            nodes["KSampler"]["input"]["required"]["seed"] = ["INT"]
            return nodes

        monkeypatch.setattr(app.state.services.engines.media, "object_info", object_info)
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Comparison shape fixture",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": graph,
            "input_schema": schema,
        },
    )
    assert created.status_code == 201, created.text
    revision_id: str = created.json()["current_revision_id"]
    _trust(revision_id)
    with SessionLocal() as session:
        profile = ModelProfile(name="Comparison shape fixture", role="image", engine="comfyui")
        session.add(profile)
        session.commit()
        return revision_id, profile.id


def _request(revision_id: str, profile_id: str, geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "Shape comparison",
        "operation": "text_to_image",
        "prompt": "A plain gray square on a white table",
        "geometry": geometry,
        "seed_policy": {"kind": "independent_deterministic"},
        "arms": [
            {"label": label, "profile_id": profile_id, "workflow_revision_id": revision_id}
            for label in ("First", "Second")
        ],
    }


@pytest.mark.parametrize("preset_id", ["3:2", "1:1"])
async def test_a_shape_comes_to_the_size_the_workflow_is_shown_to_make(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, preset_id: str
) -> None:
    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch, seeded=True)
    response = await client.post(
        PREFLIGHT,
        json=_request(revision_id, profile_id, {"mode": "preset", "preset_id": preset_id}),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == "compatible", body["refusals"]
    expected = await _resolved(client, revision_id, preset_id)
    for arm in body["arms"]:
        assert (arm["width"], arm["height"]) == expected
        settings = arm["effective_settings"]
        assert (settings["width"], settings["height"]) == expected


async def test_a_workflow_whose_seed_cannot_be_set_is_refused(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its result could not be reproduced, so it cannot be compared fairly."""

    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch, seeded=False)
    body = (
        await client.post(
            PREFLIGHT,
            json=_request(revision_id, profile_id, {"mode": "size", "width": 1024, "height": 768}),
        )
    ).json()
    assert [(item["code"], item["arm_ordinal"]) for item in body["refusals"]] == [
        ("arm-seed-unsupported", 1),
        ("arm-seed-unsupported", 2),
    ]
