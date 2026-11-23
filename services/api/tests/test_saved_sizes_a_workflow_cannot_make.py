"""A saved size the turn's workflow cannot make is refused, never half applied.

Settings saved on a profile, preset, project or chat are read leniently, so a
setting a workflow no longer offers cannot stop a conversation from working.
A size used to be dropped that way too: the workflow's own width beside the
saved height, a picture of a size nobody chose, with nothing to say so. The
turn's own size was always refused when the workflow could not make it.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_default_output_shape import _chat, _picture_workflow
from test_workflow_output_geometry_api import _dimension, _generation_row_counts, _schema
from test_workflow_output_geometry_api import geometry_runtime as geometry_runtime
from workflow_fixtures import seed_workflow_trust

from local_lm.db import SessionLocal
from local_lm.domain import new_id
from local_lm.models import GenerationPreset, ModelProfile, Run

#: A workflow that makes 256 to 2048 px a side, in steps of 64, 1408 square by default.
CAPPED = _schema(_dimension(256, 1408, 2048), _dimension(256, 1408, 2048))
TOO_WIDE = {"width": 2176, "height": 768}
FITS = {"width": 2048, "height": 768}

#: Where a size can be saved, from the one that wins down, and how a refusal names it.
SAVED = {
    "turn preset": "in the preset chosen for this turn",
    "chat": "for this conversation",
    "chat preset": "in this conversation's preset",
    "project": "for this project",
    "project preset": "in this project's preset",
    "default preset": "in the default preset",
    "model profile": "in the model profile",
    "model profile load settings": "in the model profile",
}


def _preset(settings: dict[str, Any]) -> str:
    with SessionLocal() as session:
        preset = GenerationPreset(name=new_id("size"), role="image", settings_json=settings)
        session.add(preset)
        session.commit()
        return preset.id


async def _save(
    client: AsyncClient, chat_id: str, profile_id: str, sizes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Save each size where a person can, and return what the turn adds to name its preset."""

    chat: dict[str, Any] = {}
    if "chat" in sizes:
        chat["generation_settings_json"] = {"image": sizes["chat"]}
    if "chat preset" in sizes:
        chat["generation_preset_ids_json"] = {"image": _preset(sizes["chat preset"])}
    project: dict[str, Any] = {}
    if "project" in sizes:
        project["generation_settings_json"] = {"image": sizes["project"]}
    if "project preset" in sizes:
        project["generation_preset_ids_json"] = {"image": _preset(sizes["project preset"])}
    if project:
        created = await client.post("/api/projects", json={"name": "Saved size", **project})
        assert created.status_code == 201, created.text
        chat["project_id"] = created.json()["id"]
    if chat:
        saved = await client.patch(f"/api/chats/{chat_id}", json=chat)
        assert saved.status_code == 200, saved.text
    if "default preset" in sizes:
        default = await client.post(
            "/api/presets",
            json={
                "name": new_id("default"),
                "role": "image",
                "settings": sizes["default preset"],
                "is_default": True,
            },
        )
        assert default.status_code == 201, default.text
    with SessionLocal() as session:
        profile = session.get(ModelProfile, profile_id)
        assert profile is not None
        profile.request_settings_json = sizes.get("model profile", {})
        profile.load_settings_json = sizes.get("model profile load settings", {})
        session.commit()
    if "turn preset" in sizes:
        return {"preset_id": _preset(sizes["turn preset"])}
    return {}


async def _send(
    app: FastAPI, client: AsyncClient, chat_id: str, revision_id: str, profile_id: str, **turn: Any
) -> Any:
    async with app.state.services.scheduler.lease("primary"):
        return await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": "A plain gray square on a white table",
                "mode": "image",
                "workflow_revision_id": revision_id,
                "profile_id": profile_id,
                **turn,
            },
        )


def _ran_at(response: Any) -> tuple[int, int]:
    assert response.status_code == 202, response.text
    with SessionLocal() as session:
        run = session.get(Run, response.json()["run"]["id"])
        assert run is not None
        return run.settings_json["width"], run.settings_json["height"]


def _refused_for(response: Any, key: str, value: int, where: str) -> str:
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "turn-size-unsupported", body
    detail = str(body["detail"])
    assert f"The {key} of {value} px saved {where} is not one this workflow makes" in detail
    return detail


async def _capped(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str, str]:
    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch, CAPPED)
    return revision_id, profile_id, await _chat(client)


@pytest.mark.parametrize("layer", sorted(SAVED))
async def test_a_saved_size_the_workflow_cannot_make_is_refused_and_says_why(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, layer: str
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)
    turn = await _save(client, chat_id, profile_id, {layer: TOO_WIDE})
    before = _generation_row_counts()

    response = await _send(app, client, chat_id, revision_id, profile_id, **turn)

    detail = _refused_for(response, "width", 2176, SAVED[layer])
    assert "(256 to 2048 px, in steps of 64). Change it there" in detail
    assert _generation_row_counts() == before, "nothing was queued"


@pytest.mark.parametrize("layer", sorted(SAVED))
async def test_a_saved_size_the_workflow_makes_is_the_size_it_makes(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, layer: str
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)
    turn = await _save(client, chat_id, profile_id, {layer: FITS})

    response = await _send(app, client, chat_id, revision_id, profile_id, **turn)

    assert _ran_at(response) == (2048, 768)


@pytest.mark.parametrize(("higher", "lower"), list(pairwise(SAVED)))
async def test_the_higher_of_two_saved_sizes_decides_whether_it_is_refused(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, higher: str, lower: str
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)
    turn = await _save(client, chat_id, profile_id, {higher: TOO_WIDE, lower: FITS})
    refused = await _send(app, client, chat_id, revision_id, profile_id, **turn)

    other = await _chat(client)
    turn = await _save(client, other, profile_id, {higher: FITS, lower: TOO_WIDE})
    made = await _send(app, client, other, revision_id, profile_id, **turn)

    _refused_for(refused, "width", 2176, SAVED[higher])
    assert _ran_at(made) == (2048, 768)


async def _recipe(client: AsyncClient, size: dict[str, Any]) -> None:
    created = await client.post(
        "/api/workflow-use-case-presets",
        json={
            "name": "Square recipe",
            "use_case": "image_generation",
            "settings_json": size,
            "is_default": True,
        },
    )
    assert created.status_code == 201, created.text


async def test_a_recipe_outranks_a_saved_size_below_the_turns_own_preset(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)
    await _recipe(client, {"width": 1024, "height": 1024})
    await _save(client, chat_id, profile_id, {"chat": TOO_WIDE})

    response = await _send(app, client, chat_id, revision_id, profile_id)

    assert _ran_at(response) == (1024, 1024)


async def test_the_turns_own_preset_outranks_a_recipe_and_is_refused_whole(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)
    await _recipe(client, {"width": 1024, "height": 1024})

    refused = await _send(
        app, client, chat_id, revision_id, profile_id, preset_id=_preset(TOO_WIDE)
    )
    made = await _send(app, client, chat_id, revision_id, profile_id, preset_id=_preset(FITS))

    _refused_for(refused, "width", 2176, SAVED["turn preset"])
    assert _ran_at(made) == (2048, 768)


async def test_a_saved_height_it_cannot_make_is_refused_not_dropped_beside_its_width(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)
    await _save(client, chat_id, profile_id, {"chat": {"width": 1024, "height": 2240}})

    response = await _send(app, client, chat_id, revision_id, profile_id)

    _refused_for(response, "height", 2240, SAVED["chat"])


async def test_the_turns_own_size_outranks_a_saved_one_the_workflow_cannot_make(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)
    await _save(client, chat_id, profile_id, {"chat": TOO_WIDE})

    chosen = await _send(
        app, client, chat_id, revision_id, profile_id, settings={"width": 1024, "height": 512}
    )
    half = await _send(app, client, chat_id, revision_id, profile_id, settings={"height": 512})

    assert _ran_at(chosen) == (1024, 512)
    _refused_for(half, "width", 2176, SAVED["chat"])


async def test_the_turns_own_size_past_the_workflows_is_refused_as_before(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)

    response = await _send(app, client, chat_id, revision_id, profile_id, settings=TOO_WIDE)

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "turn-invalid"


@pytest.mark.parametrize(
    "dimension",
    [{"type": "integer", "const": 1024}, {"type": "integer", "readOnly": True}],
    ids=["fixed", "not offered"],
)
async def test_a_workflow_that_fixes_its_size_or_takes_none_keeps_its_own(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, dimension: dict[str, Any]
) -> None:
    """Nothing to choose, so a saved size is set aside, as it always was."""

    revision_id, profile_id = await _picture_workflow(
        app, client, monkeypatch, _schema(dict(dimension), dict(dimension))
    )
    chat_id = await _chat(client)
    await _save(client, chat_id, profile_id, {"chat": TOO_WIDE})

    response = await _send(app, client, chat_id, revision_id, profile_id)

    assert response.status_code == 202, response.text
    if "const" in dimension:
        assert _ran_at(response) == (1024, 1024)


async def test_a_saved_size_outside_a_workflows_choices_is_refused_naming_them(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    offered = {"type": "integer", "enum": [512, 1024], "default": 1024}
    revision_id, profile_id = await _picture_workflow(
        app, client, monkeypatch, _schema(dict(offered), dict(offered))
    )
    chat_id = await _chat(client)
    await _save(client, chat_id, profile_id, {"chat": {"width": 768, "height": 512}})

    response = await _send(app, client, chat_id, revision_id, profile_id)

    detail = _refused_for(response, "width", 768, SAVED["chat"])
    assert "(512, 1024 px)" in detail


async def test_a_saved_video_size_the_workflow_cannot_make_is_refused(
    client: AsyncClient,
) -> None:
    span = {"type": "integer", "default": 768, "minimum": 128, "maximum": 1536, "multipleOf": 32}
    workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Capped video",
            "operation": "text_to_video",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "Mock"}},
            "input_schema": {
                "type": "object",
                "properties": {
                    "width": dict(span),
                    "height": dict(span),
                    "frames": {"type": "integer", "default": 81, "minimum": 1},
                    "fps": {"type": "number", "default": 16, "minimum": 1},
                },
            },
        },
    )
    assert workflow.status_code == 201, workflow.text
    revision_id = workflow.json()["current_revision_id"]
    seed_workflow_trust(revision_id)
    project = await client.post(
        "/api/projects", json={"name": "Capped video", "video_workflow_revision_id": revision_id}
    )
    assert project.status_code == 201, project.text
    chat = await client.post(
        "/api/chats", json={"title": "Capped video", "project_id": project.json()["id"]}
    )
    assert chat.status_code == 201, chat.text
    saved = await client.patch(
        f"/api/chats/{chat.json()['id']}",
        json={"generation_settings_json": {"video": {"width": 1920, "height": 1088}}},
    )
    assert saved.status_code == 200, saved.text

    response = await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={"text": "A slow camera move across a gray table", "mode": "video"},
    )

    detail = _refused_for(response, "width", 1920, SAVED["chat"])
    assert "(128 to 1536 px, in steps of 32)" in detail
