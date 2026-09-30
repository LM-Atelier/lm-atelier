"""A default shape sizes new work only where nothing else chooses a size."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_output_geometry_api import _graph, _schema, _trust
from test_workflow_output_geometry_api import geometry_runtime as geometry_runtime

from local_lm.db import SessionLocal
from local_lm.default_output_shape import default_output_size, size_is_chosen
from local_lm.domain import Operation
from local_lm.models import Chat, GenerationPreset, ModelProfile, Project, Run, WorkflowRevision

CHOSEN = {"width": 512, "height": 768}
# The fixture workflow's own size, from its schema defaults: a 4:3 picture.
WORKFLOW_SIZE = (1024, 768)


async def _picture_workflow(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    schema: dict[str, object] | None = None,
) -> tuple[str, str]:
    """A trusted picture workflow whose size the geometry proof covers, and a profile for it."""
    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    # A turn's work starts as soon as the lease is let go. With the engine set to
    # ComfyUI, starting or restarting that worker would first set up its runtime by
    # downloading it; the mock engine that runs the work here needs no worker.
    monkeypatch.setattr(app.state.services.processes, "start_media", AsyncMock())
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Default shape fixture",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _graph(),
            "input_schema": schema if schema is not None else _schema(),
        },
    )
    assert created.status_code == 201, created.text
    revision_id: str = created.json()["current_revision_id"]
    _trust(revision_id)
    with SessionLocal() as session:
        profile = ModelProfile(name="Default shape fixture", role="image", engine="comfyui")
        session.add(profile)
        session.commit()
        profile_id = profile.id
    return revision_id, profile_id


async def _chat(client: AsyncClient) -> str:
    chat = await client.post("/api/chats", json={"title": "Default shape"})
    assert chat.status_code == 201, chat.text
    chat_id: str = chat.json()["id"]
    return chat_id


async def _picture(
    app: FastAPI,
    client: AsyncClient,
    chat_id: str,
    revision_id: str,
    profile_id: str,
    **turn: Any,
) -> Run:
    """Send a picture turn and read back the run it queued, before anything runs it."""
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": "A plain gray square on a white table",
                "mode": "image",
                "workflow_revision_id": revision_id,
                "profile_id": profile_id,
                **turn,
            },
        )
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            run = session.get(Run, response.json()["run"]["id"])
            assert run is not None
            session.expunge(run)
    return run


async def _resolved(client: AsyncClient, revision_id: str, preset_id: str) -> tuple[int, int]:
    """What the composer's shape button resolves the same shape to on the same revision."""
    answer = await client.post(
        f"/api/workflow-revisions/{revision_id}/output-geometry/resolve",
        json={"mode": "image", "size_mode": "preset", "preset_id": preset_id},
    )
    assert answer.status_code == 200, answer.text
    return answer.json()["width"], answer.json()["height"]


async def test_a_default_shape_sizes_a_new_picture_when_nothing_else_chooses_one(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch)
    chat_id = await _chat(client)

    plain = await _picture(app, client, chat_id, revision_id, profile_id)
    shaped = await _picture(
        app,
        client,
        chat_id,
        revision_id,
        profile_id,
        default_output_shapes={"image": "3:2"},
    )

    # Without a default the workflow keeps its own size, and says nothing about a shape.
    assert (plain.settings_json["width"], plain.settings_json["height"]) == WORKFLOW_SIZE
    assert "output_shape" not in plain.provenance_json
    # With one, the picture takes the size the composer's button would have chosen.
    width, height = shaped.settings_json["width"], shaped.settings_json["height"]
    assert (width, height) == await _resolved(client, revision_id, "3:2")
    assert width * 2 == height * 3
    assert shaped.provenance_json["output_shape"] == {
        "source": "default",
        "preset_id": "3:2",
        "width": width,
        "height": height,
    }
    assert shaped.provenance_json["resolved_settings"]["width"] == width


def _saved_size(layer: str, profile_id: str, chat_id: str) -> None:
    """Save the chosen size in one layer below the turn."""
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        profile = session.get(ModelProfile, profile_id)
        assert chat is not None and profile is not None
        if layer == "profile load":
            profile.load_settings_json = dict(CHOSEN)
        elif layer == "profile request":
            profile.request_settings_json = dict(CHOSEN)
        elif layer == "chat":
            chat.generation_settings_json = {"image": dict(CHOSEN)}
        elif layer == "project":
            project = Project(
                name="Default shape project", generation_settings_json={"image": dict(CHOSEN)}
            )
            session.add(project)
            session.flush()
            chat.project_id = project.id
        else:
            preset = GenerationPreset(
                name=f"Sized {layer}",
                role="image",
                settings_json=dict(CHOSEN),
                is_default=layer == "default preset",
            )
            session.add(preset)
            session.flush()
            if layer == "chat preset":
                chat.generation_preset_ids_json = {"image": preset.id}
            elif layer == "project preset":
                project = Project(
                    name="Default shape preset project",
                    generation_preset_ids_json={"image": preset.id},
                )
                session.add(project)
                session.flush()
                chat.project_id = project.id
        session.commit()


@pytest.mark.parametrize(
    "layer",
    [
        "profile load",
        "profile request",
        "default preset",
        "project preset",
        "project",
        "chat preset",
        "chat",
    ],
)
async def test_a_size_saved_in_any_layer_outranks_the_default(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, layer: str
) -> None:
    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch)
    chat_id = await _chat(client)
    _saved_size(layer, profile_id, chat_id)

    run = await _picture(
        app,
        client,
        chat_id,
        revision_id,
        profile_id,
        default_output_shapes={"image": "3:2"},
    )

    assert (run.settings_json["width"], run.settings_json["height"]) == (512, 768)
    assert "output_shape" not in run.provenance_json


async def test_a_size_the_turn_chooses_outranks_the_default_whole(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch)
    chat_id = await _chat(client)
    with SessionLocal() as session:
        preset = GenerationPreset(
            name="Sized turn preset", role="image", settings_json=dict(CHOSEN)
        )
        session.add(preset)
        session.commit()
        preset_id = preset.id
    shapes = {"image": "3:2"}

    chosen = await _picture(
        app, client, chat_id, revision_id, profile_id, default_output_shapes=shapes, settings=CHOSEN
    )
    from_preset = await _picture(
        app,
        client,
        chat_id,
        revision_id,
        profile_id,
        default_output_shapes=shapes,
        preset_id=preset_id,
    )
    # A height alone still keeps the default out: a size is never half one choice
    # and half another, so the width is the workflow's own.
    tall = await _picture(
        app,
        client,
        chat_id,
        revision_id,
        profile_id,
        default_output_shapes=shapes,
        settings={"height": 512},
    )

    for run in (chosen, from_preset):
        assert (run.settings_json["width"], run.settings_json["height"]) == (512, 768)
        assert "output_shape" not in run.provenance_json
    assert (tall.settings_json["width"], tall.settings_json["height"]) == (WORKFLOW_SIZE[0], 512)
    assert "output_shape" not in tall.provenance_json


async def test_a_video_default_leaves_a_picture_at_its_own_size(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch)
    chat_id = await _chat(client)

    run = await _picture(
        app,
        client,
        chat_id,
        revision_id,
        profile_id,
        default_output_shapes={"video": "16:9"},
    )

    assert (run.settings_json["width"], run.settings_json["height"]) == WORKFLOW_SIZE
    assert "output_shape" not in run.provenance_json


async def test_an_ordered_plan_sizes_its_picture_step_by_the_default(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _picture_workflow(app, client, monkeypatch)
    chat_id = await _chat(client)

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": "Write a short story about a paper boat, then create an image based on it",
                "mode": "auto",
                "confirm_media": True,
                "default_output_shapes": {"image": "3:2"},
                "role_overrides": {
                    "image": {"workflow_revision_id": revision_id, "profile_id": profile_id}
                },
            },
        )
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            runs = {
                run.operation: run
                for run in session.scalars(
                    select(Run).where(Run.work_plan_id == response.json()["run"]["work_plan_id"])
                )
            }
            assert set(runs) == {"text", "text_to_image"}
            picture, story = runs["text_to_image"], runs["text"]
            width, height = picture.settings_json["width"], picture.settings_json["height"]
            assert picture.provenance_json["output_shape"]["preset_id"] == "3:2"
            assert "output_shape" not in story.provenance_json

    assert (width, height) == await _resolved(client, revision_id, "3:2")


async def test_the_default_applies_only_to_work_made_from_nothing(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, _ = await _picture_workflow(app, client, monkeypatch)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        made = default_output_size(session, Operation.TEXT_TO_IMAGE, revision, "1:1", None)
        assert made is not None and made.width == made.height
        # An edit keeps its source's shape, a video made from a picture follows
        # the picture, and text has no size at all.
        for operation in (Operation.IMAGE_TO_IMAGE, Operation.IMAGE_TO_VIDEO, Operation.TEXT):
            assert default_output_size(session, operation, revision, "1:1", "1:1") is None
        # The preference for the other kind of output, or none at all, is no default.
        assert default_output_size(session, Operation.TEXT_TO_IMAGE, revision, None, "1:1") is None
        assert default_output_size(session, Operation.TEXT_TO_IMAGE, None, "1:1", None) is None


async def test_a_workflow_that_cannot_make_the_shape_exactly_keeps_its_own_size(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sides of 64 or 128 only: squares and 1:2 or 2:1, never 3:2.
    side = {"type": "integer", "default": 128, "minimum": 64, "maximum": 128, "multipleOf": 64}
    revision_id, _ = await _picture_workflow(
        app, client, monkeypatch, schema=_schema(width=side, height=side)
    )
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        assert default_output_size(session, Operation.TEXT_TO_IMAGE, revision, "3:2", None) is None
        assert (
            default_output_size(session, Operation.TEXT_TO_IMAGE, revision, "1:1", None) is not None
        )
        # An untrusted revision has no proof, so it keeps its own size too.
        revision.trusted = False
        assert default_output_size(session, Operation.TEXT_TO_IMAGE, revision, "1:1", None) is None


def test_only_a_named_width_or_height_counts_as_a_chosen_size() -> None:
    assert not size_is_chosen([None, {}, {"steps": 4}])
    assert size_is_chosen([None, {"width": 512}])
    assert size_is_chosen([{}, {"height": 512}])
