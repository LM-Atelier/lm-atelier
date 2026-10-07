"""A remix can start from the picture itself, and changes it exactly as previewed."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from httpx2 import AsyncClient as CurrentAsyncClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from sqlalchemy import func, select
from test_external_generation_metadata import _graph as _file_graph
from test_picture_remix import PROMPT, _new_chat, _runs
from test_source_crop_acceptance import object_info
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_crop_route import graph

from local_lm import picture_remix
from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    Job,
    Message,
    MessagePart,
    ModelProfile,
    Run,
    RunContextArtifact,
)
from local_lm.output_recipe_replay import edit_prompt_preamble

COUNTED = (Chat, Message, MessagePart, Run, Job, RunContextArtifact, ArtifactLibraryEntry, Artifact)


def _picture(
    size: tuple[int, int] = (512, 768),
    *,
    said: str = "512x768",
    strength: str = "0.45",
    turned: bool = False,
    frames: int = 1,
    graph: dict[str, Any] | None = None,
) -> bytes:
    text = (
        f"{PROMPT}\nSteps: 20, Sampler: Euler a, Seed: 12345, Size: {said}, "
        f"Denoising strength: {strength}"
    )
    info = PngInfo()
    if graph is None:
        info.add_text("parameters", text)
    else:
        info.add_text("prompt", json.dumps(graph))
    exif = Image.Exif()
    if turned:
        exif[0x0112] = 6
    pictures = [Image.new("RGB", size, (90, 120, 150 - index)) for index in range(frames)]
    output = BytesIO()
    pictures[0].save(
        output,
        format="PNG",
        pnginfo=info,
        exif=exif,
        save_all=frames > 1,
        append_images=pictures[1:],
    )
    return output.getvalue()


async def _uploaded(client: AsyncClient | CurrentAsyncClient, content: bytes) -> str:
    """A picture added to the Media Library, as its own upload adds one."""

    response = await client.post(
        "/api/artifacts",
        params={"kind": "image"},
        files={"file": ("picture.png", content, "image/png")},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _edit_workflow(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    api_graph: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """A reviewed workflow that changes the picture it is given, and a model for it."""

    workflow = graph() if api_graph is None else api_graph
    workflow["positive"]["inputs"]["text"] = "$" + "{prompt}"

    async def describe() -> dict[str, Any]:
        return object_info(workflow)

    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    monkeypatch.setattr(app.state.services.engines.media, "object_info", describe, raising=False)
    # Work starts once the lease is let go; the engine's own runtime is never set up here.
    monkeypatch.setattr(app.state.services.processes, "start_media", AsyncMock())
    with SessionLocal() as session:
        profile = ModelProfile(name="Edit fixture", role="image", engine="comfyui")
        session.add(profile)
        session.commit()
        profile_id = profile.id
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Edit fixture",
            "operation": "image_to_image",
            "engine": "comfyui",
            "api_graph": workflow,
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "input_image": {"type": "string"},
                    "checkpoint": {"type": "string", "default": "fixture.safetensors"},
                    "denoise": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.6},
                },
            },
        },
    )
    assert created.status_code == 201, created.text
    definition = created.json()
    revision_id: str = definition["current_revision_id"]
    url = f"/api/workflows/{definition['id']}/revisions/{revision_id}/review"
    review = await client.get(url)
    assert review.status_code == 200, review.text
    approved = await client.post(
        url, json={"action": "approve", "subject_sha256": review.json()["subject_sha256"]}
    )
    assert approved.status_code == 200, approved.text
    return revision_id, profile_id


async def _preview(
    client: AsyncClient,
    artifact_id: str,
    revision_id: str,
    profile_id: str,
    apply: list[str] | None = None,
    role: str = "edit",
) -> Any:
    return await client.post(
        f"/api/artifacts/{artifact_id}/remix-preview",
        json={
            "workflow_revision_id": revision_id,
            "profile_id": profile_id,
            "apply": apply or [],
            "role": role,
        },
    )


async def _queue(
    client: AsyncClient, chat_id: str, artifact_id: str, preview: dict[str, Any], apply: list[str]
) -> Any:
    return await client.post(
        f"/api/chats/{chat_id}/remixes",
        json={
            "artifact_id": artifact_id,
            "workflow_revision_id": preview["workflow_revision_id"],
            "profile_id": preview["profile_id"],
            "apply": apply,
            "role": preview["role"],
            "review_digest": preview["review_digest"],
        },
    )


def _states(body: dict[str, Any]) -> dict[str, tuple[str | None, str, str | None, bool]]:
    return {
        claim["key"]: (claim["setting"], claim["state"], claim["reason"], claim["applied"])
        for claim in body["claims"]
    }


def _counts() -> tuple[int, ...]:
    with SessionLocal() as session:
        return tuple(
            int(session.scalar(select(func.count()).select_from(model)) or 0) for model in COUNTED
        )


async def test_a_remix_can_start_from_the_picture_itself(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch)
    artifact_id = await _uploaded(client, _picture())
    before = _counts()

    estimated = (await _preview(client, artifact_id, revision_id, profile_id)).json()
    chosen = (await _preview(client, artifact_id, revision_id, profile_id, ["denoise"])).json()

    assert _counts() == before
    assert estimated["ready"] is True, estimated
    assert (estimated["role"], estimated["operation"]) == ("edit", "image_to_image")
    assert estimated["source"] == {"width": 512, "height": 768}
    states = _states(estimated)
    assert states["denoise"] == ("denoise", "supported", None, False)
    # Its size is kept, not set: nothing applies it, and it is used.
    assert states["width"] == (None, "supported", None, True)
    assert states["height"] == (None, "supported", None, True)
    # The graph fixes these, so the engine's own list of settings does not offer them.
    for key in ("steps", "seed", "sampler"):
        assert states[key][1:3] == ("incompatible", "not_offered")
    resolved = estimated["resolved"]
    assert resolved["engine_prompt"] == f"{edit_prompt_preamble()}{PROMPT}"
    assert resolved["strength"]["mode"] == "auto"
    assert resolved["strength"]["from_file"] is False
    assert chosen["resolved"]["strength"] == {
        "parameter": "denoise",
        "mode": "manual",
        "value": 0.45,
        "from_file": True,
    }
    assert chosen["resolved"]["settings"]["denoise"] == 0.45
    assert chosen["review_digest"] != estimated["review_digest"]


async def test_a_remix_starting_from_the_picture_is_queued_as_it_was_previewed(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch)
    artifact_id = await _uploaded(client, _picture())
    preview = (await _preview(client, artifact_id, revision_id, profile_id, ["denoise"])).json()
    chat_id = await _new_chat(client)

    async with app.state.services.scheduler.lease("primary"):
        response = await _queue(client, chat_id, artifact_id, preview, ["denoise"])

        assert response.status_code == 202, response.text
        [run] = _runs(chat_id)
        assert run.operation == "image_to_image"
        assert run.settings_json == preview["resolved"]["settings"]
        assert run.provenance_json["image_edit"]["strength"]["mode"] == "manual"
        receipt = run.provenance_json["remix"]
        assert (receipt["role"], receipt["operation"]) == ("edit", "image_to_image")
        assert receipt["strength"] == {"parameter": "denoise", "mode": "manual", "from_file": True}
        assert all(set(claim) == {"key", "state", "applied"} for claim in receipt["claims"])
        with SessionLocal() as session:
            stored = session.get(Run, run.id)
            assert stored is not None
            context = accepted_context(session, stored)
            assert context is not None
            assert context.input_artifact_ids == [artifact_id]
            assert context.media_prompt == preview["resolved"]["engine_prompt"]
            # The made picture is not checked again, which would make a second one,
            # and the chat's own setting is as it was for any later edit.
            assert context.vision_settings.get("verify_image_edits") is False
            chat = session.get(Chat, chat_id)
            assert chat is not None
            assert chat.vision_settings_json.get("verify_image_edits") is not False


async def test_a_remix_previewed_from_words_is_not_queued_as_a_change_to_the_picture(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch)
    artifact_id = await _uploaded(client, _picture())
    preview = (await _preview(client, artifact_id, revision_id, profile_id)).json()
    chat_id = await _new_chat(client)

    response = await _queue(client, chat_id, artifact_id, {**preview, "role": "words"}, [])

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "remix-unavailable"
    assert _runs(chat_id) == []


@pytest.mark.parametrize(
    ("strength", "state"),
    [
        # The most a workflow allows redraws the whole picture.
        ("1", ("incompatible", "replaces_picture")),
        ("0.2", ("supported", None)),
    ],
)
async def test_the_files_strength_is_never_offered_where_it_redraws_the_picture(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    strength: str,
    state: tuple[str, str | None],
) -> None:
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch)
    artifact_id = await _uploaded(client, _picture(strength=strength))

    body = (await _preview(client, artifact_id, revision_id, profile_id)).json()
    applied = await _preview(client, artifact_id, revision_id, profile_id, ["denoise"])

    assert _states(body)["denoise"][1:3] == state
    assert applied.status_code == (200 if state[0] == "supported" else 422)


@pytest.mark.parametrize(
    ("size", "said", "state"),
    [
        ((512, 768), "1024x1536", ("incompatible", "size_differs")),
        # The encoder trims a side to a multiple of eight.
        ((515, 768), "515x768", ("unresolved", "size_unproven")),
    ],
)
async def test_a_kept_size_is_said_only_where_it_is_the_picture_s_own(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
    said: str,
    state: tuple[str, str],
) -> None:
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch)
    artifact_id = await _uploaded(client, _picture(size, said=said))

    body = (await _preview(client, artifact_id, revision_id, profile_id)).json()
    applied = await _preview(client, artifact_id, revision_id, profile_id, ["width", "height"])

    assert body["ready"] is True
    assert _states(body)["width"] == (None, *state, False)
    assert body["shape"] is None
    assert applied.status_code == 422


def _set_aside_source() -> dict[str, Any]:
    # The picture given is loaded, but a fixed picture is what is changed.
    workflow = graph()
    workflow["unused"] = {"class_type": "LoadImage", "inputs": {"image": "$" + "{input_image}"}}
    workflow["source"]["inputs"]["image"] = "fixed.png"
    return workflow


def _fixed_strength() -> dict[str, Any]:
    workflow = graph()
    workflow["sample"]["inputs"]["denoise"] = 0.5
    return workflow


@pytest.mark.parametrize("make", [_set_aside_source, _fixed_strength])
async def test_a_workflow_that_does_not_change_the_given_picture_by_a_chosen_strength_is_refused(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, make: Any
) -> None:
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch, make())
    artifact_id = await _uploaded(client, _picture())

    body = (await _preview(client, artifact_id, revision_id, profile_id)).json()

    assert body["ready"] is False
    assert [item["code"] for item in body["refusals"]] == ["remix-edit-unusable"]


async def test_only_a_picture_the_media_library_shows_can_be_started_from(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch)
    # A file sent with a message is kept for it, and is no picture in the library.
    attached = await client.post(
        "/api/artifacts", files={"file": ("picture.png", _picture(), "image/png")}
    )
    assert attached.status_code == 201, attached.text

    # A picture the app keeps for its own work, never added to the library.
    with SessionLocal() as session:
        # Other bytes, so it is not the attached file under the same content address.
        kept = app.state.services.artifacts.ingest_bytes(
            session, _picture(strength="0.3"), kind=ArtifactKind.IMAGE, media_type="image/png"
        )
        session.commit()
        kept_id = kept.id
    assert kept_id != attached.json()["id"]

    for artifact_id in (attached.json()["id"], kept_id):
        edit = (await _preview(client, artifact_id, revision_id, profile_id)).json()

        assert [item["code"] for item in edit["refusals"]] == ["remix-picture-unusable"]
        assert edit["source"] is None


async def test_a_turned_picture_s_kept_size_is_not_promised(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch)
    # Stored 512 wide and 768 tall, and shown the other way round.
    artifact_id = await _uploaded(client, _picture(said="768x512", turned=True))

    body = (await _preview(client, artifact_id, revision_id, profile_id)).json()

    assert _states(body)["width"] == (None, "unresolved", "size_unproven", False)


async def test_a_moving_picture_is_no_starting_picture(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch)
    artifact_id = await _uploaded(client, _picture(frames=3))

    body = (await _preview(client, artifact_id, revision_id, profile_id)).json()

    assert [item["code"] for item in body["refusals"]] == ["remix-picture-unusable"]


async def test_a_prompt_with_space_around_it_is_queued_as_the_words_a_turn_takes(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch)
    graph = _file_graph()
    text_inputs = graph["6"]["inputs"]
    size_inputs = graph["5"]["inputs"]
    assert isinstance(text_inputs, dict)
    assert isinstance(size_inputs, dict)
    text_inputs["text"] = f"{PROMPT}, \n"
    size_inputs.update({"width": 512, "height": 768})
    artifact_id = await _uploaded(client, _picture(graph=graph))
    preview = (await _preview(client, artifact_id, revision_id, profile_id)).json()
    chat_id = await _new_chat(client)

    async with app.state.services.scheduler.lease("primary"):
        response = await _queue(client, chat_id, artifact_id, preview, [])

        assert response.status_code == 202, response.text
        assert preview["resolved"]["text"] == f"{PROMPT},"
        [run] = _runs(chat_id)
        assert run.standalone_prompt == f"{PROMPT},"


async def test_a_remix_is_not_made_when_the_engine_would_get_other_words_than_shown(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id = await _edit_workflow(app, client, monkeypatch)
    artifact_id = await _uploaded(client, _picture())
    shown = picture_remix._engine_prompt

    def other(*args: Any) -> str:
        return shown(*args) + " and something else"

    # The preview and the queue both show these words; the run is given the real ones.
    monkeypatch.setattr(picture_remix, "_engine_prompt", other)
    preview = (await _preview(client, artifact_id, revision_id, profile_id)).json()
    chat_id = await _new_chat(client)

    response = await _queue(client, chat_id, artifact_id, preview, [])

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "remix-differs"
    assert _runs(chat_id) == []
