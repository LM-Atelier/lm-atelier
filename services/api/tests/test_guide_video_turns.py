"""A video turn that starts from a picture may bring guide videos for a workflow that reads them.

Everything a guide video depends on is checked when the turn is accepted: that
the chosen workflow has a slot for each video and is handed a video for each
slot it names. A turn that would leave one unread, or one empty, is refused
before anything is queued rather than failing once the run starts.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_media_turn_input_kinds import _stored
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.domain import PartType
from local_lm.models import Message, MessagePart, Run


def _graph(*video_slots: str, picture: bool = True) -> dict[str, Any]:
    """A video graph that reads the named video slots, and a picture unless told not to."""
    graph: dict[str, Any] = {
        "encode": {"class_type": "CLIPTextEncode", "inputs": {"text": "${prompt}"}},
        "save": {"class_type": "SaveVideo", "inputs": {"images": ["encode", 0]}},
    }
    if picture:
        graph["picture"] = {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}}
    for index, slot in enumerate(video_slots):
        graph[f"clip_{index}"] = {"class_type": "LoadVideo", "inputs": {"file": f"${{{slot}}}"}}
    return graph


def _object_info(graph: dict[str, Any]) -> dict[str, Any]:
    """How the fixture runtime describes each node the graph uses."""
    return {
        node["class_type"]: {
            "python_module": "nodes",
            "input": {"required": {key: ["*", {}] for key in node["inputs"]}},
            "output": [] if node["class_type"] == "SaveVideo" else ["*"],
        }
        for node in graph.values()
    }


async def _reviewed_workflow(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *video_slots: str,
    operation: str = "image_to_video",
    engine: str = "comfyui",
) -> str:
    """Create a workflow reading the named video slots, and approve it."""
    graph = _graph(*video_slots, picture=operation.startswith("image_to"))

    async def describe() -> dict[str, Any]:
        return _object_info(graph)

    monkeypatch.setattr(app.state.services.engines.media, "object_info", describe, raising=False)
    properties: dict[str, Any] = {
        "prompt": {"type": "string"},
        **({"input_image": {"type": "string"}} if "picture" in graph else {}),
        **{slot: {"type": "string"} for slot in video_slots},
    }
    created = await client.post(
        "/api/workflows",
        json={
            "name": name,
            "operation": operation,
            "engine": engine,
            "api_graph": graph,
            "input_schema": {"type": "object", "properties": properties},
        },
    )
    assert created.status_code == 201, created.text
    definition = created.json()
    revision_id = str(definition["current_revision_id"])
    url = f"/api/workflows/{definition['id']}/revisions/{revision_id}/review"
    preview = await client.get(url)
    assert preview.status_code == 200, preview.text
    reviewed = await client.post(
        url, json={"action": "approve", "subject_sha256": preview.json()["subject_sha256"]}
    )
    assert reviewed.status_code == 200, reviewed.text
    return revision_id


def _inputs() -> None:
    _stored("guide-video", "video/mp4", "a")
    _stored("still-picture", "image/png", "b")
    _stored("second-video", "video/webm", "c")
    _stored("second-picture", "image/jpeg", "d")
    _stored("sound", "audio/mpeg", "e")


async def _turn(
    app: FastAPI, client: AsyncClient, **fields: Any
) -> tuple[int, dict[str, Any], str]:
    chat = (await client.post("/api/chats", json={"title": "Guide videos"})).json()
    # Held, so an accepted turn stays queued instead of reaching the engine.
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": "The leaves drift across the frame.", "mode": "video", **fields},
        )
    return response.status_code, response.json(), chat["id"]


@pytest.fixture
def comfyui(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")


@pytest.mark.usefixtures("comfyui")
async def test_a_picture_and_its_guide_video_run_on_a_workflow_that_reads_both(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attached video first, the picture still becomes the input the run starts from."""
    revision_id = await _reviewed_workflow(app, client, monkeypatch, "Guided", "input_video")
    _inputs()

    status, body, _ = await _turn(
        app,
        client,
        input_artifact_ids=["guide-video", "still-picture"],
        workflow_revision_id=revision_id,
        confirm_media=True,
    )

    assert status == 202, body
    with SessionLocal() as session:
        run = session.get(Run, body["run"]["id"])
        assert run is not None
        assert run.workflow_revision_id == revision_id
        assert run.provenance_json["input_artifact_ids"] == ["still-picture", "guide-video"]
        message = session.get(Message, run.user_message_id)
        assert message is not None
        parts = session.scalars(
            select(MessagePart)
            .where(MessagePart.message_id == message.id)
            .order_by(MessagePart.position)
        ).all()
        inputs = [(part.type, part.artifact_id) for part in parts if part.artifact_id]
        assert inputs == [(PartType.IMAGE, "still-picture"), (PartType.VIDEO, "guide-video")]


@pytest.mark.usefixtures("comfyui")
async def test_the_apps_own_choice_runs_the_workflow_that_reads_the_guide_video(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    guided = await _reviewed_workflow(app, client, monkeypatch, "Guided", "input_video")
    plain = await _reviewed_workflow(app, client, monkeypatch, "Plain")
    _inputs()

    status, body, _ = await _turn(
        app, client, input_artifact_ids=["still-picture", "guide-video"], confirm_media=True
    )
    assert status == 202, body
    plain_status, plain_body, _ = await _turn(
        app, client, input_artifact_ids=["still-picture"], confirm_media=True
    )
    assert plain_status == 202, plain_body

    with SessionLocal() as session:
        guided_run = session.get(Run, body["run"]["id"])
        plain_run = session.get(Run, plain_body["run"]["id"])
        assert guided_run is not None and plain_run is not None
        assert guided_run.workflow_revision_id == guided
        assert plain_run.workflow_revision_id == plain


@pytest.mark.usefixtures("comfyui")
async def test_only_pictures_count_against_the_pictures_a_workflow_takes(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id = await _reviewed_workflow(app, client, monkeypatch, "Guided", "input_video")
    _inputs()

    status, body, chat_id = await _turn(
        app,
        client,
        input_artifact_ids=["still-picture", "second-picture", "guide-video"],
        workflow_revision_id=revision_id,
        confirm_media=True,
    )

    assert status == 422, body
    assert body["detail"] == (
        "This workflow uses 1 reference image, and 2 were attached. Choose a workflow "
        "built for multiple references, or attach fewer."
    )
    assert (await client.get(f"/api/chats/{chat_id}")).json()["messages"] == []


@pytest.mark.parametrize(
    "video_slots,inputs,refusal",
    [
        (
            (),
            ["still-picture", "guide-video"],
            "This workflow does not take a guide video. Choose a workflow built for one, "
            "or attach only pictures.",
        ),
        (
            ("input_video",),
            ["still-picture"],
            "This workflow needs a guide video. Attach one beside the picture, or choose "
            "another workflow.",
        ),
        (
            ("input_video",),
            ["still-picture", "guide-video", "second-video"],
            "This workflow does not read attached video 2 of 2. Choose a workflow built for "
            "more guide videos, or attach fewer.",
        ),
        (
            ("input_video_1",),
            ["still-picture", "guide-video"],
            "This workflow needs 2 guide videos, and 1 was attached.",
        ),
        (
            ("input_video", "input_video_1"),
            ["still-picture"],
            "This workflow needs 2 guide videos. Attach them beside the picture, or choose "
            "another workflow.",
        ),
        (
            ("input_video_1",),
            ["still-picture", "guide-video", "second-video"],
            "This workflow does not read attached video 1 of 2. Choose a workflow that reads "
            "every attached video.",
        ),
    ],
    ids=[
        "reads-no-video",
        "waits-for-a-video",
        "a-video-too-many",
        "a-video-too-few",
        "waits-for-two-videos",
        "skips-a-slot",
    ],
)
@pytest.mark.usefixtures("comfyui")
async def test_a_turn_whose_videos_do_not_fit_the_named_workflow_is_refused_before_it_runs(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    video_slots: tuple[str, ...],
    inputs: list[str],
    refusal: str,
) -> None:
    revision_id = await _reviewed_workflow(app, client, monkeypatch, "Named", *video_slots)
    _inputs()

    status, body, chat_id = await _turn(
        app,
        client,
        input_artifact_ids=inputs,
        workflow_revision_id=revision_id,
        confirm_media=True,
    )

    assert status == 422, body
    assert body["code"] == "turn-invalid"
    assert body["detail"] == refusal
    assert (await client.get(f"/api/chats/{chat_id}")).json()["messages"] == []


@pytest.mark.parametrize("media_type", ["video/mpeg", "video/ogg", "video/x-ms-wmv"])
@pytest.mark.usefixtures("comfyui")
async def test_a_video_in_a_format_comfyui_cannot_load_is_refused_before_it_runs(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, media_type: str
) -> None:
    revision_id = await _reviewed_workflow(app, client, monkeypatch, "Guided", "input_video")
    _inputs()
    _stored("other-format", media_type, "f")

    status, body, chat_id = await _turn(
        app,
        client,
        input_artifact_ids=["still-picture", "other-format"],
        workflow_revision_id=revision_id,
        confirm_media=True,
    )

    assert status == 422, body
    assert body["detail"] == (
        "This video cannot guide a workflow. Attach an MP4, MOV, WebM, MKV or AVI file."
    )
    assert (await client.get(f"/api/chats/{chat_id}")).json()["messages"] == []


@pytest.mark.parametrize("mode", ["video", "image"])
@pytest.mark.usefixtures("comfyui")
async def test_any_other_file_beside_a_picture_is_still_refused(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    await _reviewed_workflow(app, client, monkeypatch, "Plain")
    _inputs()

    status, body, chat_id = await _turn(
        app, client, mode=mode, input_artifact_ids=["still-picture", "sound"], confirm_media=True
    )

    assert status == 422, body
    assert body["detail"] == "This media step requires image-compatible inputs."
    assert (await client.get(f"/api/chats/{chat_id}")).json()["messages"] == []


@pytest.mark.parametrize(
    "operation,mode,inputs",
    [("image_to_image", "image", ["still-picture"]), ("text_to_video", "video", [])],
    ids=["picture-edit", "video-from-words"],
)
@pytest.mark.usefixtures("comfyui")
async def test_a_workflow_waiting_for_a_guide_video_is_refused_where_no_video_can_come(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    mode: str,
    inputs: list[str],
) -> None:
    revision_id = await _reviewed_workflow(
        app, client, monkeypatch, "Waiting", "input_video", operation=operation
    )
    _inputs()

    status, body, chat_id = await _turn(
        app,
        client,
        mode=mode,
        input_artifact_ids=inputs,
        workflow_revision_id=revision_id,
        confirm_media=True,
    )

    assert status == 422, body
    assert body["detail"] == (
        "This workflow needs a guide video, which only a video made from a picture can "
        "bring. Choose another workflow."
    )
    assert (await client.get(f"/api/chats/{chat_id}")).json()["messages"] == []


async def test_an_engine_that_binds_videos_its_own_way_is_left_to_decide(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mock engine reads no video slots, so the slot checks are ComfyUI's alone."""
    revision_id = await _reviewed_workflow(
        app, client, monkeypatch, "Mock waiting", "input_video", engine="mock"
    )
    _inputs()

    status, body, _ = await _turn(
        app,
        client,
        input_artifact_ids=["still-picture"],
        workflow_revision_id=revision_id,
        confirm_media=True,
    )

    assert status == 202, body


ORDERED = "Animate the image into a video, then summarize the video."


@pytest.mark.usefixtures("comfyui")
async def test_a_step_of_a_longer_request_passes_over_a_workflow_waiting_for_a_video(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = await _reviewed_workflow(app, client, monkeypatch, "Plain")
    guided = await _reviewed_workflow(app, client, monkeypatch, "Guided", "input_video")
    _inputs()

    status, body, chat_id = await _turn(
        app,
        client,
        text=ORDERED,
        mode="auto",
        input_artifact_ids=["still-picture"],
        confirm_media=True,
    )

    assert status == 202, body
    with SessionLocal() as session:
        chosen = session.scalars(
            select(Run.workflow_revision_id).where(
                Run.chat_id == chat_id, Run.operation == "image_to_video"
            )
        ).all()
    assert chosen == [plain]
    assert guided not in chosen


@pytest.mark.usefixtures("comfyui")
async def test_a_step_named_a_workflow_waiting_for_a_video_is_refused_before_it_runs(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    guided = await _reviewed_workflow(app, client, monkeypatch, "Guided", "input_video")
    _inputs()

    status, body, chat_id = await _turn(
        app,
        client,
        text=ORDERED,
        mode="auto",
        input_artifact_ids=["still-picture"],
        role_overrides={"video": {"workflow_revision_id": guided}},
        confirm_media=True,
    )

    assert status == 422, body
    assert body["detail"] == (
        "Ordered step 1 selected a workflow that needs a guide video, which no step of a "
        "longer request can bring."
    )
    assert (await client.get(f"/api/chats/{chat_id}")).json()["messages"] == []
