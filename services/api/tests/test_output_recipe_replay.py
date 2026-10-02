"""Generating a record again exactly, through the real routes."""

from __future__ import annotations

import copy
import io
from typing import Any, cast

from httpx2 import AsyncClient
from PIL import Image
from run_waits import wait_for_terminal_status
from sqlalchemy import func, select

from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.models import (
    Chat,
    GenerationPreset,
    Job,
    Message,
    ModelInstall,
    ModelProfile,
    Run,
    RunContextSnapshot,
    WorkflowRevision,
    WorkflowUseCasePreset,
    WorkPlan,
)
from local_lm.output_recipe_replay import edit_prompt_preamble
from local_lm.output_recipe_v1 import open_output_recipe, seal_output_recipe

PROMPT = "a ceramic cup on a wooden table"
FILES = {"model.safetensors": "1" * 64}
EXECUTION = ("operation", "prompt", "seed", "settings", "inputs", "model", "loras")


def _profile(role: str = "image") -> str:
    with SessionLocal() as session:
        install = ModelInstall(
            name="Recorded model",
            role=role,
            engine="mock",
            local_path="neutral-model-folder",
            compatibility="compatible",
            manifest_json={"expected_sha256": dict(FILES)},
            active=True,
        )
        session.add(install)
        session.flush()
        profile = ModelProfile(
            name="Recorded model", role=role, engine="mock", model_install_id=install.id
        )
        session.add(profile)
        session.commit()
        return profile.id


async def _finished(client: AsyncClient, run_id: str) -> dict[str, Any]:
    async def read() -> dict[str, Any]:
        return cast(dict[str, Any], (await client.get(f"/api/runs/{run_id}")).json())

    return dict(await wait_for_terminal_status(read, what=run_id, expected="complete"))


async def _record_of(client: AsyncClient, run: dict[str, Any]) -> bytes:
    artifact_id = run["provenance_json"]["outputs"][0]["artifact_id"]
    response = await client.get(
        f"/api/runs/{run['id']}/outputs/{artifact_id}/recipe", params={"prompts": "include"}
    )
    assert response.status_code == 200, response.text
    return response.content


async def _recorded(
    client: AsyncClient,
    profile_id: str,
    *,
    mode: str = "image",
    revision_id: str | None = None,
    settings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes]:
    chat = (await client.post("/api/chats", json={"title": "Recorded"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": PROMPT,
            "mode": mode,
            "profile_id": profile_id,
            "workflow_revision_id": revision_id,
            "settings": settings if settings is not None else {"seed": 1234, "steps": 9},
        },
    )
    assert turn.status_code == 202, turn.text
    run = await _finished(client, turn.json()["run"]["id"])
    return run, await _record_of(client, run)


async def _new_chat(client: AsyncClient) -> str:
    # Titled as a new chat is, so a turn would rename it from its prompt.
    return cast(str, (await client.post("/api/chats", json={"title": "New chat"})).json()["id"])


async def _workflow(client: AsyncClient, operation: str, schema: dict[str, Any]) -> str:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral workflow",
            "operation": operation,
            "engine": "mock",
            "api_graph": {"1": {"class_type": "Mock", "inputs": {"length": "${frames}"}}}
            if "frames" in schema.get("properties", {})
            else {"1": {"class_type": "Mock", "inputs": {}}},
            "input_schema": schema,
        },
    )
    assert created.status_code == 201, created.text
    revision_id = cast(str, created.json()["current_revision_id"])
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.trusted = True
        session.commit()
    return revision_id


async def _replays_exactly(client: AsyncClient, content: bytes) -> dict[str, Any]:
    response = await _replay(client, await _new_chat(client), content)
    assert response.status_code == 202, response.text
    again = await _finished(client, response.json()["run"]["id"])
    first = open_output_recipe(content)
    second = open_output_recipe(await _record_of(client, again))
    for section in EXECUTION:
        assert second[section] == first[section], section
    return again


async def _replay(client: AsyncClient, chat_id: str, content: bytes) -> Any:
    return await client.post(
        f"/api/chats/{chat_id}/replays",
        content=content,
        headers={"content-type": "application/octet-stream"},
    )


def _counts() -> dict[str, int]:
    with SessionLocal() as session:
        return {
            model.__name__: session.scalar(select(func.count()).select_from(model)) or 0
            for model in (Message, Run, Job, RunContextSnapshot, WorkPlan)
        }


def _resealed(content: bytes, change: Any) -> bytes:
    payload = copy.deepcopy(open_output_recipe(content))
    del payload["digest"]
    change(payload)
    return seal_output_recipe(payload)


async def test_a_record_generates_again_exactly_in_a_new_chat(client: AsyncClient) -> None:
    profile_id = _profile()
    original, content = await _recorded(client, profile_id)
    chat_id = await _new_chat(client)

    response = await _replay(client, chat_id, content)

    assert response.status_code == 202, response.text
    again = await _finished(client, response.json()["run"]["id"])
    assert again["chat_id"] == chat_id
    assert again["workflow_revision_id"] == original["workflow_revision_id"]
    assert again["profile_id"] == profile_id
    with SessionLocal() as session:
        stored = session.get(Run, again["id"])
        assert stored is not None
        # Frozen, so what runs is fixed at acceptance.
        assert accepted_context(session, stored) is not None
    first = open_output_recipe(content)
    second = open_output_recipe(await _record_of(client, again))
    for section in EXECUTION:
        assert second[section] == first[section], section
    assert {**second["workflow"], "graph_source": None} == {
        **first["workflow"],
        "graph_source": None,
    }
    # The new record no longer lacks a frozen copy of its inputs.
    assert "frozen_snapshot_absent" not in second["reproducibility"]["missing"]


async def test_presets_and_defaults_are_neither_applied_nor_changed(
    client: AsyncClient,
) -> None:
    profile_id = _profile()
    _, content = await _recorded(client, profile_id)
    with SessionLocal() as session:
        session.add(
            GenerationPreset(
                name="A different default",
                role="image",
                settings_json={"steps": 30},
                is_default=True,
            )
        )
        session.commit()
    chat_id = await _new_chat(client)

    response = await _replay(client, chat_id, content)

    assert response.status_code == 202, response.text
    again = await _finished(client, response.json()["run"]["id"])
    second = open_output_recipe(await _record_of(client, again))
    assert second["settings"] == open_output_recipe(content)["settings"]
    # No preset layer reached the run at all, the default included.
    assert again["provenance_json"]["preset_layers"] == []
    assert again["provenance_json"]["preset"] is None
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        assert chat.routing_mode == "auto"
        assert chat.generation_settings_json == {}
        preset = session.scalar(select(GenerationPreset).where(GenerationPreset.is_default))
        assert preset is not None and preset.settings_json == {"steps": 30}


async def test_a_turn_admission_would_change_is_refused_and_nothing_is_written(
    client: AsyncClient,
) -> None:
    profile_id = _profile()
    _, content = await _recorded(client, profile_id)
    chat_id = await _new_chat(client)

    def without_cfg(record: dict[str, Any]) -> None:
        # A record that leaves a setting out, which admission fills with its default.
        del record["settings"]["unbound"]["cfg"]

    before = _counts()
    response = await _replay(client, chat_id, _resealed(content, without_cfg))

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "replay-differs"
    assert response.json()["sections"] == ["settings"]
    assert _counts() == before
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        # The refused turn's renaming of the chat was discarded with it.
        assert chat is not None and chat.title == "New chat"
    assert "ceramic" not in response.text


async def test_a_record_that_cannot_be_matched_or_a_used_chat_is_refused(
    client: AsyncClient,
) -> None:
    profile_id = _profile()
    original, content = await _recorded(client, profile_id)

    def without_prompt(record: dict[str, Any]) -> None:
        record["prompt"] = {
            "included": False,
            "positive": None,
            "negative": None,
            "omitted_reason": "chosen",
        }
        record["reproducibility"]["missing"] = sorted(
            [*record["reproducibility"]["missing"], "prompt_omitted"]
        )

    unavailable = await _replay(client, await _new_chat(client), _resealed(content, without_prompt))
    used = await _replay(client, original["chat_id"], content)
    nowhere = await _replay(client, "chat_" + "0" * 32, content)

    assert unavailable.status_code == 409
    assert unavailable.json()["code"] == "replay-unavailable"
    assert unavailable.json()["refusals"][0]["reasons"] == ["prompt_omitted"]
    assert used.status_code == 409
    assert used.json()["code"] == "replay-chat-not-clean"
    assert nowhere.status_code == 404
    assert nowhere.json()["code"] == "chat-not-found"


async def test_a_workflow_without_a_negative_prompt_replays_without_one(
    client: AsyncClient,
) -> None:
    revision_id = await _workflow(
        client,
        "text_to_image",
        {"type": "object", "properties": {"negative_prompt": {"readOnly": True}}},
    )
    _, content = await _recorded(
        client, _profile(), revision_id=revision_id, settings={"seed": 1234}
    )

    await _replays_exactly(client, content)


async def test_a_cleared_default_negative_prompt_stays_cleared(client: AsyncClient) -> None:
    schema = {
        "type": "object",
        "properties": {"negative_prompt": {"type": "string", "default": "neutral words"}},
    }
    revision_id = await _workflow(client, "text_to_video", schema)
    _, content = await _recorded(
        client,
        _profile("video"),
        mode="video",
        revision_id=revision_id,
        settings={"seed": 1234, "negative_prompt": ""},
    )
    assert open_output_recipe(content)["prompt"]["negative"] is None

    await _replays_exactly(client, content)


async def test_settings_admission_works_out_are_worked_out_again(client: AsyncClient) -> None:
    """A video's frames follow from its length; they are not sent, and still match."""

    revision_id = await _workflow(
        client,
        "text_to_video",
        {
            "type": "object",
            "properties": {
                "frames": {"type": "integer", "default": 49, "minimum": 17, "maximum": 81},
                "fps": {"type": "number", "const": 16},
            },
            "x-lm-atelier-video-length": {
                "version": 1,
                "frames_parameter": "frames",
                "fps_parameter": "fps",
                "fps_numerator": 16,
                "fps_denominator": 1,
                "frame_alignment": 16,
                "frame_offset": 1,
            },
        },
    )
    _, content = await _recorded(
        client,
        _profile("video"),
        mode="video",
        revision_id=revision_id,
        settings={"seed": 1234, "duration_seconds": 3},
    )
    assert open_output_recipe(content)["settings"]["bound"]["frames"] == 49

    again = await _replays_exactly(client, content)

    assert again["settings_json"]["frames"] == 49


async def test_a_record_that_would_make_several_runs_is_refused(client: AsyncClient) -> None:
    profile_id = _profile()
    _, content = await _recorded(client, profile_id)

    def three(record: dict[str, Any]) -> None:
        # Each run keeps a batch of one, so only a record from elsewhere says three.
        record["settings"]["unbound"]["batch_size"] = 3

    before = _counts()
    response = await _replay(client, await _new_chat(client), _resealed(content, three))

    assert response.status_code == 409, response.text
    assert response.json()["sections"] == ["output_count"]
    assert _counts() == before


async def test_a_video_from_a_picture_generates_again_from_the_same_picture(
    client: AsyncClient,
) -> None:
    picture = io.BytesIO()
    Image.new("RGB", (8, 6), (40, 90, 160)).save(picture, format="PNG")
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("still.png", picture.getvalue(), "image/png")}
    )
    assert uploaded.status_code == 201, uploaded.text
    source = uploaded.json()["id"]
    profile_id = _profile("video")
    chat = (await client.post("/api/chats", json={"title": "Recorded"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": PROMPT,
            "mode": "video",
            "profile_id": profile_id,
            "input_artifact_ids": [source],
            "settings": {"seed": 1234},
        },
    )
    assert turn.status_code == 202, turn.text
    original = await _finished(client, turn.json()["run"]["id"])
    content = await _record_of(client, original)
    record = open_output_recipe(content)
    assert record["operation"] == "image_to_video"
    assert [item["role"] for item in record["inputs"]] == ["source"]

    again = await _replays_exactly(client, content)

    assert again["provenance_json"]["input_artifact_ids"] == [source]


async def test_a_workspace_recipe_is_kept_out_of_the_replay(client: AsyncClient) -> None:
    profile_id = _profile()
    _, content = await _recorded(client, profile_id)
    with SessionLocal() as session:
        session.add(
            WorkflowUseCasePreset(
                name="A neutral recipe",
                use_case="image_generation",
                is_default=True,
                settings_json={"cfg": 3},
            )
        )
        session.commit()

    again = await _replays_exactly(client, content)

    # The recipe would sit under the turn's settings and leave its receipt on the run.
    assert not again["provenance_json"].get("workflow_use_case_preset")


async def _picture(client: AsyncClient) -> str:
    picture = io.BytesIO()
    Image.new("RGB", (8, 6), (40, 90, 160)).save(picture, format="PNG")
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("still.png", picture.getvalue(), "image/png")}
    )
    assert uploaded.status_code == 201, uploaded.text
    return cast(str, uploaded.json()["id"])


async def _edit_chat(client: AsyncClient) -> str:
    created = await client.post(
        "/api/chats",
        json={"title": "New chat", "vision_settings_json": {"verify_image_edits": False}},
    )
    assert created.status_code in {200, 201}, created.text
    return cast(str, created.json()["id"])


async def test_an_edit_of_a_picture_generates_again_from_its_request(
    client: AsyncClient,
) -> None:
    source = await _picture(client)
    profile_id = _profile()
    turn = await client.post(
        f"/api/chats/{await _edit_chat(client)}/turns",
        json={
            "text": "make the cup blue",
            "mode": "image",
            "profile_id": profile_id,
            "input_artifact_ids": [source],
            "settings": {"seed": 1234},
        },
    )
    assert turn.status_code == 202, turn.text
    original = await _finished(client, turn.json()["run"]["id"])
    content = await _record_of(client, original)
    record = open_output_recipe(content)
    assert record["operation"] == "image_to_image"
    assert record["prompt"]["positive"].endswith("make the cup blue")

    chat_id = await _new_chat(client)

    def without_cfg(payload: dict[str, Any]) -> None:
        del payload["settings"]["unbound"]["cfg"]

    differing = await _replay(client, chat_id, _resealed(content, without_cfg))
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        # Refused after the check was turned off for it: the change went with the turn.
        assert chat.vision_settings_json["verify_image_edits"] is True
    response = await _replay(client, chat_id, content)

    assert differing.status_code == 409
    assert differing.json()["code"] == "replay-differs"
    assert response.status_code == 202, response.text
    again = await _finished(client, response.json()["run"]["id"])
    assert again["standalone_prompt"] == "make the cup blue"
    second = open_output_recipe(await _record_of(client, again))
    for section in EXECUTION:
        assert second[section] == record[section], section
    with SessionLocal() as session:
        run = session.get(Run, again["id"])
        chat = session.get(Chat, chat_id)
        assert run is not None and chat is not None
        context = accepted_context(session, run)
        assert context is not None
        # The replayed edit ran without the after-the-fact check, which would add
        # a second, different result; the chat itself still checks later edits.
        assert context.vision_settings["verify_image_edits"] is False
        assert chat.vision_settings_json["verify_image_edits"] is True


async def test_an_edit_worded_by_another_version_is_not_replayed(client: AsyncClient) -> None:
    profile_id = _profile()
    _, content = await _recorded(client, profile_id)

    def reworded(record: dict[str, Any]) -> None:
        record["operation"] = "image_to_image"
        record["inputs"] = [
            {"sha256": "5" * 64, "role": "source", "size_bytes": 10, "media_type": "image/png"}
        ]

    def nothing_after(record: dict[str, Any]) -> None:
        reworded(record)
        record["prompt"]["positive"] = edit_prompt_preamble()

    chat_id = await _new_chat(client)
    response = await _replay(client, chat_id, _resealed(content, reworded))
    nothing_asked = await _replay(client, chat_id, _resealed(content, nothing_after))

    assert response.status_code == 409
    assert nothing_asked.status_code == 409
    assert nothing_asked.json()["refusals"][0]["reasons"] == ["edit_prompt_wording"]
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None and chat.vision_settings_json["verify_image_edits"] is True
    unsupported = [
        item for item in response.json()["refusals"] if item["code"] == "replay-record-unsupported"
    ]
    assert unsupported[0]["reasons"] == ["edit_prompt_wording"]
