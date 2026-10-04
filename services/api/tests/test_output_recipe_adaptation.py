"""A record generated as a new version, with chosen stand-ins for what does not match here."""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

from httpx2 import AsyncClient, Response
from PIL import Image
from test_generation_experiment_preflight import _revision
from test_output_recipe_replay import (
    _counts,
    _finished,
    _new_chat,
    _profile,
    _record_of,
    _recorded,
    _resealed,
    _workflow,
)

from local_lm.db import SessionLocal
from local_lm.models import ModelInstall, ModelProfile
from local_lm.output_recipe_v1 import open_output_recipe
from local_lm.workflow_graph_settings_v1 import GRAPH_SETTINGS_SCHEMA_KEY

LORA = {"sha256": "3" * 64, "position": 0, "model_strength": 1.0, "clip_strength": 1.0}


async def _adapt(
    client: AsyncClient, chat_id: str, content: bytes, choices: Sequence[tuple[str, str]]
) -> Response:
    return await client.post(
        f"/api/chats/{chat_id}/adaptations",
        params=tuple(choices),
        content=content,
        headers={"content-type": "application/octet-stream"},
    )


def _model_elsewhere(record: dict[str, Any]) -> None:
    record["model"]["files"] = {"elsewhere.safetensors": "2" * 64}


def _a_lora_elsewhere(record: dict[str, Any]) -> None:
    record["loras"] = [{**LORA, "enabled": True}]


def _codes(response: Response) -> list[str]:
    return [item["code"] for item in response.json()["refusals"]]


async def test_a_model_missing_here_is_chosen_and_the_run_says_what_differs(
    client: AsyncClient,
) -> None:
    profile_id = _profile()
    _, content = await _recorded(client, profile_id)
    elsewhere = _resealed(content, _model_elsewhere)

    unchosen = await _adapt(client, await _new_chat(client), elsewhere, [])
    response = await _adapt(
        client, await _new_chat(client), elsewhere, [("profile_id", profile_id)]
    )

    assert unchosen.status_code == 409
    assert _codes(unchosen) == ["replay-model-missing"]
    assert response.status_code == 202, response.text
    adapted = await _finished(client, response.json()["run"]["id"])
    record = open_output_recipe(elsewhere)
    assert adapted["profile_id"] == profile_id
    assert adapted["provenance_json"]["adaptation"] == {
        "record_digest": record["digest"],
        "choices": [{"requirement": "model", "position": None, "chosen": profile_id}],
        "left_out_settings": [],
        "differs": ["model"],
    }
    # What the record holds was sent as it is.
    again = open_output_recipe(await _record_of(client, adapted))
    for section in ("operation", "prompt", "seed", "settings", "inputs"):
        assert again[section] == record[section], section
    # Asked how it turned out, it says it is a new version and what differs.
    outcome = await client.get(f"/api/runs/{adapted['id']}/replay-result")
    assert outcome.status_code == 200, outcome.text
    assert outcome.json() == {
        "state": "adapted",
        "record_digest": record["digest"],
        "differs": ["model"],
    }


async def test_a_chosen_workflow_is_sent_only_the_settings_it_takes(client: AsyncClient) -> None:
    profile_id = _profile()
    original, content = await _recorded(client, profile_id)
    fixed_steps = await _workflow(
        client, "text_to_image", {"type": "object", "properties": {"steps": {"readOnly": True}}}
    )

    response = await _adapt(
        client, await _new_chat(client), content, [("workflow_revision_id", fixed_steps)]
    )

    assert response.status_code == 202, response.text
    adapted = await _finished(client, response.json()["run"]["id"])
    assert adapted["workflow_revision_id"] == fixed_steps != original["workflow_revision_id"]
    adaptation = adapted["provenance_json"]["adaptation"]
    assert adaptation["choices"] == [
        {"requirement": "workflow", "position": None, "chosen": fixed_steps}
    ]
    assert adaptation["left_out_settings"] == ["steps"]
    assert {"workflow", "settings"} <= set(adaptation["differs"])
    assert "prompt" not in adaptation["differs"] and "seed" not in adaptation["differs"]


async def test_a_lora_missing_here_can_be_left_out(client: AsyncClient) -> None:
    _, content = await _recorded(client, _profile())
    elsewhere = _resealed(content, _a_lora_elsewhere)

    unchosen = await _adapt(client, await _new_chat(client), elsewhere, [])
    response = await _adapt(client, await _new_chat(client), elsewhere, [("lora", "0:omit")])

    assert unchosen.status_code == 409
    assert _codes(unchosen) == ["replay-lora-missing"]
    assert response.status_code == 202, response.text
    adapted = await _finished(client, response.json()["run"]["id"])
    adaptation = adapted["provenance_json"]["adaptation"]
    assert adaptation["choices"] == [{"requirement": "lora", "position": 0, "chosen": None}]
    assert adaptation["differs"] == ["loras"]


async def test_choices_a_record_cannot_take_are_refused_and_nothing_is_written(
    client: AsyncClient,
) -> None:
    profile_id = _profile()
    _, content = await _recorded(client, profile_id)
    with_a_lora = _resealed(content, _a_lora_elsewhere)
    video_workflow = await _workflow(client, "text_to_video", {"type": "object", "properties": {}})
    video_profile = _profile("video")
    switched_off = _profile()
    with SessionLocal() as session:
        profile = session.get(ModelProfile, switched_off)
        assert profile is not None
        install = session.get(ModelInstall, profile.model_install_id)
        assert install is not None
        install.active = False
        session.commit()
    chat_id = await _new_chat(client)
    before = _counts()

    malformed = {
        "a position it does not have": (content, [("lora", "0:omit")]),
        "a position written oddly": (with_a_lora, [("lora", "00:omit")]),
        "a position twice": (with_a_lora, [("lora", "0:omit"), ("lora", "0:omit")]),
        "two workflows": (content, [("workflow_revision_id", "a"), ("workflow_revision_id", "b")]),
        "not an identifier": (content, [("profile_id", "a profile")]),
    }
    for name, (record, choices) in malformed.items():
        response = await _adapt(client, chat_id, record, choices)
        assert response.status_code == 422, name
        assert response.json()["code"] == "adaptation-choice-invalid", name

    unusable = {
        "adaptation-workflow-unusable": (content, [("workflow_revision_id", video_workflow)]),
        "adaptation-model-unusable": (content, [("profile_id", video_profile)]),
        "a model whose files are switched off": (content, [("profile_id", switched_off)]),
        "adaptation-lora-unusable": (with_a_lora, [("lora", "0:asset_missing")]),
    }
    for name, (record, choices) in unusable.items():
        code = name if name.startswith("adaptation-") else "adaptation-model-unusable"
        response = await _adapt(client, chat_id, record, choices)
        assert response.status_code == 409, name
        assert response.json()["code"] == "adaptation-unavailable", name
        assert code in _codes(response), name

    def three(record: dict[str, Any]) -> None:
        # Each run keeps a batch of one, so only a record from elsewhere says three.
        record["settings"]["unbound"]["batch_size"] = 3

    several = await _adapt(client, chat_id, _resealed(content, three), [("profile_id", profile_id)])
    assert several.status_code == 409, several.text
    assert several.json()["code"] == "adaptation-output-count"

    def prompt_left_out(record: dict[str, Any]) -> None:
        record["prompt"] = {
            "included": False,
            "positive": None,
            "negative": None,
            "omitted_reason": "chosen",
        }
        record["reproducibility"]["missing"] = sorted(
            [*record["reproducibility"]["missing"], "prompt_omitted"]
        )

    without_words = await _adapt(
        client, chat_id, _resealed(content, prompt_left_out), [("profile_id", profile_id)]
    )
    assert without_words.status_code == 409
    assert _codes(without_words) == ["replay-record-incomplete"]
    assert _counts() == before

    # A chat that already holds something is not one a record is generated into.
    used = await _adapt(client, chat_id, content, [])
    assert used.status_code == 202, used.text
    again = await _adapt(client, chat_id, content, [])
    assert again.status_code == 409
    assert again.json()["code"] == "replay-chat-not-clean"


async def test_a_workflow_that_takes_fewer_settings_is_sent_only_those(
    client: AsyncClient,
) -> None:
    _, content = await _recorded(client, _profile())
    steps_only = _revision(
        "Steps only",
        api_graph_json={"1": {"class_type": "KSampler", "inputs": {"steps": "${steps}"}}},
        input_schema_json={
            "type": "object",
            "properties": {"steps": {"type": "integer", "default": 20, "minimum": 1}},
            GRAPH_SETTINGS_SCHEMA_KEY: {
                "version": 1,
                "bindings": [{"parameter": "steps", "node_id": "1", "input_name": "steps"}],
            },
        },
    )
    recorded = open_output_recipe(content)["settings"]["unbound"]
    assert {"steps", "cfg"} <= set(recorded)

    response = await _adapt(
        client, await _new_chat(client), content, [("workflow_revision_id", steps_only)]
    )

    assert response.status_code == 202, response.text
    adapted = await _finished(client, response.json()["run"]["id"])
    left_out = adapted["provenance_json"]["adaptation"]["left_out_settings"]
    # The workflow does not take a seed or a guidance scale, so neither was sent.
    assert {"cfg", "seed"} <= set(left_out)
    assert "steps" not in left_out
    assert adapted["settings_json"]["steps"] == recorded["steps"]


async def test_what_admission_works_out_again_is_not_called_left_out(
    client: AsyncClient,
) -> None:
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
    profile_id = _profile("video")
    _, content = await _recorded(
        client,
        profile_id,
        mode="video",
        revision_id=revision_id,
        settings={"seed": 1234, "duration_seconds": 3},
    )

    response = await _adapt(client, await _new_chat(client), content, [("profile_id", profile_id)])

    assert response.status_code == 202, response.text
    adapted = await _finished(client, response.json()["run"]["id"])
    # Its frames follow from its length; they came out as recorded.
    assert adapted["settings_json"]["frames"] == 49
    assert adapted["provenance_json"]["adaptation"]["left_out_settings"] == []


async def test_a_chosen_workflow_or_model_stands_in_for_one_the_record_lacks(
    client: AsyncClient,
) -> None:
    profile_id = _profile()
    original, content = await _recorded(client, profile_id)

    def no_model(record: dict[str, Any]) -> None:
        record["model"] = None

    def unverified(record: dict[str, Any]) -> None:
        record["workflow"]["verified"] = False

    without_model = await _adapt(
        client, await _new_chat(client), _resealed(content, no_model), [("profile_id", profile_id)]
    )
    without_workflow = await _adapt(
        client,
        await _new_chat(client),
        _resealed(content, unverified),
        [("workflow_revision_id", original["workflow_revision_id"])],
    )
    exact = await _adapt(client, await _new_chat(client), _resealed(content, unverified), [])

    assert without_model.status_code == 202, without_model.text
    assert without_workflow.status_code == 202, without_workflow.text
    assert exact.status_code == 409
    assert _codes(exact) == ["replay-record-incomplete"]


async def _picture(client: AsyncClient, colour: tuple[int, int, int]) -> str:
    picture = io.BytesIO()
    Image.new("RGB", (8, 6), colour).save(picture, format="PNG")
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("still.png", picture.getvalue(), "image/png")}
    )
    assert uploaded.status_code == 201, uploaded.text
    return str(uploaded.json()["id"])


async def test_a_picture_here_can_stand_in_for_an_input_that_is_not(
    client: AsyncClient,
) -> None:
    source = await _picture(client, (40, 90, 160))
    chat = (await client.post("/api/chats", json={"title": "Recorded"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "a slow pan",
            "mode": "video",
            "profile_id": _profile("video"),
            "input_artifact_ids": [source],
            "settings": {"seed": 1234},
        },
    )
    assert turn.status_code == 202, turn.text
    content = await _record_of(client, await _finished(client, turn.json()["run"]["id"]))

    def picture_elsewhere(record: dict[str, Any]) -> None:
        record["inputs"][0]["sha256"] = "7" * 64

    elsewhere = _resealed(content, picture_elsewhere)
    stand_in = await _picture(client, (200, 120, 40))
    unchosen = await _adapt(client, await _new_chat(client), elsewhere, [])
    response = await _adapt(
        client, await _new_chat(client), elsewhere, [("input", f"0:{stand_in}")]
    )

    assert unchosen.status_code == 409
    assert _codes(unchosen) == ["replay-input-missing"]
    assert response.status_code == 202, response.text
    adapted = await _finished(client, response.json()["run"]["id"])
    assert adapted["provenance_json"]["input_artifact_ids"] == [stand_in]
    adaptation = adapted["provenance_json"]["adaptation"]
    assert adaptation["choices"] == [{"requirement": "input", "position": 0, "chosen": stand_in}]
    assert adaptation["differs"] == ["inputs"]

    chat_id = await _new_chat(client)
    before = _counts()
    for name, choices, status, code in (
        ("not a picture here", [("input", f"0:sha256:{'8' * 64}")], 409, "adaptation-unavailable"),
        (
            "a position it does not have",
            [("input", f"1:{stand_in}")],
            422,
            "adaptation-choice-invalid",
        ),
        ("not a content identity", [("input", "0:artifact_one")], 422, "adaptation-choice-invalid"),
    ):
        refused = await _adapt(client, chat_id, elsewhere, choices)
        assert refused.status_code == status, name
        assert refused.json()["code"] == code, name
    assert _counts() == before
