"""A picture edit scoped to a selection, generated again from its record."""

from __future__ import annotations

import io
from typing import Any, cast

from httpx2 import AsyncClient
from PIL import Image
from test_output_recipe_replay import (
    _edit_chat,
    _finished,
    _new_chat,
    _profile,
    _record_of,
    _replay,
    _replays_exactly,
    _resealed,
    _workflow,
)
from test_output_recipe_replay_plan import _plan

from local_lm.db import SessionLocal
from local_lm.models import MessagePart, Run
from local_lm.output_recipe_v1 import open_output_recipe

MASKED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"mask": {"type": "object", "x-lm-atelier-kind": "mask"}},
}


async def _picture(client: AsyncClient, colour: tuple[int, int, int]) -> str:
    picture = io.BytesIO()
    Image.new("RGB", (8, 6), colour).save(picture, format="PNG")
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("still.png", picture.getvalue(), "image/png")}
    )
    assert uploaded.status_code == 201, uploaded.text
    return cast(str, uploaded.json()["id"])


async def _masked_edit(
    client: AsyncClient, selection: dict[str, Any], *, also_a_picture: bool = False
) -> tuple[str, bytes]:
    source = await _picture(client, (40, 90, 160))
    revision_id = await _workflow(client, "image_to_image", MASKED_SCHEMA)
    turn = await client.post(
        f"/api/chats/{await _edit_chat(client)}/turns",
        json={
            "text": "make the cup blue",
            "mode": "image",
            "profile_id": _profile(),
            "workflow_revision_id": revision_id,
            "input_artifact_ids": [source],
            "settings": {"seed": 1234, "mask": selection},
        },
    )
    assert turn.status_code == 202, turn.text
    original = await _finished(client, turn.json()["run"]["id"])
    if also_a_picture:
        # As a turn that was given the selection as one of its pictures as well.
        with SessionLocal() as session:
            run = session.get(Run, original["id"])
            assert run is not None
            session.add(
                MessagePart(
                    message_id=run.user_message_id,
                    position=99,
                    type="image",
                    artifact_id=selection["artifact_id"],
                    metadata_json={"input_reference": True},
                )
            )
            session.commit()
    return source, await _record_of(client, original)


async def test_an_edit_of_a_selection_generates_again_with_that_selection(
    client: AsyncClient,
) -> None:
    mask = await _picture(client, (255, 255, 255))
    source, content = await _masked_edit(client, {"artifact_id": mask})
    record = open_output_recipe(content)
    assert [(item["sha256"], item["role"]) for item in record["inputs"]] == [
        (source.removeprefix("sha256:"), "source"),
        (mask.removeprefix("sha256:"), "mask"),
    ]
    assert "settings.mask" not in record["removed"]

    plan = await _plan(client, content)
    again = await _replays_exactly(client, content)

    assert plan["ready"] is True, plan["refusals"]
    # The selection goes as the turn's setting, never as one of its pictures.
    assert plan["resolved"]["input_artifact_ids"] == [source]
    assert plan["resolved"]["mask_artifact_id"] == mask
    with SessionLocal() as session:
        run = session.get(Run, again["id"])
        assert run is not None
        assert run.settings_json["mask"] == {"artifact_id": mask}


async def test_a_selection_the_record_does_not_hold_whole_is_refused(
    client: AsyncClient,
) -> None:
    mask = await _picture(client, (255, 255, 255))
    _, feathered = await _masked_edit(
        client, {"artifact_id": mask, "feather_px": 4, "invert": True}
    )
    _, plain = await _masked_edit(client, {"artifact_id": mask})

    def on_a_video(record: dict[str, Any]) -> None:
        record["operation"] = "image_to_video"

    def before_its_picture(record: dict[str, Any]) -> None:
        record["inputs"].reverse()

    chat_id = await _new_chat(client)
    refused = {
        "feathered": await _replay(client, chat_id, feathered),
        "on a video": await _replay(client, chat_id, _resealed(plain, on_a_video)),
        "before its picture": await _replay(client, chat_id, _resealed(plain, before_its_picture)),
    }

    for name, response in refused.items():
        assert response.status_code == 409, name
        unsupported = [
            item
            for item in response.json()["refusals"]
            if item["code"] == "replay-record-unsupported"
        ]
        assert unsupported and "mask_input" in unsupported[0]["reasons"], name
    # How it was feathered and turned round was left out, so the record says it is incomplete.
    incomplete = [
        item
        for item in refused["feathered"].json()["refusals"]
        if item["code"] == "replay-record-incomplete"
    ]
    assert incomplete and "settings_removed" in incomplete[0]["reasons"]


async def test_a_studio_selection_neither_feathered_nor_turned_round_generates_again(
    client: AsyncClient,
) -> None:
    mask = await _picture(client, (255, 255, 255))
    _, content = await _masked_edit(client, {"artifact_id": mask, "feather_px": 0, "invert": False})
    assert "settings.mask" not in open_output_recipe(content)["removed"]

    again = await _replays_exactly(client, content)

    with SessionLocal() as session:
        run = session.get(Run, again["id"])
        assert run is not None
        assert run.settings_json["mask"] == {"artifact_id": mask}


async def test_a_selection_that_was_also_a_picture_is_not_generated_again(
    client: AsyncClient,
) -> None:
    mask = await _picture(client, (255, 255, 255))
    source, content = await _masked_edit(client, {"artifact_id": mask}, also_a_picture=True)
    record = open_output_recipe(content)
    # Listed as the picture it was and as the selection, so nothing is hidden.
    assert [(item["sha256"], item["role"]) for item in record["inputs"]] == [
        (source.removeprefix("sha256:"), "source"),
        (mask.removeprefix("sha256:"), "input"),
        (mask.removeprefix("sha256:"), "mask"),
    ]

    plan = await _plan(client, content)

    unsupported = [item for item in plan["refusals"] if item["code"] == "replay-record-unsupported"]
    assert plan["ready"] is False
    assert unsupported and "repeated_inputs" in unsupported[0]["reasons"]


async def test_a_selection_does_not_count_as_one_of_the_pictures(client: AsyncClient) -> None:
    mask = await _picture(client, (255, 255, 255))
    _, content = await _masked_edit(client, {"artifact_id": mask})

    def sixteen_pictures(record: dict[str, Any]) -> None:
        pictures = [
            {
                "sha256": f"{index:02d}" * 32,
                "role": "source" if index == 0 else "input",
                "size_bytes": 10,
                "media_type": "image/png",
            }
            for index in range(16)
        ]
        record["inputs"] = [*pictures, record["inputs"][-1]]

    plan = await _plan(client, _resealed(content, sixteen_pictures))

    # The pictures are not here, but the record's shape is one a turn carries.
    assert "replay-record-unsupported" not in [item["code"] for item in plan["refusals"]]
    assert "replay-input-missing" in [item["code"] for item in plan["refusals"]]
