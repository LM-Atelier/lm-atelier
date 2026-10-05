"""A comparison can make two videos from the same words, each held to its own workflow's length."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient
from test_generation_experiment_preflight import PREFLIGHT, _arm, _profile, _request, _revision
from test_generation_experiment_records import CREATE
from test_generation_experiment_start import _accepted, _start

from local_lm.db import SessionLocal
from local_lm.models import Chat, GenerationExperimentArm, Run, WorkPlan, WorkStep

#: Eight frames a second, lengths of 9 to 49 frames in steps of 8 after the first.
LENGTH: dict[str, Any] = {
    "version": 1,
    "frames_parameter": "frames",
    "fps_parameter": "fps",
    "fps_numerator": 8,
    "fps_denominator": 1,
    "frame_alignment": 8,
    "frame_offset": 1,
}
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frames": {"type": "integer", "minimum": 9, "maximum": 49, "default": 25},
        "fps": {"type": "integer", "default": 8},
    },
    "x-lm-atelier-video-length": LENGTH,
}


def _video_choice(label: str, **settings: Any) -> dict[str, Any]:
    return _arm(
        label,
        _profile(f"{label} model", role="video"),
        _revision(label, operation="text_to_video", input_schema_json=SCHEMA),
        **settings,
    )


def _videos(*arms: dict[str, Any], **values: Any) -> dict[str, Any]:
    # The test engine's video settings take no negative prompt.
    return _request(*arms, operation="text_to_video", negative_prompt="", **values)


async def test_two_videos_are_resolved_each_to_its_own_length(client: AsyncClient) -> None:
    short = _video_choice("Short take", duration_seconds=2)
    longer = _video_choice("Longer take", duration_seconds=5)

    response = await client.post(PREFLIGHT, json=_videos(short, longer))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == "compatible", body["refusals"]
    # Seconds become frames as a turn makes them: the nearest length the workflow can make.
    assert [
        (arm["effective_settings"]["frames"], arm["effective_settings"]["fps"])
        for arm in body["arms"]
    ] == [(17, 8), (41, 8)]
    assert [(arm["width"], arm["height"]) for arm in body["arms"]] == [(512, 384), (512, 384)]


async def test_a_choice_that_cannot_make_a_video_refuses_the_comparison(
    client: AsyncClient,
) -> None:
    video = _video_choice("Moving")
    still_model = _arm(
        "Still model",
        _profile("Still model"),
        _revision("Still model video", operation="text_to_video", input_schema_json=SCHEMA),
    )
    picture_workflow = _arm(
        "Picture workflow", _profile("Picture workflow model", role="video"), _revision("Stills")
    )
    too_long = _video_choice("Too long", duration_seconds=60)

    for arm, code, setting in (
        (still_model, "arm-profile-unavailable", None),
        (picture_workflow, "arm-operation-mismatch", None),
        (too_long, "arm-setting-invalid", "duration_seconds"),
    ):
        body = (await client.post(PREFLIGHT, json=_videos(video, arm))).json()
        assert body["outcome"] == "refused", arm["label"]
        assert [
            (item["code"], item["arm_ordinal"], item["setting"]) for item in body["refusals"]
        ] == [(code, 2, setting)], arm["label"]


async def test_videos_are_compared_with_their_choices_named(client: AsyncClient) -> None:
    first, second = _video_choice("First take"), _video_choice("Second take")

    named = await client.post(PREFLIGHT, json=_videos(first, second))
    blind = await client.post(PREFLIGHT, json=_videos(first, second, evaluation_mode="blind"))

    assert named.status_code == 200 and named.json()["outcome"] == "compatible", named.text
    # Refused as a malformed request, before anything is resolved.
    assert (blind.status_code, blind.json()["code"]) == (422, "request-validation-invalid")


async def test_a_video_comparison_queues_one_video_per_choice_at_its_length(
    app: FastAPI, client: AsyncClient
) -> None:
    accepted = await _accepted(
        client,
        _videos(
            _video_choice("Short take", duration_seconds=2),
            _video_choice("Longer take", duration_seconds=5),
        ),
    )
    lengths: list[dict[str, Any]] = []
    with SessionLocal() as session:
        for item in accepted["arms"]:
            arm = session.get(GenerationExperimentArm, item["id"])
            assert arm is not None
            lengths.append(arm.snapshot_json["video_length"])
    assert [(length["frames"], length["delivered_seconds"]) for length in lengths] == [
        (17, 2.125),
        (41, 5.125),
    ]

    # Held here, so what was queued stays as written.
    async with app.state.services.scheduler.lease("primary"):
        response = await _start(client, accepted)
        assert response.status_code == 202, response.text
        started = response.json()
    run_ids = [trial["run_id"] for arm in started["arms"] for trial in arm["trials"]]

    with SessionLocal() as session:
        plan = session.get(WorkPlan, started["work_plan_id"])
        assert plan is not None and plan.summary_json["routing_mode"] == "video"
        chat = session.get(Chat, plan.chat_id)
        assert chat is not None and chat.routing_mode == "video"
        for run_id, length in zip(run_ids, lengths, strict=True):
            run = session.get(Run, run_id)
            assert run is not None and run.operation == "text_to_video"
            # Each run keeps the length it was accepted with, as a video turn does.
            assert run.provenance_json["video_length"] == length
            assert run.settings_json["frames"] == length["frames"]
            step = session.get(WorkStep, run.work_step_id)
            assert step is not None and step.output_contract_json[0]["type"] == "video"


async def test_a_video_choice_is_drafted_as_a_recipe_for_making_videos(
    client: AsyncClient,
) -> None:
    accepted = await _accepted(
        client, _videos(_video_choice("First take"), _video_choice("Second take", steps=12))
    )

    response = await client.get(f"{CREATE}/{accepted['id']}/arms/2/recipe-draft")

    assert response.status_code == 200, response.text
    draft = response.json()
    assert draft["use_case"] == "video_generation"
    assert draft["settings_json"]["steps"] == 12
