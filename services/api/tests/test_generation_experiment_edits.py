"""A comparison can change one picture: both choices start from it and keep its size."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_recovery_replay import _change
from test_generation_experiment_preflight import PREFLIGHT, _arm, _request
from test_generation_experiment_records import CREATE
from test_generation_experiment_start import _accepted, _start
from test_picture_remix_edit import _picture, _uploaded
from test_source_crop_acceptance import object_info
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_crop_route import graph

from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.models import (
    ArtifactLibraryEntry,
    Chat,
    Message,
    MessagePart,
    ModelInstall,
    ModelProfile,
    Run,
    WorkPlan,
    WorkStep,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "input_image": {"type": "string"},
        "checkpoint": {"type": "string", "default": "fixture.safetensors"},
        "denoise": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.6},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2_147_483_647, "default": 1},
    },
}


def _changes_the_picture() -> dict[str, Any]:
    """A workflow that encodes the picture it is given and samples from it."""

    workflow = graph()
    workflow["positive"]["inputs"]["text"] = "$" + "{prompt}"
    workflow["sample"]["inputs"]["seed"] = "$" + "{seed}"
    return workflow


def _sets_the_picture_aside() -> dict[str, Any]:
    """A workflow that takes a picture and samples from an empty canvas instead."""

    workflow = _changes_the_picture()
    workflow["empty"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 768, "batch_size": 1},
    }
    workflow["sample"]["inputs"]["latent_image"] = ["empty", 0]
    return workflow


async def _choice(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    workflow: dict[str, Any],
    *,
    operation: str = "image_to_image",
    name: str = "Edit fixture",
) -> tuple[str, str]:
    """A reviewed workflow and a model for it, on a runtime that describes its nodes."""

    described = {**object_info(_changes_the_picture())}
    described["EmptyLatentImage"] = {
        "python_module": "nodes",
        "input": {"required": {"width": ["*", {}], "height": ["*", {}], "batch_size": ["*", {}]}},
        "output": ["*"],
    }

    async def describe() -> dict[str, Any]:
        return described

    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    monkeypatch.setattr(app.state.services.engines.media, "object_info", describe, raising=False)
    # Work waits for a runtime that is never set up here, so what was queued stays as written.
    monkeypatch.setattr(app.state.services.processes, "start_media", AsyncMock())
    with SessionLocal() as session:
        profile = ModelProfile(name=name, role="image", engine="comfyui")
        session.add(profile)
        session.commit()
        profile_id = profile.id
    created = await client.post(
        "/api/workflows",
        json={
            "name": name,
            "operation": operation,
            "engine": "comfyui",
            "api_graph": workflow,
            "input_schema": SCHEMA,
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


def _edit(source: str, *arms: dict[str, Any], **values: Any) -> dict[str, Any]:
    body = _request(
        *arms,
        operation="image_to_image",
        source_artifact_id=source,
        geometry={"mode": "source"},
        negative_prompt="",
    )
    body.update(values)
    return body


async def _two_strengths(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    revision, profile = await _choice(app, client, monkeypatch, _changes_the_picture())
    source = await _uploaded(client, _picture((512, 768)))
    light = _arm("Light touch", profile, revision, denoise=0.3)
    heavier = _arm("Heavier touch", profile, revision, denoise=0.55)
    return source, light, heavier


async def test_an_edit_comparison_resolves_both_choices_against_the_one_picture(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, light, heavier = await _two_strengths(app, client, monkeypatch)

    response = await client.post(PREFLIGHT, json=_edit(source, light, heavier))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == "compatible", body["refusals"]
    # Both keep the picture's size, and each takes its own strength as a turn would.
    assert [(arm["width"], arm["height"]) for arm in body["arms"]] == [(512, 768), (512, 768)]
    assert [arm["effective_settings"]["denoise"] for arm in body["arms"]] == [0.3, 0.55]
    assert "width" not in body["arms"][0]["effective_settings"]


async def test_an_edit_comparison_queues_each_choice_from_that_picture(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, light, heavier = await _two_strengths(app, client, monkeypatch)
    accepted = await _accepted(client, _edit(source, light, heavier))
    assert accepted["operation"] == "image_to_image"
    assert accepted["source_artifact_id"] == source
    assert accepted["geometry"] == {"mode": "source"}

    async with app.state.services.scheduler.lease("primary"):
        response = await _start(client, accepted)
        assert response.status_code == 202, response.text
        started = response.json()
    run_ids = [trial["run_id"] for arm in started["arms"] for trial in arm["trials"]]

    with SessionLocal() as session:
        plan = session.get(WorkPlan, started["work_plan_id"])
        assert plan is not None
        chat = session.get(Chat, plan.chat_id)
        assert chat is not None and chat.vision_settings_json == {"verify_image_edits": False}
        runs = [session.get(Run, run_id) for run_id in run_ids]
        first = runs[0]
        assert first is not None
        user_message = session.get(Message, first.user_message_id)
        assert user_message is not None
        parts = session.scalars(
            select(MessagePart).where(MessagePart.message_id == user_message.id)
        ).all()
        # The picture is the one input every picture's run reads, as an edit turn carries it.
        assert [(part.type, part.artifact_id, part.metadata_json) for part in parts] == [
            ("image", source, {"input_reference": True, "input_reference_source": "explicit"})
        ]
        strengths = []
        for run in runs:
            assert run is not None and run.operation == "image_to_image"
            context = accepted_context(session, run)
            assert context is not None and list(context.input_artifact_ids) == [source]
            assert run.provenance_json["input_artifact_ids"] == [source]
            edit = run.provenance_json["image_edit"]
            assert edit["policy"] == "preserve_unrequested_details_v1"
            strengths.append(edit["strength"]["value"])
            step = session.get(WorkStep, run.work_step_id)
            assert step is not None
            assert step.input_bindings_json == [
                {"type": "explicit_artifact", "artifact_id": source}
            ]
        assert strengths == [0.3, 0.55]


async def _trashed(client: AsyncClient, artifact_id: str) -> None:
    """The picture moved to the trash, as the Media Library moves one there."""

    with SessionLocal() as session:
        entry = session.scalar(
            select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == artifact_id)
        )
        assert entry is not None
        entry_id = entry.id
    await _change(
        client,
        f"/api/artifact-library/{entry_id}/deletion-impact",
        f"/api/artifact-library/{entry_id}/trash",
        "trash-the-picture",
    )


@pytest.mark.parametrize(
    "picture", ["unknown", "trashed", "attached", "odd size", "turned", "moving"]
)
async def test_a_picture_that_cannot_be_changed_here_refuses_the_comparison(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, picture: str
) -> None:
    revision, profile = await _choice(app, client, monkeypatch, _changes_the_picture())
    if picture == "unknown":
        source = "sha256:" + "0" * 64
    elif picture == "odd size":
        source = await _uploaded(client, _picture((510, 768)))
    elif picture == "turned":
        source = await _uploaded(client, _picture((512, 768), turned=True))
    elif picture == "moving":
        source = await _uploaded(client, _picture((512, 768), frames=2))
    elif picture == "attached":
        # A file sent with a message is kept for it, and is no picture in the library.
        attached = await client.post(
            "/api/artifacts", files={"file": ("picture.png", _picture(), "image/png")}
        )
        assert attached.status_code == 201, attached.text
        source = attached.json()["id"]
    else:
        source = await _uploaded(client, _picture((512, 768)))
        await _trashed(client, source)
    body = _edit(
        source,
        _arm("Light touch", profile, revision, denoise=0.3),
        _arm("Heavier touch", profile, revision, denoise=0.55),
    )

    answer = (await client.post(PREFLIGHT, json=body)).json()

    assert answer["outcome"] == "refused" and answer["preflight_sha256"] is None
    assert {"code": "source-unavailable", "arm_ordinal": None} in [
        {"code": refusal["code"], "arm_ordinal": refusal["arm_ordinal"]}
        for refusal in answer["refusals"]
    ]


@pytest.mark.parametrize("kind", ["sets the picture aside", "makes pictures from words"])
async def test_a_choice_not_shown_to_change_the_picture_is_refused(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    revision, profile = await _choice(app, client, monkeypatch, _changes_the_picture())
    if kind == "sets the picture aside":
        other, other_profile = await _choice(
            app, client, monkeypatch, _sets_the_picture_aside(), name="Aside fixture"
        )
    else:
        other, other_profile = await _choice(
            app,
            client,
            monkeypatch,
            _changes_the_picture(),
            operation="text_to_image",
            name="Words fixture",
        )
    source = await _uploaded(client, _picture((512, 768)))
    body = _edit(
        source,
        _arm("Changes it", profile, revision, denoise=0.3),
        _arm("Does not", other_profile, other, denoise=0.3),
    )

    answer = (await client.post(PREFLIGHT, json=body)).json()

    assert answer["outcome"] == "refused"
    assert [(refusal["code"], refusal["arm_ordinal"]) for refusal in answer["refusals"]] == [
        ("arm-not-an-edit", 2)
    ]


@pytest.mark.parametrize(
    "change",
    [
        "a change with no picture",
        "words with a picture",
        "a change at a chosen size",
        "words at the picture's size",
    ],
)
async def test_a_request_mixing_words_and_a_picture_is_refused(
    client: AsyncClient, change: str
) -> None:
    first = _arm("One", "profile_one", "revision_one")
    second = _arm("Two", "profile_two", "revision_two")
    body = _edit("sha256:" + "1" * 64, first, second)
    if change == "a change with no picture":
        del body["source_artifact_id"]
    elif change == "words with a picture":
        body.update(
            operation="text_to_image", geometry={"mode": "size", "width": 512, "height": 512}
        )
    elif change == "a change at a chosen size":
        body["geometry"] = {"mode": "size", "width": 512, "height": 768}
    else:
        body.update(operation="text_to_image")
        del body["source_artifact_id"]

    response = await client.post(PREFLIGHT, json=body)

    assert response.status_code == 422, response.text


async def test_a_start_refuses_once_the_picture_cannot_be_changed(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, light, heavier = await _two_strengths(app, client, monkeypatch)
    accepted = await _accepted(client, _edit(source, light, heavier))
    await _trashed(client, source)

    response = await _start(client, accepted)

    assert response.status_code == 409, response.text
    with SessionLocal() as session:
        assert (
            session.scalar(
                select(WorkPlan).where(WorkPlan.source_action == "generation_experiment")
            )
            is None
        )
    reread = (await client.get(f"{CREATE}/{accepted['id']}")).json()
    assert reread["state"] == "ready"


async def test_a_change_is_not_drafted_as_a_recipe_for_making_pictures(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, light, heavier = await _two_strengths(app, client, monkeypatch)
    accepted = await _accepted(client, _edit(source, light, heavier))

    response = await client.get(f"{CREATE}/{accepted['id']}/arms/2/recipe-draft")

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "generation-experiment-recipe-not-for-changes"


async def test_a_choice_that_changed_the_picture_is_kept_as_a_studio_recipe(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, light, heavier = await _two_strengths(app, client, monkeypatch)
    revision, profile = heavier["workflow_revision_id"], heavier["profile_id"]
    # A run names its model through the model's install, in a chat as here.
    with SessionLocal() as session:
        install = ModelInstall(
            name="Edit fixture base",
            role="image",
            engine="comfyui",
            local_path="C:/managed/edit-fixture",
            manifest_json={},
            active=True,
        )
        session.add(install)
        session.flush()
        stored = session.get(ModelProfile, profile)
        assert stored is not None
        stored.model_install_id = install.id
        session.commit()
    accepted = await _accepted(client, _edit(source, light, heavier, prompt="Warmer evening light"))
    async with app.state.services.scheduler.lease("primary"):
        response = await _start(client, accepted)
        assert response.status_code == 202, response.text
    second = response.json()["arms"][1]["trials"][0]["run_id"]

    # The words are the comparison's own, asked of every choice, though no message holds them.
    draft = await client.get(f"/api/runs/{second}/edit-recipe-draft")
    assert draft.status_code == 200, draft.text
    assert draft.json() == {"run_id": second, "instruction": "Warmer evening light"}

    saved = await client.post(
        "/api/edit-templates",
        json={
            "name": "Heavier touch",
            "instruction": "Warmer evening light",
            "from_run_id": second,
        },
    )

    assert saved.status_code == 201, saved.text
    template = saved.json()
    assert (template["workflow_revision_id"], template["model_profile_id"]) == (revision, profile)
    assert template["mask_mode"] == "none"
    # That choice's own strength, without the comparison's seed or its one-picture count.
    assert template["settings_json"]["denoise"] == 0.55
    assert not {"seed", "batch_size"} & set(template["settings_json"])
