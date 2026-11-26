"""An accepted render names each value it changed on purpose, and why.

A turn, a regenerate, a branch, an edit or a Prompt Library batch can run with
a value other than one it was given or carried over: a saved value its
workflow does not take, a count it makes another way, a strength or a length
its workflow decides, or a seed it draws again or steps. The answer names each
in `setting_changes`, with a code from one closed vocabulary and that code's
one sentence, where a value the request did not send came from, and the value
given and the value used where each is a number, true or false, or one of its
setting's own choices. Nothing else is named back, and a render that changed
nothing answers without the list. Each run keeps what it was accepted with, so
a replayed request answers with the same list.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_generation_experiment_preflight import _profile, _revision
from test_generation_offer_api import create_offer
from test_plan_refusals_name_their_limit import (
    STORY_PICTURE_VIDEO,
    STORY_THEN_PICTURE,
    _batch,
    _finished,
    _queue,
)
from test_project_portability import _manifest, _rewrite_archive
from test_saved_sizes_a_workflow_cannot_make import _save
from test_upscale_preview_api import _selection
from test_workflow_recipe_request_cases import _picture, _workflow
from workflow_fixtures import seed_workflow_trust

from local_lm.db import SessionLocal
from local_lm.domain import new_id
from local_lm.models import Chat, GenerationPreset, Run, WorkStep
from local_lm.saved_settings import MAX_SETTING_FIELDS

TEXT = "Create an image of a ceramic cup"
VIDEO_TEXT = "A slow camera move across a gray table"
#: A picture workflow taking up to 30 steps, two samplers, one scheduler and no CFG.
PROBE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {"type": "integer", "default": 12, "minimum": 1, "maximum": 30},
        "cfg": {"type": "number", "readOnly": True},
        "sampler": {"type": "string", "default": "euler", "enum": ["euler", "heun"]},
        "scheduler": {"type": "string", "const": "karras"},
    },
}
#: A video workflow that makes 17 to 81 frames at 16 a second, in steps of four.
ALIGNED_LENGTH: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frames": {"type": "integer", "minimum": 17, "maximum": 81, "default": 49},
        "fps": {"type": "integer", "default": 16},
    },
    "x-lm-atelier-video-length": {
        "version": 1,
        "frames_parameter": "frames",
        "fps_parameter": "fps",
        "fps_numerator": 16,
        "fps_denominator": 1,
        "frame_alignment": 4,
        "frame_offset": 1,
    },
}
#: A picture workflow taking a CFG of up to ten.
CFG_UP_TO_TEN: dict[str, Any] = {
    "type": "object",
    "properties": {"cfg": {"type": "number", "default": 4, "minimum": 0, "maximum": 10}},
}
#: A picture workflow taking a guidance of its own beside a CFG, each up to ten.
GUIDANCE_AND_CFG: dict[str, Any] = {
    "type": "object",
    "properties": {
        "guidance": {"type": "number", "default": 3, "minimum": 0, "maximum": 10},
        "cfg": {"type": "number", "default": 4, "minimum": 0, "maximum": 10},
    },
}
#: A picture workflow's properties taking a seed of up to ten only.
SEED_UP_TO_TEN = {"seed": {"type": "integer", "default": 5, "minimum": 0, "maximum": 10}}
#: Where a value can be saved, and the source a change names it by.
SOURCES = {
    "turn preset": "turn_preset",
    "chat": "chat",
    "chat preset": "chat_preset",
    "project": "project",
    "project preset": "project_preset",
    "default preset": "default_preset",
    "model profile": "profile",
}
#: Text saved where no setting takes it, which no answer may repeat.
MARKER = "neutral-marker-7f3a"

STEPS_SET_ASIDE = {
    "code": "set-aside-not-usable",
    "field": "steps",
    "source": "chat",
    "before": 150,
    "after": 12,
}
CFG_NOT_TAKEN = {"code": "set-aside-not-taken", "field": "cfg", "source": "chat", "before": 5}
SAMPLER_SET_ASIDE = {
    "code": "set-aside-not-usable",
    "field": "sampler",
    "source": "chat",
    "after": "euler",
}
SCHEDULER_FIXED = {
    "code": "set-aside-fixed",
    "field": "scheduler",
    "source": "chat",
    "after": "karras",
}
FRAMES_NOT_OFFERED = {
    "code": "set-aside-not-offered",
    "field": "frames",
    "source": "chat",
    "before": 10,
}
WIDTH_FOLLOWS_SOURCE = {
    "code": "size-follows-source",
    "field": "width",
    "source": "chat",
    "before": 2176,
}
ENLARGEMENT_FIXED = {"code": "set-aside-fixed", "field": "upscale_factor", "before": 2}
FOR_OFFER = {"code": "set-aside-for-offer", "field": "max_tokens", "before": 64}
BATCH_SIZE_MADE_ONE = {
    "code": "outputs-from-batch-size",
    "field": "batch_size",
    "source": "chat",
    "before": 3,
    "after": 1,
}
COUNT_RAISED = {"code": "output-count-raised", "field": "output_count", "before": 2, "after": 3}
ONE_REPLY = {"code": "one-output", "field": "output_count", "before": 3, "after": 1}
STRENGTH_FOR_WORDS = {
    "code": "strength-default-for-new-picture",
    "field": "denoise",
    "before": 0.5,
    "after": 1,
}
LENGTH_ALIGNED = {
    "code": "length-aligned-to-frames",
    "field": "duration_seconds",
    "before": 2.0,
    "after": 2.0625,
}
SEED_DRAWN = {"code": "seed-drawn-again", "field": "seed", "before": 42}
SEED_STEPPED = {"code": "seed-per-output", "field": "seed"}
SEED_PER_DRAFT = {"code": "seed-per-draft", "field": "seed", "source": "chat", "before": 42}
#: Every change an answer in this module names, by the expectation that names it.
NAMED = (
    STEPS_SET_ASIDE,
    CFG_NOT_TAKEN,
    SAMPLER_SET_ASIDE,
    SCHEDULER_FIXED,
    FRAMES_NOT_OFFERED,
    WIDTH_FOLLOWS_SOURCE,
    ENLARGEMENT_FIXED,
    FOR_OFFER,
    BATCH_SIZE_MADE_ONE,
    COUNT_RAISED,
    ONE_REPLY,
    STRENGTH_FOR_WORDS,
    LENGTH_ALIGNED,
    SEED_DRAWN,
    SEED_STEPPED,
    SEED_PER_DRAFT,
)


def _sentences(response: Any) -> set[str]:
    """The sentences an answer's changes are said with."""

    return {item["message"] for item in response.json()["setting_changes"]}


def _changes(response: Any, status: int = 202) -> list[dict[str, Any]]:
    """The changes an answer names, each checked to carry a sentence, without the sentences.

    The list is left out of an answer that names none, and no item names a
    value that is not a number, true or false, or text.
    """

    assert response.status_code == status, response.text
    body = response.json()
    assert body.get("setting_changes") != [], "an answer that names no change leaves the list out"
    items: list[dict[str, Any]] = body.get("setting_changes", [])
    for item in items:
        message = item.pop("message")
        assert isinstance(message, str) and message.endswith("."), item
        for name in ("before", "after"):
            assert name not in item or isinstance(item[name], int | float | str), item
    return items


async def _chat(client: AsyncClient, **fields: Any) -> str:
    chat = await client.post("/api/chats", json={"title": "Setting changes", **fields})
    assert chat.status_code == 201, chat.text
    chat_id: str = chat.json()["id"]
    return chat_id


async def _save_for_chat(
    client: AsyncClient, chat_id: str, role: str, settings: dict[str, Any]
) -> None:
    saved = await client.patch(
        f"/api/chats/{chat_id}", json={"generation_settings_json": {role: settings}}
    )
    assert saved.status_code == 200, saved.text


def _preset(role: str, settings: dict[str, Any]) -> str:
    """A preset holding these settings as they are, as one saved under an older workflow can."""

    with SessionLocal() as session:
        preset = GenerationPreset(name=new_id("changes"), role=role, settings_json=settings)
        session.add(preset)
        session.commit()
        return preset.id


async def _bind_chat_preset(
    client: AsyncClient, chat_id: str, role: str, settings: dict[str, Any]
) -> None:
    bound = await client.patch(
        f"/api/chats/{chat_id}",
        json={"generation_preset_ids_json": {role: _preset(role, settings)}},
    )
    assert bound.status_code == 200, bound.text


def _probe(schema: dict[str, Any] = PROBE, *, operation: str = "text_to_image") -> dict[str, str]:
    """What a turn names to run on a workflow with this schema, and a model for it."""

    role = "video" if "video" in operation else "image"
    return {
        "workflow_revision_id": _revision(
            "Change probe", operation=operation, input_schema_json=schema
        ),
        "profile_id": _profile("Change probe", role=role),
    }


async def _send(app: FastAPI, client: AsyncClient, chat_id: str, **fields: Any) -> Any:
    """A picture turn, held queued so that nothing it makes runs while it is read."""

    async with app.state.services.scheduler.lease("primary"):
        return await client.post(
            f"/api/chats/{chat_id}/turns", json={"text": TEXT, "mode": "image", **fields}
        )


def _runs(plan_id: str) -> list[Run]:
    """The runs of one accepted plan, in step order."""

    with SessionLocal() as session:
        return list(
            session.scalars(
                select(Run)
                .join(WorkStep, WorkStep.id == Run.work_step_id)
                .where(Run.work_plan_id == plan_id)
                .order_by(WorkStep.ordinal)
            )
        )


def _seeds(plan_id: str) -> list[int]:
    """The seed each output of an accepted plan runs with, in step order."""

    return [run.settings_json["seed"] for run in _runs(plan_id)]


async def _prompt_batch(
    client: AsyncClient, chat_id: str, *workflows: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """A Prompt Library batch of two drafts on workflows taking these properties, and their ids.

    One workflow makes both drafts; two take one draft each, in turn.
    """

    workflow_ids: list[str] = []
    for index, properties in enumerate(workflows, start=1):
        created = await client.post(
            "/api/workflows",
            json={
                "name": f"Changes batch workflow {index}",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {"node": {"class_type": f"ChangesBatchWorkflow{index}"}},
                "input_schema": {"type": "object", "properties": properties},
            },
        )
        assert created.status_code == 201, created.text
        seed_workflow_trust(created.json()["current_revision_id"])
        workflow_ids.append(created.json()["current_revision_id"])
    options = [
        {"workflow_revision_id": workflow_id, "lora_policy": {"mode": "none"}}
        for workflow_id in workflow_ids
    ]
    policy = (
        {"mode": "fixed", **options[0]}
        if len(options) == 1
        else {"mode": "pool", "strategy": "round_robin", "options": options}
    )
    template = await client.post(
        "/api/prompt-templates",
        json={
            "idempotency_key": "changes-batch-template",
            "name": "Changes batch",
            "description": "One subject slot",
            "contract": {
                "schema_version": 1,
                "operation": "text_to_image",
                "body": "A {{subject}}.",
                "slots": [{"name": "subject", "mode": "input", "variation_scope": "item"}],
                "resource_policy": policy,
            },
        },
    )
    assert template.status_code == 201, template.text
    revision = template.json()["revision"]
    created_batch = await client.post(
        f"/api/chats/{chat_id}/prompt-batches",
        json={
            "idempotency_key": "changes-batch-preview",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": 2,
            "selection_seed": 0,
            "inputs": {"subject": ["gray square", "ceramic cup"]},
        },
    )
    assert created_batch.status_code in (200, 201), created_batch.text
    batch: dict[str, Any] = created_batch.json()
    return batch, workflow_ids


async def _save_as_default(client: AsyncClient, settings: dict[str, Any]) -> None:
    """Save image settings in the default preset, which a batch does not check per workflow."""

    default = await client.post(
        "/api/presets",
        json={"name": new_id("default"), "role": "image", "settings": settings, "is_default": True},
    )
    assert default.status_code == 201, default.text


async def _imported_chat(
    client: AsyncClient, preset: dict[str, Any], *, project: dict[str, Any] | None = None
) -> str:
    """A chat of a project brought in from an archive, bound to an image preset holding `preset`.

    An archive's presets are stored as it carries them, so a retired key
    arrives in one as an older version saved it. `project` is the project's
    own image settings as the archive carries them, where a key the engine
    does not describe is kept.
    """

    created_preset = await client.post(
        "/api/presets",
        json={"name": "Archived changes", "role": "image", "settings": {"steps": 20}},
    )
    assert created_preset.status_code == 201, created_preset.text
    preset_id = created_preset.json()["id"]
    created_project = await client.post("/api/projects", json={"name": "Archived changes"})
    assert created_project.status_code == 201, created_project.text
    project_id = created_project.json()["id"]
    chat_id = await _chat(client, project_id=project_id)
    bound = await client.patch(
        f"/api/chats/{chat_id}", json={"generation_preset_ids_json": {"image": preset_id}}
    )
    assert bound.status_code == 200, bound.text
    exported = await client.post(
        f"/api/projects/{project_id}/export", params={"include_media": False}
    )
    assert exported.status_code == 201, exported.text
    archive = await client.get(exported.json()["url"])
    manifest = _manifest(archive.content)
    [carried] = [
        item for item in manifest["dependencies"]["presets"] if item["source_id"] == preset_id
    ]
    carried["settings"] = preset
    if project is not None:
        manifest["project"]["generation_settings_json"] = {"image": project}
    imported = await client.post(
        "/api/projects/import",
        files={
            "archive": (
                "archived-changes.lm-atelier.zip",
                _rewrite_archive(archive.content, manifest),
                "application/zip",
            )
        },
    )
    assert imported.status_code == 201, imported.text
    chats = await client.get("/api/chats", params={"project_id": imported.json()["id"]})
    assert chats.status_code == 200, chats.text
    [chat] = chats.json()
    return str(chat["id"])


async def _made(client: AsyncClient, chat_id: str, **fields: Any) -> dict[str, Any]:
    """A picture turn made to the end, so its response can be regenerated."""

    response = await client.post(
        f"/api/chats/{chat_id}/turns", json={"text": TEXT, "mode": "image", **fields}
    )
    assert response.status_code == 202, response.text
    await _finished(client, response.json()["run"]["id"])
    accepted: dict[str, Any] = response.json()
    return accepted


# Saved values the chosen workflow does not take.


async def test_saved_values_the_workflow_does_not_take_are_named_with_why(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id = await _chat(client)
    await _save_for_chat(
        client,
        chat_id,
        "image",
        {"steps": 150, "cfg": 5, "sampler": "unlisted_sampler", "scheduler": "normal"},
    )

    response = await _send(app, client, chat_id, **_probe())

    assert _changes(response) == [
        STEPS_SET_ASIDE,
        CFG_NOT_TAKEN,
        SAMPLER_SET_ASIDE,
        SCHEDULER_FIXED,
    ]
    assert "unlisted_sampler" not in response.text
    with SessionLocal() as session:
        run = session.get(Run, response.json()["run"]["id"])
        assert run is not None
        assert run.settings_json["steps"] == 12
        assert [item["code"] for item in run.provenance_json["setting_changes"]] == [
            "set-aside-not-usable",
            "set-aside-not-taken",
            "set-aside-not-usable",
            "set-aside-fixed",
        ]


async def test_the_value_used_for_one_set_aside_is_the_one_saved_below_it(
    app: FastAPI, client: AsyncClient
) -> None:
    project = await client.post(
        "/api/projects",
        json={"name": "Setting changes", "generation_settings_json": {"image": {"steps": 20}}},
    )
    assert project.status_code == 201, project.text
    chat_id = await _chat(client, project_id=project.json()["id"])
    await _save_for_chat(client, chat_id, "image", {"steps": 150})

    response = await _send(app, client, chat_id, **_probe())

    assert _changes(response) == [{**STEPS_SET_ASIDE, "after": 20}]


async def test_a_value_the_turn_sends_itself_leaves_nothing_saved_to_name(
    app: FastAPI, client: AsyncClient
) -> None:
    """The turn's own value wins over the saved one, so nothing is set aside."""

    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"steps": 150})

    response = await _send(app, client, chat_id, **_probe(), settings={"steps": 20})

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()
    with SessionLocal() as session:
        run = session.get(Run, response.json()["run"]["id"])
        assert run is not None
        assert "setting_changes" not in run.provenance_json


async def test_a_turn_that_changes_nothing_answers_without_the_list(
    app: FastAPI, client: AsyncClient
) -> None:
    response = await _send(app, client, await _chat(client))

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()


@pytest.mark.parametrize("layer", sorted(SOURCES))
async def test_a_value_set_aside_names_the_layer_it_is_saved_in(
    app: FastAPI, client: AsyncClient, layer: str
) -> None:
    probe = _probe()
    chat_id = await _chat(client)
    turn = await _save(client, chat_id, probe["profile_id"], {layer: {"steps": 150}})

    response = await _send(app, client, chat_id, **probe, **turn)

    assert _changes(response) == [{**STEPS_SET_ASIDE, "source": SOURCES[layer]}]


async def test_a_models_load_settings_are_not_named(app: FastAPI, client: AsyncClient) -> None:
    """They configure the model rather than the render."""

    probe = _probe()
    chat_id = await _chat(client)
    await _save(
        client, chat_id, probe["profile_id"], {"model profile load settings": {"steps": 150}}
    )

    response = await _send(app, client, chat_id, **probe)

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()


async def test_a_size_a_workflow_fixes_is_named_with_the_size_it_makes(
    app: FastAPI, client: AsyncClient
) -> None:
    fixed = {"type": "integer", "const": 1024}
    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"width": 2176, "height": 768})

    response = await _send(
        app,
        client,
        chat_id,
        **_probe({"type": "object", "properties": {"width": fixed, "height": dict(fixed)}}),
    )

    side = {"code": "set-aside-fixed", "source": "chat", "after": 1024}
    assert _changes(response) == [
        {**side, "field": "width", "before": 2176},
        {**side, "field": "height", "before": 768},
    ]


async def test_a_saved_length_a_workflow_sets_itself_is_named_as_not_offered(
    app: FastAPI, client: AsyncClient
) -> None:
    """A video workflow that describes its controls and not its length fixes the length."""

    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "video", {"frames": 10})
    probe = _probe(
        {"type": "object", "properties": {"width": {"type": "integer", "default": 512}}},
        operation="text_to_video",
    )

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={"text": VIDEO_TEXT, "mode": "video", "confirm_media": True, **probe},
        )

    assert _changes(response) == [FRAMES_NOT_OFFERED]


async def test_a_saved_size_on_an_edit_follows_the_source(
    app: FastAPI, client: AsyncClient
) -> None:
    side = {"type": "integer", "readOnly": True}
    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"width": 2176, "height": 768})
    probe = _probe(
        {"type": "object", "properties": {"width": side, "height": dict(side)}},
        operation="image_to_image",
    )

    response = await _send(
        app,
        client,
        chat_id,
        text="Make the cup blue",
        input_artifact_ids=[await _picture(client)],
        **probe,
    )

    assert response.json()["run"]["operation"] == "image_to_image"
    assert _sentences(response) == {
        "Work made from a picture keeps that picture's shape, so this saved size was set aside."
    }
    assert _changes(response) == [
        WIDTH_FOLLOWS_SOURCE,
        {**WIDTH_FOLLOWS_SOURCE, "field": "height", "before": 768},
    ]


async def test_a_set_aside_value_is_named_only_when_it_is_a_number_or_one_of_its_choices(
    app: FastAPI, client: AsyncClient
) -> None:
    """Text, lists, sets of named values, a prompt's words and unnamed keys are never named."""

    chat_id = await _chat(client)
    await _bind_chat_preset(
        client,
        chat_id,
        "image",
        {
            "caption_note": MARKER,
            "layout_hint": {"rows": 2},
            "tags": ["gray", "square"],
            "Not A Name": 3,
            "negative_prompt": 5,
        },
    )

    response = await _send(app, client, chat_id, **_probe())

    not_taken = {"code": "set-aside-not-taken", "source": "chat_preset"}
    assert _changes(response) == [
        {**not_taken, "field": "caption_note"},
        {**not_taken, "field": "layout_hint"},
        {**not_taken, "field": "tags"},
        {**not_taken, "field": "*"},
        {"code": "set-aside-not-usable", "field": "negative_prompt", "source": "chat_preset"},
    ]
    assert MARKER not in response.text


async def test_a_choice_is_not_named_from_a_menu_that_holds_one_it_may_not_name(
    app: FastAPI, client: AsyncClient
) -> None:
    menu = {"type": "string", "default": "euler", "enum": ["euler", "model.v2"]}
    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"sampler": "heun"})

    response = await _send(
        app, client, chat_id, **_probe({"type": "object", "properties": {"sampler": menu}})
    )

    assert _changes(response) == [
        {"code": "set-aside-not-usable", "field": "sampler", "source": "chat"}
    ]


async def test_each_saved_value_set_aside_above_the_one_kept_is_named(
    app: FastAPI, client: AsyncClient
) -> None:
    """A value set aside gives way to the next one down, so each is named down to one kept."""

    probe = _probe()
    chat_id = await _chat(client)
    turn = await _save(
        client,
        chat_id,
        probe["profile_id"],
        {"chat": {"steps": 150}, "project": {"steps": 99}, "project preset": {"steps": 20}},
    )

    response = await _send(app, client, chat_id, **probe, **turn)

    assert _changes(response) == [
        {**STEPS_SET_ASIDE, "after": 20},
        {**STEPS_SET_ASIDE, "source": "project", "before": 99, "after": 20},
    ]


async def test_a_retired_key_saved_below_the_turns_own_value_is_not_named(
    app: FastAPI, client: AsyncClient
) -> None:
    """A CFG saved under the name it had before gives way to the CFG the turn sends itself."""

    chat_id = await _imported_chat(client, {"guidance": 99})

    response = await _send(app, client, chat_id, **_probe(CFG_UP_TO_TEN), settings={"cfg": 5})

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()
    assert response.json()["run"]["settings_json"]["cfg"] == 5


async def test_a_retired_key_that_wins_leaves_a_lower_value_unnamed(
    app: FastAPI, client: AsyncClient
) -> None:
    """The chat's preset saves CFG by its retired name, above a project CFG out of range."""

    chat_id = await _imported_chat(client, {"guidance": 7}, project={"cfg": 20})

    response = await _send(app, client, chat_id, **_probe(CFG_UP_TO_TEN))

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()
    assert response.json()["run"]["settings_json"]["cfg"] == 7


async def test_a_retired_key_saved_beside_the_key_that_replaced_it_is_named(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id = await _imported_chat(client, {"guidance": 7, "cfg": 5})

    response = await _send(app, client, chat_id, **_probe(CFG_UP_TO_TEN))

    assert _changes(response) == [
        {"code": "set-aside-not-taken", "field": "guidance", "source": "chat_preset", "before": 7}
    ]
    assert response.json()["run"]["settings_json"]["cfg"] == 5


async def test_a_guidance_a_workflow_takes_saved_beside_a_cfg_is_not_named(
    app: FastAPI, client: AsyncClient
) -> None:
    """Where the workflow takes guidance by that name, it is no retired name for CFG."""

    chat_id = await _imported_chat(client, {"guidance": 2, "cfg": 5})

    response = await _send(app, client, chat_id, **_probe(GUIDANCE_AND_CFG))

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()
    assert response.json()["run"]["settings_json"]["guidance"] == 2
    assert response.json()["run"]["settings_json"]["cfg"] == 5


async def test_a_guidance_a_workflow_takes_is_followed_on_its_own_below_a_cfg(
    app: FastAPI, client: AsyncClient
) -> None:
    """A CFG the chat's preset saves does not stand for a guidance saved below it."""

    chat_id = await _imported_chat(client, {"cfg": 5}, project={"guidance": 99})

    response = await _send(app, client, chat_id, **_probe(GUIDANCE_AND_CFG))

    assert _changes(response) == [
        {
            "code": "set-aside-not-usable",
            "field": "guidance",
            "source": "project",
            "before": 99,
            "after": 3,
        }
    ]


async def test_a_guidance_a_workflow_takes_is_not_named_beside_a_cfg_set_aside(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id = await _imported_chat(client, {"guidance": 2, "cfg": 99})

    response = await _send(app, client, chat_id, **_probe(GUIDANCE_AND_CFG))

    assert _changes(response) == [
        {
            "code": "set-aside-not-usable",
            "field": "cfg",
            "source": "chat_preset",
            "before": 99,
            "after": 4,
        }
    ]
    assert response.json()["run"]["settings_json"]["guidance"] == 2


async def test_a_size_fixed_for_work_from_a_picture_is_named_with_the_size_it_makes(
    app: FastAPI, client: AsyncClient
) -> None:
    fixed = {"type": "integer", "const": 1024}
    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"width": 2176, "height": 768})
    probe = _probe(
        {"type": "object", "properties": {"width": fixed, "height": dict(fixed)}},
        operation="image_to_image",
    )

    response = await _send(
        app,
        client,
        chat_id,
        text="Make the cup blue",
        input_artifact_ids=[await _picture(client)],
        **probe,
    )

    side = {"code": "set-aside-fixed", "source": "chat", "after": 1024}
    assert _changes(response) == [
        {**side, "field": "width", "before": 2176},
        {**side, "field": "height", "before": 768},
    ]


async def test_a_saved_strength_mode_is_read_by_the_strength_and_never_named(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"_image_edit_strength_mode": "auto"})
    probe = _probe({"type": "object", "properties": {}}, operation="image_to_image")

    response = await _send(
        app,
        client,
        chat_id,
        text="Make the cup blue",
        input_artifact_ids=[await _picture(client)],
        **probe,
    )

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()


async def test_a_text_reply_names_no_saved_value_its_model_does_not_take(
    client: AsyncClient,
) -> None:
    """A saved value a reply's model does not take is named only for a media turn."""

    chat_id = await _chat(client)
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.generation_settings_json = {"chat": {"temperature": "warm"}}
        session.commit()

    response = await client.post(
        f"/api/chats/{chat_id}/turns", json={"text": "Describe a ceramic cup", "mode": "text"}
    )

    assert response.json()["run"]["operation"] == "text"
    assert "setting_changes" not in response.json()


# The turn's own values.


async def test_an_enlargement_factor_a_workflow_fixes_itself_is_named(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id, revision_id, payload = await _selection(client, adjustable=False)

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                **payload,
                "workflow_revision_id": revision_id,
                "settings": {"upscale_factor": 2},
            },
        )

    assert _changes(response) == [ENLARGEMENT_FIXED]
    assert "upscale_factor" not in response.json()["run"]["settings_json"]


async def test_an_enlargement_factor_a_workflow_takes_is_not_named(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id, revision_id, payload = await _selection(client, adjustable=True)

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                **payload,
                "workflow_revision_id": revision_id,
                "settings": {"upscale_factor": 3},
            },
        )

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()
    assert response.json()["run"]["settings_json"]["upscale_factor"] == 3


async def test_a_value_a_recipe_chose_is_named_as_saved(app: FastAPI, client: AsyncClient) -> None:
    family_id, revision_id = _workflow("image_generation", "text_to_image", "image")
    chat_id = await _chat(client)
    recipe = await client.post(
        "/api/workflow-use-case-presets",
        json={
            "name": "Three at once",
            "use_case": "image_generation",
            "settings_json": {"batch_size": 3},
            "is_default": True,
        },
    )
    assert recipe.status_code == 201, recipe.text
    selected = await client.put(
        f"/api/chats/{chat_id}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": family_id},
    )
    assert selected.status_code == 200, selected.text

    response = await _send(app, client, chat_id)

    assert response.json()["run"]["workflow_revision_id"] == revision_id
    assert _changes(response) == [{**BATCH_SIZE_MADE_ONE, "source": "saved"}]


async def test_a_saved_batch_size_made_into_separate_outputs_is_named(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"batch_size": 3})

    response = await _send(app, client, chat_id)

    assert len(response.json()["assistant_messages"]) == 3
    assert _changes(response) == [BATCH_SIZE_MADE_ONE]


async def test_a_count_the_words_raise_is_named(app: FastAPI, client: AsyncClient) -> None:
    response = await _send(
        app,
        client,
        await _chat(client),
        text="Create three images of a ceramic cup",
        output_count=2,
    )

    assert len(response.json()["assistant_messages"]) == 3
    assert _changes(response) == [COUNT_RAISED]


async def test_a_count_given_to_a_text_reply_is_named(app: FastAPI, client: AsyncClient) -> None:
    chat_id = await _chat(client)

    response = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={"text": "Describe a ceramic cup", "mode": "text", "output_count": 3},
    )

    assert response.json()["run"]["operation"] == "text"
    assert _changes(response) == [ONE_REPLY]


async def test_accepting_an_offer_names_what_the_reply_sent_with_it(client: AsyncClient) -> None:
    chat, _assistant = await create_offer(client, multiple=False)

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Yes, please",
            "mode": "text",
            "settings": {"max_tokens": 64},
            "output_count": 2,
            "confirm_media": True,
        },
    )

    assert response.json()["run"]["operation"] == "text_to_image"
    assert _sentences(response) == {
        "Accepting an offer makes what was offered, so the settings, ordered settings and count "
        "the reply sent were set aside; a role override for the offered role still applies."
    }
    assert _changes(response) == [
        FOR_OFFER,
        {"code": "set-aside-for-offer", "field": "output_count", "before": 2},
    ]


async def test_accepting_an_offer_names_the_ordered_settings_the_reply_sent(
    client: AsyncClient,
) -> None:
    chat, _assistant = await create_offer(client, multiple=False)

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Yes, please",
            "mode": "text",
            "ordered_settings": {"image": {"steps": 10}},
            "confirm_media": True,
        },
    )

    assert _changes(response) == [{"code": "set-aside-for-offer", "field": "steps", "before": 10}]


async def test_a_role_override_a_reply_accepting_an_offer_sends_still_applies(
    client: AsyncClient,
) -> None:
    """Its shared settings are set aside; an override for the offered role is not."""

    chat, _assistant = await create_offer(client, multiple=False)

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Yes, please",
            "mode": "text",
            "settings": {"steps": 10},
            "role_overrides": {"image": {"settings": {"steps": 12}}},
            "confirm_media": True,
        },
    )

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()
    assert response.json()["run"]["settings_json"]["steps"] == 12


async def test_accepting_an_offer_names_more_settings_than_one_request_may_name_once(
    client: AsyncClient,
) -> None:
    chat, _assistant = await create_offer(client, multiple=False)

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Yes, please",
            "mode": "text",
            "settings": {f"note_{index}": index for index in range(MAX_SETTING_FIELDS + 1)},
            "output_count": 2,
            "confirm_media": True,
        },
    )

    assert _changes(response) == [
        {"code": "set-aside-for-offer", "field": "*"},
        {"code": "set-aside-for-offer", "field": "output_count", "before": 2},
    ]


@pytest.mark.parametrize("multiple", [False, True], ids=["one offer", "several offers"])
async def test_a_branch_of_a_reply_accepting_an_offer_names_nothing_it_runs_as_carried(
    app: FastAPI, client: AsyncClient, multiple: bool
) -> None:
    """The branch carries over every setting the reply ran with, and runs with them again."""

    chat, _assistant = await create_offer(client, multiple=multiple)
    await _save_for_chat(client, chat["id"], "image", {"seed": 42})
    assent = {"text": "Generate them" if multiple else "Yes, please", "confirm_media": True}

    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat['id']}/turns", json={**assent, "mode": "text"}
        )
        assert accepted.status_code == 202, accepted.text
        branched = await client.post(
            f"/api/messages/{accepted.json()['user_message']['id']}/branch", json=assent
        )

    assert len(branched.json()["assistant_messages"]) == (2 if multiple else 1)
    assert _changes(branched) == []


async def test_accepting_several_offers_names_it_once_for_every_step(
    client: AsyncClient,
) -> None:
    chat, _assistant = await create_offer(client, multiple=True)

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Generate them",
            "mode": "text",
            "settings": {"max_tokens": 64},
            "confirm_media": True,
        },
    )

    assert len(response.json()["assistant_messages"]) == 2
    assert _changes(response) == [FOR_OFFER]


async def test_a_strength_given_to_a_picture_made_from_words_is_named(
    app: FastAPI, client: AsyncClient
) -> None:
    response = await _send(app, client, await _chat(client), settings={"denoise": 0.5})

    assert _changes(response) == [STRENGTH_FOR_WORDS]
    assert response.json()["run"]["settings_json"]["denoise"] == 1


async def test_a_strength_the_turns_preset_gives_is_named_and_a_saved_one_is_not(
    app: FastAPI, client: AsyncClient
) -> None:
    """A strength saved for every turn would be named on every new picture."""

    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"denoise": 0.5})

    saved = await _send(app, client, chat_id)
    preset = await _send(app, client, chat_id, preset_id=_preset("image", {"denoise": 0.5}))

    assert saved.status_code == 202, saved.text
    assert "setting_changes" not in saved.json()
    assert _changes(preset) == [{**STRENGTH_FOR_WORDS, "source": "turn_preset"}]


@pytest.mark.parametrize("given", ["sent", "turn preset"])
async def test_a_length_the_workflow_makes_in_whole_frames_is_named(
    app: FastAPI, client: AsyncClient, given: str
) -> None:
    chat_id = await _chat(client)
    probe = _probe(ALIGNED_LENGTH, operation="text_to_video")
    length = {"duration_seconds": 2.0}
    turn = {"settings": length} if given == "sent" else {"preset_id": _preset("video", length)}

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={"text": VIDEO_TEXT, "mode": "video", **probe, **turn},
        )

    assert _changes(response) == [
        LENGTH_ALIGNED if given == "sent" else {**LENGTH_ALIGNED, "source": "turn_preset"}
    ]
    assert response.json()["run"]["settings_json"]["frames"] == 33


async def test_a_length_already_in_whole_frames_is_not_named(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id = await _chat(client)
    probe = _probe(ALIGNED_LENGTH, operation="text_to_video")

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": VIDEO_TEXT,
                "mode": "video",
                **probe,
                "settings": {"duration_seconds": 3.0625},
            },
        )

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()
    assert response.json()["run"]["settings_json"]["frames"] == 49


# Seeds.


@pytest.mark.parametrize("sent", [True, False], ids=["sent", "carried over"])
async def test_a_regenerate_names_the_seed_it_draws_again(
    app: FastAPI, client: AsyncClient, sent: bool
) -> None:
    chat_id = await _chat(client)
    original = await _made(client, chat_id, settings={"seed": 42})

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/messages/{original['assistant_message']['id']}/regenerate",
            json={"settings": {"seed": 42} if sent else {}},
        )

    assert _changes(response) == [SEED_DRAWN if sent else {**SEED_DRAWN, "source": "inherited"}]
    assert response.json()["run"]["settings_json"]["seed"] != 42


@pytest.mark.parametrize("saved", [False, True], ids=["sent", "saved for the chat"])
async def test_a_seed_given_to_several_outputs_is_named_as_stepped(
    app: FastAPI, client: AsyncClient, saved: bool
) -> None:
    chat_id = await _chat(client)
    if saved:
        await _save_for_chat(client, chat_id, "image", {"seed": 42})

    response = await _send(
        app, client, chat_id, output_count=3, **({} if saved else {"settings": {"seed": 42}})
    )

    assert _changes(response) == [{**SEED_STEPPED, "source": "chat"} if saved else SEED_STEPPED]
    assert _seeds(response.json()["run"]["work_plan_id"]) == [42, 43, 44]


async def test_a_seed_drawn_for_several_outputs_is_not_named(
    app: FastAPI, client: AsyncClient
) -> None:
    """A seed of -1 asks for one to be drawn, so there is no seed given to step."""

    response = await _send(app, client, await _chat(client), output_count=2)

    assert response.status_code == 202, response.text
    assert "setting_changes" not in response.json()


async def test_a_saved_seed_each_draft_replaces_is_named_for_the_batch(
    client: AsyncClient,
) -> None:
    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"seed": 42})
    batch = await _batch(client, chat_id, ["gray square", "ceramic cup"])

    queued = await _queue(client, batch)
    replayed = await _queue(client, batch)

    assert _changes(queued) == [SEED_PER_DRAFT]
    assert replayed.json()["replayed"] is True
    assert _changes(replayed) == [SEED_PER_DRAFT]
    assert len(set(_seeds(queued.json()["work_plan_id"]))) == 2


async def test_a_saved_seed_a_batchs_workflow_does_not_take_is_named_once_as_replaced(
    client: AsyncClient,
) -> None:
    """Each draft runs with a seed of its own, whether or not its workflow takes the saved one."""

    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"seed": 42})
    batch, _workflow_ids = await _prompt_batch(client, chat_id, SEED_UP_TO_TEN)

    queued = await _queue(client, batch)

    assert _changes(queued) == [SEED_PER_DRAFT]


async def test_a_saved_seed_one_pooled_workflow_does_not_take_is_named_once_as_replaced(
    client: AsyncClient,
) -> None:
    await _save_as_default(client, {"seed": 42})
    chat_id = await _chat(client)
    batch, _workflow_ids = await _prompt_batch(client, chat_id, SEED_UP_TO_TEN, {})

    queued = await _queue(client, batch)

    assert _changes(queued) == [{**SEED_PER_DRAFT, "source": "default_preset"}]


async def test_a_saved_seed_several_outputs_cannot_take_names_no_seed_as_used(
    app: FastAPI, client: AsyncClient
) -> None:
    """The outputs run with the workflow's own seed stepped, so no one seed is the one used."""

    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"seed": 42})

    response = await _send(
        app,
        client,
        chat_id,
        output_count=3,
        **_probe({"type": "object", "properties": SEED_UP_TO_TEN}),
    )

    assert _changes(response) == [
        {"code": "set-aside-not-usable", "field": "seed", "source": "chat", "before": 42}
    ]
    assert _seeds(response.json()["run"]["work_plan_id"]) == [5, 6, 7]


async def test_a_batch_on_several_workflows_names_a_change_only_some_drafts_make_by_step(
    client: AsyncClient,
) -> None:
    """The first workflow takes 30 steps and the second 20, so only the second sets 25 aside."""

    steps = {"type": "integer", "default": 8, "minimum": 1}
    await _save_as_default(client, {"steps": 25})
    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"seed": 42})
    batch, workflow_ids = await _prompt_batch(
        client,
        chat_id,
        {"steps": {**steps, "maximum": 30}},
        {"steps": {**steps, "maximum": 20}},
    )

    queued = await _queue(client, batch)

    assert _changes(queued) == [
        SEED_PER_DRAFT,
        {
            "code": "set-aside-not-usable",
            "field": "steps",
            "source": "default_preset",
            "before": 25,
            "after": 8,
            "step": 2,
        },
    ]
    runs = _runs(queued.json()["work_plan_id"])
    assert [run.workflow_revision_id for run in runs] == workflow_ids
    assert [run.settings_json["steps"] for run in runs] == [25, 8]


async def test_each_draft_of_a_batch_on_several_workflows_names_its_batch_size(
    client: AsyncClient,
) -> None:
    await _save_as_default(client, {"batch_size": 3})
    chat_id = await _chat(client)
    batch, _workflow_ids = await _prompt_batch(client, chat_id, {}, {})

    queued = await _queue(client, batch)

    assert _changes(queued) == [{**BATCH_SIZE_MADE_ONE, "source": "default_preset"}]
    assert [run.settings_json["batch_size"] for run in _runs(queued.json()["work_plan_id"])] == [
        1,
        1,
    ]


# Plans of steps, and the routes that carry values over.


async def test_a_step_of_an_ordered_plan_names_its_own_change_by_step(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id = await _chat(client)
    await _bind_chat_preset(client, chat_id, "image", {"caption_note": 3})

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={"text": STORY_THEN_PICTURE, "mode": "auto", "confirm_media": True},
        )

    assert [run.operation for run in _runs(response.json()["run"]["work_plan_id"])] == [
        "text",
        "text_to_image",
    ]
    assert _changes(response) == [
        {
            "code": "set-aside-not-taken",
            "field": "caption_note",
            "source": "chat_preset",
            "before": 3,
            "step": 2,
        }
    ]


async def test_a_step_of_a_plan_names_the_strength_its_new_picture_sets_aside(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{await _chat(client)}/turns",
            json={
                "text": STORY_THEN_PICTURE,
                "mode": "auto",
                "confirm_media": True,
                "ordered_settings": {"image": {"denoise": 0.5}},
            },
        )

    assert _changes(response) == [{**STRENGTH_FOR_WORDS, "step": 2}]


async def test_a_step_of_a_plan_names_the_length_its_workflow_aligns(
    app: FastAPI, client: AsyncClient
) -> None:
    animation = {
        "workflow_revision_id": _revision(
            "Aligned animation", operation="image_to_video", input_schema_json=ALIGNED_LENGTH
        ),
        "profile_id": _profile("Aligned animation", role="video"),
    }

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{await _chat(client)}/turns",
            json={
                "text": STORY_PICTURE_VIDEO,
                "mode": "auto",
                "confirm_media": True,
                "role_overrides": {"video": animation},
                "ordered_settings": {"video": {"duration_seconds": 2.0}},
            },
        )

    runs = _runs(response.json()["run"]["work_plan_id"])
    assert [run.operation for run in runs] == ["text", "text_to_image", "image_to_video"]
    assert _changes(response) == [{**LENGTH_ALIGNED, "step": 3}]
    assert runs[2].settings_json["frames"] == 33


async def test_an_edit_and_a_branch_name_a_change_to_what_they_carry_over_as_inherited(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id = await _chat(client)
    carried = {**SEED_STEPPED, "source": "inherited"}
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={"text": TEXT, "mode": "image", "settings": {"seed": 42}, "output_count": 3},
        )
        message_id = source.json()["user_message"]["id"]
        edit = {"text": TEXT, "mode": "image", "idempotency_key": "carried-seed-edit"}
        edited = await client.post(f"/api/messages/{message_id}/edits", json=edit)
        edited_again = await client.post(f"/api/messages/{message_id}/edits", json=edit)
        branched = await client.post(
            f"/api/messages/{message_id}/branch",
            json={"text": TEXT, "mode": "image", "output_count": 3},
        )

    assert _changes(source) == [SEED_STEPPED]
    assert _changes(edited) == [carried]
    assert edited_again.json()["replayed"] is True
    assert _changes(edited_again) == [carried]
    assert _changes(branched) == [carried]


async def test_a_replayed_turn_and_regenerate_answer_the_same_changes(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id = await _chat(client)
    await _save_for_chat(client, chat_id, "image", {"batch_size": 3})
    turn = {"idempotency_key": "replayed-changes"}
    first = await _send(app, client, chat_id, **turn)
    again = await _send(app, client, chat_id, **turn)
    original = await _made(client, await _chat(client), settings={"seed": 42})
    regenerate = {"settings": {"seed": 42}, "idempotency_key": "replayed-regenerate"}
    async with app.state.services.scheduler.lease("primary"):
        regenerated = await client.post(
            f"/api/messages/{original['assistant_message']['id']}/regenerate", json=regenerate
        )
        regenerated_again = await client.post(
            f"/api/messages/{original['assistant_message']['id']}/regenerate", json=regenerate
        )

    assert again.json()["replayed"] is True
    assert _changes(first) == _changes(again) == [BATCH_SIZE_MADE_ONE]
    assert regenerated_again.json()["replayed"] is True
    assert _changes(regenerated) == _changes(regenerated_again) == [SEED_DRAWN]


def test_every_change_in_the_vocabulary_is_one_a_render_can_make() -> None:
    """Each code is reached through a real route, and each has one sentence of its own.

    The strength an extension sets aside is reached in its own module, which
    needs a reviewed source-canvas workflow.
    """

    from local_lm.setting_changes import SETTING_CHANGE_MESSAGES, SettingChangeCode

    reached = {item["code"] for item in NAMED} | {"strength-full-for-extension"}
    assert reached == {code.value for code in SettingChangeCode}
    assert set(SETTING_CHANGE_MESSAGES) == set(SettingChangeCode)
    assert len(set(SETTING_CHANGE_MESSAGES.values())) == len(SettingChangeCode)
    assert all(message.endswith(".") for message in SETTING_CHANGE_MESSAGES.values())
