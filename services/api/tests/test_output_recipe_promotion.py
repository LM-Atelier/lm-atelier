"""A recipe drafted from one finished generation, to review before saving it."""

from __future__ import annotations

import io
from typing import Any

from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import func, select
from test_generation_experiment_preflight import _revision
from test_output_recipe_replay import PROMPT, _finished, _profile, _workflow

from local_lm.db import SessionLocal
from local_lm.generation_experiments_v1 import RECIPE_LEFT_OUT_MESSAGES
from local_lm.models import (
    Message,
    ModelProfile,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
    WorkflowUseCasePreset,
)
from local_lm.workflow_graph_settings_v1 import GRAPH_SETTINGS_SCHEMA_KEY

RECIPES = "/api/workflow-use-case-presets"
OPEN_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _draft_path(run_id: str) -> str:
    return f"/api/runs/{run_id}/recipe-draft"


def _left_out(setting: str, reason: str) -> dict[str, str]:
    return {"setting": setting, "reason": reason, "message": RECIPE_LEFT_OUT_MESSAGES[reason]}


def _counts() -> dict[str, int]:
    with SessionLocal() as session:
        return {
            model.__name__: session.scalar(select(func.count()).select_from(model)) or 0
            for model in (Message, Run, WorkflowUseCasePreset)
        }


async def _generated(
    client: AsyncClient,
    profile_id: str,
    *,
    mode: str = "image",
    revision_id: str | None = None,
    settings: dict[str, Any] | None = None,
    inputs: list[str] | None = None,
) -> dict[str, Any]:
    chat = (await client.post("/api/chats", json={"title": "Generated"})).json()
    body: dict[str, Any] = {
        "text": PROMPT,
        "mode": mode,
        "profile_id": profile_id,
        "settings": settings if settings is not None else {"seed": 1234, "steps": 9},
    }
    if revision_id is not None:
        body["workflow_revision_id"] = revision_id
    if inputs is not None:
        body["input_artifact_ids"] = inputs
    turn = await client.post(f"/api/chats/{chat['id']}/turns", json=body)
    assert turn.status_code == 202, turn.text
    return await _finished(client, turn.json()["run"]["id"])


async def _record(client: AsyncClient, run: dict[str, Any]) -> bytes:
    artifact_id = run["provenance_json"]["outputs"][0]["artifact_id"]
    response = await client.get(
        f"/api/runs/{run['id']}/outputs/{artifact_id}/recipe", params={"prompts": "omit"}
    )
    assert response.status_code == 200, response.text
    return response.content


async def _picture(client: AsyncClient) -> str:
    picture = io.BytesIO()
    Image.new("RGB", (8, 6), (40, 90, 160)).save(picture, format="PNG")
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("still.png", picture.getvalue(), "image/png")}
    )
    assert uploaded.status_code == 201, uploaded.text
    return str(uploaded.json()["id"])


async def test_a_finished_picture_drafts_the_recipe_its_settings_make_and_writes_nothing(
    client: AsyncClient,
) -> None:
    revision_id = await _workflow(client, "text_to_image", OPEN_SCHEMA)
    run = await _generated(
        client,
        _profile(),
        revision_id=revision_id,
        settings={"seed": 1234, "steps": 9, "negative_prompt": "soft focus"},
    )
    record = await _record(client, run)
    before = _counts()

    response = await client.get(_draft_path(run["id"]))

    assert response.status_code == 200, response.text
    draft = response.json()
    assert draft["run_id"] == run["id"] and draft["use_case"] == "image_generation"
    assert draft["name"] == "Neutral workflow"
    # The model and workflow are named beside the settings, not held by them.
    assert draft["profile_id"] == run["profile_id"] and draft["profile_name"] == "Recorded model"
    assert draft["workflow_revision_id"] == revision_id
    assert draft["workflow_name"] == "Neutral workflow"
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        assert draft["workflow_id"] == revision.workflow_id
        assert draft["workflow_version"] == revision.version
        assert draft["workflow_family_id"] == revision.definition.family_id
    settings = draft["settings_json"]
    assert settings["steps"] == 9
    # The words and the seed were this request's own, and a recipe holds neither.
    assert "negative_prompt" not in settings and "seed" not in settings
    assert {
        "setting": "negative_prompt",
        "reason": "recipe-prompt",
        "message": RECIPE_LEFT_OUT_MESSAGES["recipe-prompt"],
    } in draft["left_out"]
    assert {
        "setting": "seed",
        "reason": "recipe-seed",
        "message": RECIPE_LEFT_OUT_MESSAGES["recipe-seed"],
    } in draft["left_out"]
    # Every other setting the generation ran with is in the draft.
    left = {item["setting"] for item in draft["left_out"]}
    assert set(settings) | left == set(run["settings_json"])
    # Nothing was written, and the generation's record is what it was.
    assert _counts() == before
    assert await _record(client, run) == record

    saved = await client.post(
        RECIPES,
        json={"name": draft["name"], "use_case": draft["use_case"], "settings_json": settings},
    )
    assert saved.status_code == 201, saved.text
    assert saved.json()["settings_json"] == settings and saved.json()["is_default"] is False


async def test_a_setting_the_workflow_does_not_let_a_recipe_set_is_left_out_with_its_reason(
    client: AsyncClient,
) -> None:
    revision_id = await _workflow(client, "text_to_image", OPEN_SCHEMA)
    run = await _generated(client, _profile(), revision_id=revision_id)
    # The workflow now fixes the step count, so a recipe may no longer set it.
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.input_schema_json = {
            "type": "object",
            "properties": {"steps": {"type": "integer", "default": 20, "readOnly": True}},
        }
        session.commit()

    draft = (await client.get(_draft_path(run["id"]))).json()

    assert "steps" not in draft["settings_json"]
    assert {
        "setting": "steps",
        "reason": "workflow-use-case-preset-setting-unavailable",
        "message": RECIPE_LEFT_OUT_MESSAGES["workflow-use-case-preset-setting-unavailable"],
    } in draft["left_out"]


async def test_a_video_drafts_a_recipe_for_the_way_it_was_made(client: AsyncClient) -> None:
    revision_id = await _workflow(client, "text_to_video", OPEN_SCHEMA)
    from_words = await _generated(
        client, _profile("video"), mode="video", revision_id=revision_id, settings={"seed": 1234}
    )
    from_picture = await _generated(
        client,
        _profile("video"),
        mode="video",
        settings={"seed": 1234},
        inputs=[await _picture(client)],
    )
    assert from_picture["operation"] == "image_to_video"

    words = await client.get(_draft_path(from_words["id"]))
    picture = await client.get(_draft_path(from_picture["id"]))

    assert words.status_code == 200, words.text
    assert words.json()["use_case"] == "video_generation"
    assert picture.status_code == 200, picture.text
    assert picture.json()["use_case"] == "video_animate"


async def test_a_draft_is_refused_for_a_generation_it_cannot_be_made_from(
    client: AsyncClient,
) -> None:
    revision_id = await _workflow(client, "text_to_image", OPEN_SCHEMA)
    edited = await _generated(client, _profile(), revision_id=revision_id)
    failed = await _generated(client, _profile(), revision_id=revision_id)
    gone = await _generated(client, _profile(), revision_id=revision_id)
    with SessionLocal() as session:
        edit = session.get(Run, edited["id"])
        unfinished = session.get(Run, failed["id"])
        assert edit is not None and unfinished is not None
        edit.operation = "image_to_image"
        unfinished.status = "failed"
        session.commit()

    takes_none = await _generated(client, _profile(), revision_id=revision_id)
    other_engine = await _generated(client, _profile(), revision_id=revision_id)
    with SessionLocal() as session:
        profile = session.get(ModelProfile, other_engine["profile_id"])
        assert profile is not None
        profile.engine = "comfyui"
        session.commit()
    not_configured = await client.get(_draft_path(other_engine["id"]))
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        # The workflow now says it makes videos, so no picture recipe fits it.
        definition.operation = "text_to_video"
        session.commit()
    no_recipe = await client.get(_draft_path(takes_none["id"]))
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.definition.operation = "text_to_image"
        session.commit()

    missing = await client.get(_draft_path("run_missing"))
    from_an_edit = await client.get(_draft_path(edited["id"]))
    from_a_failure = await client.get(_draft_path(failed["id"]))
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.definition.current_revision_id = None
        session.delete(revision)
        session.commit()
    without_its_workflow = await client.get(_draft_path(gone["id"]))

    assert missing.status_code == 404
    assert missing.json()["code"] == "output-recipe-run-not-found"
    assert from_an_edit.status_code == 422
    assert from_an_edit.json()["code"] == "output-recipe-draft-unsupported"
    assert from_a_failure.status_code == 409
    assert from_a_failure.json()["code"] == "output-recipe-draft-unfinished"
    assert without_its_workflow.status_code == 409
    assert without_its_workflow.json()["code"] == "output-recipe-draft-unavailable"
    assert no_recipe.status_code == 409
    assert no_recipe.json()["code"] == "output-recipe-draft-unavailable"
    # Made with an engine that is not the one running now.
    assert not_configured.status_code == 409
    assert not_configured.json()["code"] == "engine-not-configured"


async def test_how_many_results_a_request_made_is_left_to_each_request(
    client: AsyncClient,
) -> None:
    revision_id = await _workflow(client, "text_to_image", OPEN_SCHEMA)
    run = await _generated(
        client, _profile(), revision_id=revision_id, settings={"seed": 1234, "batch_size": 2}
    )
    # Each of the two pictures was its own run, so each records a batch of one.
    assert run["settings_json"]["batch_size"] == 1

    draft = (await client.get(_draft_path(run["id"]))).json()

    assert "batch_size" not in draft["settings_json"]
    assert _left_out("batch_size", "recipe-output-count") in draft["left_out"]


async def test_every_seed_the_workflow_binds_is_left_out(client: AsyncClient) -> None:
    seeds = ("workflow_seed_aaaaaaaaaaaa", "workflow_seed_bbbbbbbbbbbb")
    revision_id = _revision(
        "Two samplers",
        api_graph_json={
            "1": {"class_type": "KSampler", "inputs": {"seed": "${" + seeds[0] + "}"}},
            "2": {"class_type": "KSampler", "inputs": {"noise_seed": "${" + seeds[1] + "}"}},
        },
        input_schema_json={
            "type": "object",
            "properties": {
                **{name: {"type": "integer", "default": 7} for name in seeds},
                "steps": {"type": "integer", "default": 20},
            },
            GRAPH_SETTINGS_SCHEMA_KEY: {
                "version": 1,
                "bindings": [
                    {"parameter": seeds[0], "node_id": "1", "input_name": "seed"},
                    {"parameter": seeds[1], "node_id": "2", "input_name": "noise_seed"},
                ],
            },
        },
    )
    run = await _generated(
        client,
        _profile(),
        revision_id=revision_id,
        settings={seeds[0]: 5, seeds[1]: 6, "steps": 9},
    )
    assert {seeds[0]: 5, seeds[1]: 6}.items() <= run["settings_json"].items()

    draft = (await client.get(_draft_path(run["id"]))).json()

    assert draft["settings_json"]["steps"] == 9
    for name in seeds:
        assert name not in draft["settings_json"]
        assert _left_out(name, "recipe-seed") in draft["left_out"]


async def test_loras_the_words_or_a_saved_prompt_chose_are_left_out(
    client: AsyncClient,
) -> None:
    revision_id = await _workflow(client, "text_to_image", OPEN_SCHEMA)
    run = await _generated(client, _profile(), revision_id=revision_id)
    stack = [{"asset_id": "asset_neutral", "model_strength": 1.0, "clip_strength": 1.0}]
    # Each way a run records how its stack was chosen.
    ways: dict[str, dict[str, Any]] = {
        "automatic": {"auxiliary_assets": {"selection": {"mode": "automatic", "selected": []}}},
        "saved prompt": {
            "auxiliary_assets": {"selection": {"mode": "explicit"}},
            "prompt_source": {"kind": "prompt_template"},
        },
        "chosen": {"auxiliary_assets": {"selection": {"mode": "explicit"}}},
    }
    reasons: dict[str, set[str]] = {}
    for name, provenance in ways.items():
        with SessionLocal() as session:
            stored = session.get(Run, run["id"])
            assert stored is not None
            stored.settings_json = {**stored.settings_json, "loras": stack}
            kept = {
                key: value
                for key, value in stored.provenance_json.items()
                if key not in {"auxiliary_assets", "prompt_source"}
            }
            stored.provenance_json = {**kept, **provenance}
            session.commit()
        draft = (await client.get(_draft_path(run["id"]))).json()
        assert "loras" not in draft["settings_json"], name
        reasons[name] = {item["reason"] for item in draft["left_out"] if item["setting"] == "loras"}

    assert reasons["automatic"] == {"recipe-matched-loras"}
    assert reasons["saved prompt"] == {"recipe-matched-loras"}
    # A stack chosen for the request goes to the recipe check like any setting;
    # this workflow takes no added LoRAs, so that check is what leaves it out.
    assert reasons["chosen"] and "recipe-matched-loras" not in reasons["chosen"]
