"""A recipe drafted from one choice of a comparison, to review before saving it."""

from __future__ import annotations

from typing import Any

from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_generation_experiment_preflight import _arm, _counts, _profile, _request, _revision
from test_generation_experiment_records import CREATE, _accept, _tamper

from local_lm.db import SessionLocal
from local_lm.generation_experiments_v1 import RECIPE_LEFT_OUT_MESSAGES
from local_lm.models import WorkflowRevision, WorkflowUseCasePreset

RECIPES = "/api/workflow-use-case-presets"


def _recipes() -> int:
    with SessionLocal() as session:
        return int(session.scalar(select(func.count()).select_from(WorkflowUseCasePreset)) or 0)


def _draft_path(experiment_id: str, ordinal: int) -> str:
    return f"{CREATE}/{experiment_id}/arms/{ordinal}/recipe-draft"


async def _accepted(
    client: AsyncClient, second_defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    first = _arm("Fewer steps", _profile("Harbor one"), _revision("Harbor one"), steps=8)
    second = _arm(
        "More steps",
        _profile("Harbor two", request_settings_json=second_defaults or {}),
        _revision("Harbor two"),
        steps=20,
    )
    return await _accept(client, _request(first, second))


async def test_a_choice_drafts_the_recipe_its_settings_make_and_writes_nothing(
    client: AsyncClient,
) -> None:
    accepted = await _accepted(client)
    before = (_counts(), _recipes())

    response = await client.get(_draft_path(accepted["id"], 2))

    assert response.status_code == 200, response.text
    draft = response.json()
    chosen = accepted["arms"][1]
    assert draft["experiment_id"] == accepted["id"] and draft["arm_ordinal"] == 2
    assert draft["use_case"] == "image_generation" and draft["name"] == "More steps"
    # The model and workflow are named beside the settings, not held by them.
    assert draft["profile_id"] == chosen["profile_id"] and draft["profile_name"] == "Harbor two"
    assert draft["workflow_revision_id"] == chosen["workflow_revision_id"]
    assert draft["workflow_name"] == "Harbor two" and draft["workflow_version"] == 1
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, chosen["workflow_revision_id"])
        assert revision is not None
        assert draft["workflow_id"] == revision.workflow_id
        family = revision.definition.family_id
        assert family is not None and draft["workflow_family_id"] == family
    settings = draft["settings_json"]
    assert settings["steps"] == 20
    assert (settings["width"], settings["height"]) == (512, 384)
    # The words were the comparison's, and a recipe never holds them.
    assert "negative_prompt" not in settings
    assert {
        "setting": "negative_prompt",
        "reason": "recipe-prompt",
        "message": RECIPE_LEFT_OUT_MESSAGES["recipe-prompt"],
    } in draft["left_out"]
    assert "seed" not in settings
    # Every other setting the choice ran with is in the draft.
    left = {item["setting"] for item in draft["left_out"]}
    assert set(settings) | left == set(chosen["effective_settings"])
    assert (_counts(), _recipes()) == before

    saved = await client.post(
        RECIPES,
        json={"name": draft["name"], "use_case": draft["use_case"], "settings_json": settings},
    )
    assert saved.status_code == 201, saved.text
    assert saved.json()["settings_json"] == settings and saved.json()["is_default"] is False


async def test_a_setting_the_comparison_changed_for_its_own_pictures_is_left_out(
    client: AsyncClient,
) -> None:
    # The model makes four pictures at a time; the comparison made one per choice.
    accepted = await _accepted(client, {"batch_size": 4})

    draft = (await client.get(_draft_path(accepted["id"], 2))).json()

    assert "batch_size" not in draft["settings_json"]
    assert {
        "setting": "batch_size",
        "reason": "comparison-adapted",
        "message": RECIPE_LEFT_OUT_MESSAGES["comparison-adapted"],
    } in draft["left_out"]
    # The other choice asked for nothing the comparison changed.
    other = (await client.get(_draft_path(accepted["id"], 1))).json()
    assert all(item["setting"] != "batch_size" for item in other["left_out"])


async def test_a_setting_a_recipe_here_cannot_hold_is_left_out_with_its_reason(
    client: AsyncClient,
) -> None:
    accepted = await _accepted(client)
    # The workflow now fixes the step count, so a recipe may no longer set it.
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, accepted["arms"][1]["workflow_revision_id"])
        assert revision is not None
        revision.input_schema_json = {
            "type": "object",
            "properties": {"steps": {"type": "integer", "default": 20, "readOnly": True}},
        }
        session.commit()

    draft = (await client.get(_draft_path(accepted["id"], 2))).json()

    assert "steps" not in draft["settings_json"]
    assert {
        "setting": "steps",
        "reason": "workflow-use-case-preset-setting-unavailable",
        "message": RECIPE_LEFT_OUT_MESSAGES["workflow-use-case-preset-setting-unavailable"],
    } in draft["left_out"]
    assert (draft["settings_json"]["width"], draft["settings_json"]["height"]) == (512, 384)


async def test_a_draft_is_refused_for_a_choice_it_cannot_be_made_from(
    client: AsyncClient,
) -> None:
    accepted = await _accepted(client)

    missing_comparison = await client.get(_draft_path("gexp_missing", 1))
    missing_choice = await client.get(_draft_path(accepted["id"], 3))
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, accepted["arms"][0]["workflow_revision_id"])
        assert revision is not None
        revision.definition.current_revision_id = None
        session.delete(revision)
        session.commit()
    gone_workflow = await client.get(_draft_path(accepted["id"], 1))

    assert missing_comparison.status_code == 404
    assert missing_comparison.json()["code"] == "generation-experiment-not-found"
    assert missing_choice.status_code == 404
    assert missing_choice.json()["code"] == "generation-experiment-arm-not-found"
    assert gone_workflow.status_code == 409
    assert gone_workflow.json()["code"] == "generation-experiment-recipe-unavailable"
    # The other choice's workflow is still there, so its draft still comes.
    assert (await client.get(_draft_path(accepted["id"], 2))).status_code == 200


async def test_a_record_that_changed_after_it_was_accepted_drafts_nothing(
    client: AsyncClient,
) -> None:
    accepted = await _accepted(client)
    _tamper(accepted["id"], "effective")

    response = await client.get(_draft_path(accepted["id"], 1))

    assert response.status_code == 409
    assert response.json()["code"] == "generation-experiment-record-invalid"
