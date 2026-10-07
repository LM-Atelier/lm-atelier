"""Apply scoped recipes to real admission and preserve their accepted values."""

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_turn_workflow_choices import _image_workflow, _resource_batch

from local_lm.db import SessionLocal
from local_lm.models import Run, WorkflowRevision, WorkflowUseCasePreset, WorkStep
from local_lm.prompt_expansion_use import read_prompt_batch_queue_selection
from local_lm.schemas import TurnRequest


def _recipe(settings: dict[str, Any]) -> str:
    with SessionLocal() as session:
        preset = WorkflowUseCasePreset(
            name="Example image recipe",
            use_case="image_generation",
            is_default=True,
            settings_json=settings,
        )
        session.add(preset)
        session.commit()
        return preset.id


@pytest.mark.parametrize("override", [None, 19])
async def test_turn_applies_recipe_below_explicit_settings_and_captures_its_layer(
    app: FastAPI,
    client: AsyncClient,
    override: int | None,
) -> None:
    recipe_id = _recipe({"steps": 7})
    chat = (await client.post("/api/chats", json={"title": "Recipe layers"})).json()
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "A blue square",
                "mode": "image",
                "settings": {} if override is None else {"steps": override},
            },
        )
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            run = session.get(Run, response.json()["run"]["id"])
            assert run is not None
            assert run.settings_json["steps"] == (7 if override is None else override)
            receipt = run.provenance_json["workflow_use_case_preset"]
            assert receipt["preset_id"] == recipe_id
            assert receipt["settings_json"] == {"steps": 7}
            assert receipt["workflow_revision_id"] == run.workflow_revision_id


@pytest.mark.parametrize("configured", [False, True])
async def test_edits_keep_original_recipe_after_workspace_default_changes(
    app: FastAPI,
    client: AsyncClient,
    configured: bool,
) -> None:
    recipe_id = _recipe({"steps": 7}) if configured else None
    chat = (await client.post("/api/chats", json={"title": "Repeated recipe"})).json()
    async with app.state.services.scheduler.lease("primary"):
        original = await client.post(
            f"/api/chats/{chat['id']}/turns", json={"text": "A blue square", "mode": "image"}
        )
        assert original.status_code == 202, original.text
        with SessionLocal() as session:
            prior = session.get(Run, original.json()["run"]["id"])
            assert prior is not None
            original_steps = prior.settings_json["steps"]
            if recipe_id:
                recipe = session.get(WorkflowUseCasePreset, recipe_id)
                assert recipe is not None
                recipe.settings_json = {"steps": 29}
                session.commit()
        if not configured:
            _recipe({"steps": 29})
        edited = await client.post(
            f"/api/messages/{original.json()['user_message']['id']}/edits",
            json={"text": "A green square", "idempotency_key": "repeat-recipe"},
        )
        assert edited.status_code == 202, edited.text
        with SessionLocal() as session:
            run = session.get(Run, edited.json()["run"]["id"])
            assert run is not None and run.settings_json["steps"] == original_steps
            receipt = run.provenance_json.get("workflow_use_case_preset")
            if configured:
                assert receipt is not None and receipt["settings_json"] == {"steps": 7}
            else:
                assert receipt is None


async def test_ordered_steps_apply_only_their_resolved_use_case_recipe(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    recipe_id = _recipe({"steps": 7})
    chat = (await client.post("/api/chats", json={"title": "Ordered recipes"})).json()
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": (
                    "Write a short story about a paper boat, then create an image based on it, "
                    "then animate the image into a video, then summarize the video"
                ),
                "mode": "auto",
                "confirm_media": True,
            },
        )
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            runs = list(
                session.scalars(
                    select(Run)
                    .join(WorkStep, Run.work_step_id == WorkStep.id)
                    .where(Run.work_plan_id == response.json()["run"]["work_plan_id"])
                    .order_by(WorkStep.ordinal)
                )
            )
            assert len(runs) == 4
            image = runs[1]
            assert image.operation == "text_to_image" and image.settings_json["steps"] == 7
            assert image.provenance_json["workflow_use_case_preset"]["preset_id"] == recipe_id
            assert all(
                "workflow_use_case_preset" not in run.provenance_json
                for run in (runs[0], runs[2], runs[3])
            )


async def test_incompatible_recipe_refuses_before_any_run_is_created(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    _recipe({"unknown_control": 7})
    chat = (await client.post("/api/chats", json={"title": "Unsupported recipe"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "A blue square", "mode": "image"}
    )
    assert response.status_code in {409, 422}, response.text
    with SessionLocal() as session:
        assert session.scalar(select(Run.id).where(Run.chat_id == chat["id"])) is None


@pytest.mark.parametrize("incompatible_second", [False, True])
async def test_reviewed_batch_checks_each_exact_revision_and_records_each_recipe(
    app: FastAPI,
    client: AsyncClient,
    incompatible_second: bool,
) -> None:
    _recipe({"steps": 7})
    chat = (await client.post("/api/chats", json={"title": "Batch recipes"})).json()
    with SessionLocal() as session:
        _, first_id = _image_workflow(session, "First recipe workflow")
        _, second_id = _image_workflow(session, "Second recipe workflow")
        if incompatible_second:
            second = session.get(WorkflowRevision, second_id)
            assert second is not None
            second.input_schema_json = {
                "properties": {
                    "steps": {
                        "type": "integer",
                        "default": 2,
                        "minimum": 1,
                        "maximum": 5,
                    }
                }
            }
        session.commit()
    batch = await _resource_batch(client, chat["id"], [first_id, second_id])
    orchestrator = app.state.services.orchestrator
    async with app.state.services.scheduler.lease("primary"), orchestrator.chat_guard(chat["id"]):
        with SessionLocal() as session:
            selection = read_prompt_batch_queue_selection(
                session,
                chat["id"],
                batch["id"],
                batch["plan_version"],
                batch["plan_sha256"],
                expected_engine="mock",
            )
            request = TurnRequest(
                text=batch["items"][0]["reviewed_prompt"],
                mode="image",
                output_count=2,
                idempotency_key="recipe-batch",
            )
            if incompatible_second:
                with pytest.raises(ValueError):
                    await orchestrator._create_new_turn(
                        session,
                        chat["id"],
                        request,
                        source_action="prompt_library",
                        prompt_batch_selection=selection,
                    )
                assert session.scalar(select(Run.id).where(Run.chat_id == chat["id"])) is None
                return
            accepted = await orchestrator._create_new_turn(
                session,
                chat["id"],
                request,
                source_action="prompt_library",
                prompt_batch_selection=selection,
            )
            runs = list(
                session.scalars(select(Run).where(Run.work_plan_id == accepted.run.work_plan_id))
            )
            assert {run.workflow_revision_id for run in runs} == {first_id, second_id}
            for run in runs:
                assert run.settings_json["steps"] == 7
                receipt = run.provenance_json["workflow_use_case_preset"]
                assert receipt["workflow_revision_id"] == run.workflow_revision_id
                assert receipt["settings_json"] == {"steps": 7}
