"""An edit kept as an Image Studio recipe: its own words, and nothing one request brings."""

from __future__ import annotations

from typing import Any

from httpx2 import AsyncClient
from sqlalchemy import select
from test_generation_experiment_preflight import _revision
from test_output_recipe_replay import _finished

from local_lm.db import SessionLocal
from local_lm.models import Run
from local_lm.workflow_graph_settings_v1 import GRAPH_SETTINGS_SCHEMA_KEY


async def _turn(client: AsyncClient, chat_id: str, text: str, mode: str) -> dict[str, Any]:
    response = await client.post(f"/api/chats/{chat_id}/turns", json={"text": text, "mode": mode})
    assert response.status_code == 202, response.text
    return await _finished(client, response.json()["run"]["id"])


async def test_a_follow_up_edit_keeps_the_words_typed_for_it(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Apples"})).json()
    picture = await _turn(client, chat["id"], "Make an image of a red apple", "image")
    edit = await _turn(client, chat["id"], "Make it green", "auto")
    # The run carries the earlier picture's prompt too, which a recipe must not.
    assert edit["operation"] == "image_to_image"
    assert edit["standalone_prompt"] == (
        "Make an image of a red apple. Follow-up instruction: Make it green"
    )

    draft = await client.get(f"/api/runs/{edit['id']}/edit-recipe-draft")
    not_an_edit = await client.get(f"/api/runs/{picture['id']}/edit-recipe-draft")
    missing = await client.get("/api/runs/run_missing/edit-recipe-draft")

    assert draft.status_code == 200, draft.text
    assert draft.json() == {"run_id": edit["id"], "instruction": "Make it green"}
    assert not_an_edit.status_code == 422
    assert not_an_edit.json()["code"] == "edit-recipe-draft-unsupported"
    assert missing.status_code == 404
    assert missing.json()["code"] == "output-recipe-run-not-found"


async def test_an_edit_recipe_leaves_out_bound_seeds_and_loras_matched_to_its_words(
    client: AsyncClient,
) -> None:
    seed = "workflow_seed_aaaaaaaaaaaa"
    revision_id = _revision(
        "Two samplers",
        operation="image_to_image",
        api_graph_json={"1": {"class_type": "KSampler", "inputs": {"seed": "${" + seed + "}"}}},
        input_schema_json={
            "type": "object",
            "properties": {seed: {"type": "integer", "default": 7}},
            GRAPH_SETTINGS_SCHEMA_KEY: {
                "version": 1,
                "bindings": [{"parameter": seed, "node_id": "1", "input_name": "seed"}],
            },
        },
    )
    chat = (await client.post("/api/chats", json={"title": "Edits"})).json()
    await _turn(client, chat["id"], "Make an image of a red apple", "image")
    edit = await _turn(client, chat["id"], "Make it green", "auto")
    stack = [{"asset_id": "asset_neutral", "model_strength": 1.0, "clip_strength": 1.0}]
    # As a run records an edit on that workflow whose LoRAs were matched to its words.
    with SessionLocal() as session:
        run = session.get(Run, edit["id"])
        assert run is not None
        run.provenance_json = {
            **run.provenance_json,
            "workflow": {**run.provenance_json.get("workflow", {}), "revision_id": revision_id},
            "resolved_settings": {"steps": 9, seed: 5, "loras": stack},
            "auxiliary_assets": {"selection": {"mode": "automatic", "selected": []}},
        }
        session.commit()

    saved = await client.post(
        "/api/edit-templates",
        json={"name": "Greener", "instruction": "Make it green", "from_run_id": edit["id"]},
    )

    assert saved.status_code == 201, saved.text
    assert saved.json()["settings_json"] == {"steps": 9}
    assert saved.json()["workflow_revision_id"] == revision_id


async def test_one_step_of_a_longer_request_keeps_that_steps_own_words(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Two steps"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image of a red apple, then make it a watercolor picture",
            "mode": "auto",
        },
    )
    assert response.status_code == 202, response.text
    first = await _finished(client, response.json()["run"]["id"])
    with SessionLocal() as session:
        runs = session.scalars(
            select(Run).where(Run.work_plan_id == first["work_plan_id"]).order_by(Run.created_at)
        ).all()
        edits = [run for run in runs if run.operation == "image_to_image"]
        assert len(runs) == 2 and len(edits) == 1, [run.operation for run in runs]
        step_words = edits[0].standalone_prompt
        edit_id = edits[0].id
    await _finished(client, edit_id)

    draft = (await client.get(f"/api/runs/{edit_id}/edit-recipe-draft")).json()

    assert draft["instruction"] == step_words
    assert "red apple" not in draft["instruction"]


async def test_words_taken_back_from_the_chat_are_never_offered(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Apples"})).json()
    await _turn(client, chat["id"], "Make an image of a red apple", "image")
    edit = await _turn(client, chat["id"], "Make it green", "auto")
    detail = (await client.get(f"/api/chats/{chat['id']}")).json()
    first_message = next(
        message["id"]
        for message in detail["messages"]
        if message["role"] == "user"
        and any(part.get("text") == "Make an image of a red apple" for part in message["parts"])
    )
    preview = await client.get(f"/api/messages/{first_message}/removal-impact")
    preview.raise_for_status()
    removed = await client.post(
        f"/api/messages/{first_message}/remove-content",
        json={
            "expected_message_id": first_message,
            "expected_revision_id": preview.json()["message_revision_id"],
            "operation_key": "remove-the-first-request",
        },
    )
    removed.raise_for_status()

    draft = (await client.get(f"/api/runs/{edit['id']}/edit-recipe-draft")).json()

    # The edit's prompt may still carry the removed words, so none are offered.
    assert draft["instruction"] == ""
