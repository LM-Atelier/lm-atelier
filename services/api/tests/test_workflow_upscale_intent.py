"""Keep an enlargement request distinct from its optional numeric factor."""

import hashlib
import json

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from pydantic import ValidationError
from sqlalchemy import select
from test_accepted_turn_context import _accept_context
from test_media_regeneration_seeds import _wait_for_run
from test_workflow_recipe_request_cases import _picture, _workflow

from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import Run, RunContextSnapshot, WorkflowPreference, WorkflowRevision
from local_lm.schemas import TurnRequest
from local_lm.workflow_use_case_execution import workflow_use_case_inputs
from local_lm.workflow_use_cases_v1 import classify_workflow_use_case


def test_enlargement_intent_does_not_require_a_numeric_setting() -> None:
    request = TurnRequest.model_validate({"text": "A blue square", "upscale": True})

    facts = workflow_use_case_inputs(Operation.IMAGE_TO_IMAGE, request, source_present=True)

    assert classify_workflow_use_case(facts).use_case.value == "image_upscale"
    assert request.settings == {}


def test_a_numeric_setting_alone_does_not_change_an_edits_intent() -> None:
    request = TurnRequest(text="A blue square", settings={"upscale_factor": 2})

    facts = workflow_use_case_inputs(Operation.IMAGE_TO_IMAGE, request, source_present=True)

    assert classify_workflow_use_case(facts).use_case.value == "image_edit"


def test_an_image_role_can_override_the_enlargement_intent() -> None:
    request = TurnRequest.model_validate(
        {"text": "A blue square", "upscale": True, "role_overrides": {"image": {"upscale": False}}}
    )

    assert request.for_role("image").upscale is False
    assert request.upscale is True


def test_enlargement_intent_rejects_text_instead_of_a_boolean() -> None:
    with pytest.raises(ValidationError):
        TurnRequest.model_validate({"text": "A blue square", "upscale": "yes"})


async def test_a_snapshot_without_enlargement_intent_keeps_its_original_digest(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        run_id, _, _ = await _accept_context(app, client)
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            row = session.get(RunContextSnapshot, run_id)
            assert run is not None and row is not None
            payload = {key: value for key, value in row.payload_json.items() if key != "upscale"}
            digest = hashlib.sha256(
                json.dumps(
                    payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ).encode()
            ).hexdigest()
            row.payload_json = payload
            row.sha256 = digest
            run.provenance_json = {**run.provenance_json, "accepted_context_sha256": digest}
            session.commit()
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            row = session.get(RunContextSnapshot, run_id)
            assert run is not None and row is not None
            snapshot = accepted_context(session, run)
            assert snapshot is not None and snapshot.upscale is False
            assert row.sha256 == digest
            assert "upscale" not in row.payload_json


@pytest.mark.parametrize("choice", ["family", "revision"])
@pytest.mark.parametrize("recipe_mode", ["unconfigured", "automatic"])
async def test_enlargement_rejects_an_ordinary_edit_without_a_recipe(
    app: FastAPI, client: AsyncClient, choice: str, recipe_mode: str
) -> None:
    family_id, revision_id = _workflow("image_edit", "image_to_image", "image")
    chat = (await client.post("/api/chats", json={"title": "Enlargement selection"})).json()
    if recipe_mode == "automatic":
        selected = await client.put(
            f"/api/chats/{chat['id']}/workflow-use-case-presets/image_upscale",
            json={"mode": "automatic"},
        )
        assert selected.status_code == 200, selected.text
    selection = (
        {"mode": "family", "workflow_family_id": family_id}
        if choice == "family"
        else {"mode": "revision", "workflow_revision_id": revision_id}
    )
    picture_id = await _picture(client)
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "A blue square",
                "mode": "image",
                "upscale": True,
                "input_artifact_ids": [picture_id],
                "workflow_selection": {"selector_capability": "image", **selection},
            },
        )
    if choice == "revision":
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "upscale-selection-unavailable"
    else:
        assert response.status_code == 422, response.text
        assert "workflow-use-case-upscale-unsupported" in response.text


async def test_automatic_enlargement_skips_the_default_ordinary_editor(
    app: FastAPI, client: AsyncClient
) -> None:
    ordinary_family, _ = _workflow("image_edit", "image_to_image", "image")
    _, enlargement_revision = _workflow("image_upscale", "image_to_image", "image")
    with SessionLocal() as session:
        for existing_preference in session.scalars(select(WorkflowPreference)).all():
            existing_preference.is_default = False
        session.flush()
        preference = session.scalar(
            select(WorkflowPreference).where(
                WorkflowPreference.workflow_family_id == ordinary_family
            )
        )
        assert preference is not None
        preference.is_default = True
        session.commit()
    chat = (await client.post("/api/chats", json={"title": "Automatic enlargement"})).json()
    picture_id = await _picture(client)
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "A blue square",
                "mode": "image",
                "upscale": True,
                "input_artifact_ids": [picture_id],
                "workflow_selection": {"selector_capability": "image", "mode": "automatic"},
            },
        )
        assert response.status_code == 202, response.text
        assert response.json()["run"]["workflow_revision_id"] == enlargement_revision


@pytest.mark.parametrize("action", ["edit", "regenerate"])
async def test_a_fixed_enlargement_keeps_its_recipe_when_repeated(
    client: AsyncClient, action: str
) -> None:
    family_id, revision_id = _workflow("image_upscale", "image_to_image", "image")
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.input_schema_json = {
            "type": "object",
            "properties": {
                "recipe_quality": {"type": "number", "default": 0.25},
                "upscale_factor": {
                    "type": "number",
                    "readOnly": True,
                    "x-lm-atelier-kind": "upscale",
                },
            },
        }
        revision.api_graph_json = {
            **revision.api_graph_json,
            "scale": {
                "class_type": "ImageScaleBy",
                "inputs": {"image": ["source", 0], "scale_by": 4},
            },
        }
        session.commit()
    chat = (await client.post("/api/chats", json={"title": "Fixed enlargement"})).json()
    recipe = await client.post(
        "/api/workflow-use-case-presets",
        json={
            "name": "Fixed enlargement recipe",
            "use_case": "image_upscale",
            "settings_json": {"recipe_quality": 0.75},
            "is_default": True,
        },
    )
    assert recipe.status_code == 201, recipe.text
    selected = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": family_id},
    )
    assert selected.status_code == 200, selected.text
    picture_id = await _picture(client)
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "A blue square",
            "mode": "image",
            "upscale": True,
            "input_artifact_ids": [picture_id],
        },
    )
    assert accepted.status_code == 202, accepted.text
    finished = await _wait_for_run(client, accepted.json()["run"]["id"])
    assert finished["status"] == "complete"
    with SessionLocal() as session:
        original = session.get(Run, accepted.json()["run"]["id"])
        assert original is not None
        snapshot = accepted_context(session, original)
        assert snapshot is not None and snapshot.upscale is True
        assert "upscale_factor" not in snapshot.settings
    changed = await client.put(
        f"/api/chats/{chat['id']}/workflow-use-case-presets/image_upscale",
        json={"mode": "automatic"},
    )
    assert changed.status_code == 200, changed.text
    if action == "edit":
        edited = await client.post(
            f"/api/messages/{accepted.json()['user_message']['id']}/edits",
            json={"text": "A green square", "idempotency_key": "keep-fixed-enlargement"},
        )
    else:
        edited = await client.post(
            f"/api/messages/{accepted.json()['assistant_message']['id']}/regenerate",
            json={"settings": {}},
        )
    assert edited.status_code == 202, edited.text
    with SessionLocal() as session:
        repeated = session.get(Run, edited.json()["run"]["id"])
        assert repeated is not None
        snapshot = accepted_context(session, repeated)
        assert snapshot is not None and snapshot.upscale is True
        assert snapshot.workflow_revision_id == revision_id
        assert snapshot.settings["recipe_quality"] == 0.75
        assert "upscale_factor" not in snapshot.settings
        assert snapshot.workflow_use_case_preset is not None
        assert snapshot.workflow_use_case_preset.preset_id == recipe.json()["id"]
