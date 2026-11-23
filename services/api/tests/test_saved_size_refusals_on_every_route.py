"""Every route that resolves a turn's settings answers a saved size it cannot make as such.

The refusal is raised where a turn's saved layers are read, so an Enhance
preview, a remix, a comparison and a Prompt Library batch meet it too. Each
says that the saved size is the problem, as sending the turn does, rather than
blaming the workflow or the settings in general.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_generation_experiment_preflight import PREFLIGHT, _arm, _profile, _request
from test_generation_experiment_preflight import _revision as _plain_revision
from test_picture_remix import SETTINGS, _png, _revision, _uploaded
from test_upscale_preview_api import _selection
from workflow_fixtures import seed_workflow_trust

from local_lm.db import SessionLocal
from local_lm.models import ModelProfile, WorkflowRevision
from local_lm.settings_registry import SavedSizeUnsupported, builtin_settings_for_role

CAP = {"type": "integer", "default": 1024, "minimum": 256, "maximum": 2048, "multipleOf": 64}
TOO_WIDE = {"width": 2176, "height": 768}
SAID = "The width of 2176 px saved for this conversation is not one this workflow makes"


async def _save_on_chat(client: AsyncClient, chat_id: str, size: dict[str, Any]) -> None:
    saved = await client.patch(
        f"/api/chats/{chat_id}", json={"generation_settings_json": {"image": size}}
    )
    assert saved.status_code == 200, saved.text


def _said(response: Any) -> None:
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "turn-size-unsupported"
    assert SAID in response.json()["detail"]


async def test_a_prompt_library_batch_is_refused_for_the_saved_size(
    app: FastAPI, client: AsyncClient
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Capped batch workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "CappedBatchWorkflow"}},
            "input_schema": {"type": "object", "properties": {"width": CAP, "height": CAP}},
        },
    )
    assert created.status_code == 201, created.text
    revision_id = created.json()["current_revision_id"]
    seed_workflow_trust(revision_id)
    template = await client.post(
        "/api/prompt-templates",
        json={
            "idempotency_key": "saved-size-template",
            "name": "Saved size",
            "description": "One subject slot",
            "contract": {
                "schema_version": 1,
                "operation": "text_to_image",
                "body": "A {{subject}}.",
                "slots": [{"name": "subject", "mode": "input", "variation_scope": "item"}],
                "resource_policy": {
                    "mode": "fixed",
                    "workflow_revision_id": revision_id,
                    "lora_policy": {"mode": "none"},
                },
            },
        },
    )
    assert template.status_code == 201, template.text
    revision = template.json()["revision"]
    chat = (await client.post("/api/chats", json={"title": "Saved size"})).json()
    await _save_on_chat(client, chat["id"], TOO_WIDE)
    created_batch = await client.post(
        f"/api/chats/{chat['id']}/prompt-batches",
        json={
            "idempotency_key": "saved-size-preview",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": 1,
            "selection_seed": 3,
            "inputs": {"subject": ["gray square"]},
        },
    )
    assert created_batch.status_code in (200, 201), created_batch.text
    batch = created_batch.json()

    async with app.state.services.scheduler.lease("primary"):
        queued = await client.post(
            f"/api/prompt-batches/{batch['id']}/queue",
            json={
                "idempotency_key": "saved-size-queue",
                "expected_plan_version": batch["plan_version"],
                "expected_plan_sha256": batch["plan_sha256"],
            },
        )

    _said(queued)


async def test_a_remix_says_the_saved_size_is_the_problem(client: AsyncClient) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    revision_id = _revision(
        "Capped workflow",
        input_schema={"type": "object", "properties": {"width": dict(CAP), "height": dict(CAP)}},
    )
    with SessionLocal() as session:
        profile = ModelProfile(name="Neutral model", role="image", engine="mock")
        session.add(profile)
        session.commit()
        profile_id = profile.id

    async def remix() -> Any:
        return await client.post(
            f"/api/artifacts/{artifact_id}/remix-preview",
            json={"workflow_revision_id": revision_id, "profile_id": profile_id, "apply": []},
        )

    clean = await remix()
    saved = await client.patch(f"/api/profiles/{profile_id}", json={"request_settings": TOO_WIDE})
    assert saved.status_code == 200, saved.text
    past = await remix()

    assert clean.status_code == 200 and clean.json()["ready"] is True, clean.text
    assert past.status_code == 200, past.text
    assert past.json()["ready"] is False
    assert [item["code"] for item in past.json()["refusals"]] == ["remix-size-unsupported"]


async def test_an_enhance_preview_says_the_saved_size_is_the_problem(
    app: FastAPI, client: AsyncClient
) -> None:
    chat_id, revision_id, payload = await _selection(client, adjustable=False)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        schema = copy.deepcopy(revision.input_schema_json)
        schema["properties"]["width"] = dict(CAP)
        revision.input_schema_json = schema
        session.commit()
    url = f"/api/chats/{chat_id}/upscale/preview"
    await _save_on_chat(client, chat_id, {"width": 2048})
    inside = await client.post(url, json=payload)
    await _save_on_chat(client, chat_id, {"width": 2176})

    past = await client.post(url, json=payload)

    assert inside.status_code == 200, inside.text
    _said(past)


async def test_a_comparison_names_the_saved_size_as_the_setting_it_cannot_run(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a choice that keeps a picture's own size can meet it, so the resolver is made to."""

    first = _arm("First", _profile("Ready model"), _plain_revision("Ready workflow"))
    refused_profile = _profile("Saved size model")
    second = _arm("Second", refused_profile, _plain_revision("Second workflow"))
    orchestrator = app.state.services.orchestrator
    resolve = orchestrator.resolve_turn_setting_layers
    width = next(field for field in builtin_settings_for_role("image") if field.key == "width")

    def saved_size_refused(
        session: Any, chat: Any, operation: Any, profile: Any, *rest: Any, **named: Any
    ) -> Any:
        if profile is not None and profile.id == refused_profile:
            raise SavedSizeUnsupported("width", 8192, width, "in the model profile")
        return resolve(session, chat, operation, profile, *rest, **named)

    monkeypatch.setattr(orchestrator, "resolve_turn_setting_layers", saved_size_refused)

    response = await client.post(PREFLIGHT, json=_request(first, second))

    assert response.status_code == 200, response.text
    refusals = response.json()["refusals"]
    assert [(item["code"], item["arm_ordinal"], item["setting"]) for item in refusals] == [
        ("arm-setting-invalid", 2, "width")
    ]
