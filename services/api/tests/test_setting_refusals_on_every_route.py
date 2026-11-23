"""Every route that refuses a render for a setting names the setting beside its own code.

A turn, a Prompt Library batch, a regenerate, an edit, a branch, a remix, an
Enhance preview and a comparison each answer with the code and detail they
always did, and add the same `errors` items, so a client learns which setting
was refused and why without telling the routes apart. A setting carried over
from an earlier turn, or saved on a layer, says where it came from.
"""

from __future__ import annotations

import copy
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from test_generation_experiment_preflight import PREFLIGHT, _arm, _profile, _request, _revision
from test_picture_remix import SETTINGS, _png, _uploaded
from test_upscale_preview_api import _selection
from workflow_fixtures import seed_workflow_trust

from local_lm.db import SessionLocal
from local_lm.models import (
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.settings_registry import SavedSizeUnsupported, builtin_settings_for_role

CAP = {"type": "integer", "default": 1024, "minimum": 256, "maximum": 2048, "multipleOf": 64}
TOO_WIDE = {"width": 2176, "height": 768}
DETAIL = {"type": "integer", "default": 1, "minimum": 0, "maximum": 5}
TEXT = "Create an image of a ceramic cup"
#: What a built-in picture setting past its maximum is named as, wherever it is refused.
STEPS_PAST = [{"code": "setting-above-maximum", "field": "steps", "limit": 200}]
#: What a size saved for the conversation past the workflow's maximum is named as.
WIDTH_SAVED = [{"code": "setting-above-maximum", "field": "width", "limit": 2048, "source": "chat"}]
CARRIED_OVER = [{"code": "setting-unknown", "field": "detail", "source": "inherited"}]


def _errors(response: Any, code: str = "turn-invalid", status: int = 422) -> list[dict[str, Any]]:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["code"] == code, body
    items: list[dict[str, Any]] = body["errors"]
    for item in items:
        assert isinstance(item.pop("message"), str), item
    return items


async def _chat(client: AsyncClient) -> str:
    chat = await client.post("/api/chats", json={"title": "Refused settings"})
    assert chat.status_code == 201, chat.text
    chat_id: str = chat.json()["id"]
    return chat_id


def _family(name: str, input_schema: dict[str, Any]) -> str:
    """A picture workflow family a chat can choose, with one trusted revision."""

    with SessionLocal() as session:
        family = WorkflowFamily(name=name, use_case=name)
        definition = WorkflowDefinition(
            family=family, name=name, operation="text_to_image", variant_key="create"
        )
        revision = WorkflowRevision(
            definition=definition,
            version=1,
            engine="mock",
            trusted=True,
            api_graph_json={},
            input_schema_json=input_schema,
            dependencies_json={},
        )
        preference = WorkflowPreference(family=family, selector_capability="image")
        session.add_all([family, definition, revision, preference])
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        return family.id


async def _choose(client: AsyncClient, chat_id: str, family_id: str) -> None:
    chosen = await client.put(
        f"/api/chats/{chat_id}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": family_id},
    )
    assert chosen.status_code == 200, chosen.text


async def _completed(client: AsyncClient, chat_id: str, settings: dict[str, Any]) -> Any:
    """A picture turn, made to the end, so it can be regenerated."""

    response = await client.post(
        f"/api/chats/{chat_id}/turns", json={"text": TEXT, "mode": "image", "settings": settings}
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["run"]["id"]

    async def read() -> dict[str, Any]:
        run = await client.get(f"/api/runs/{run_id}")
        assert run.status_code == 200, run.text
        return cast(dict[str, Any], run.json())

    await wait_for_terminal_status(read, what=f"run {run_id}")
    return response.json()


async def test_a_turn_names_the_setting(app: FastAPI, client: AsyncClient) -> None:
    chat_id = await _chat(client)
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={"text": TEXT, "mode": "image", "settings": {"steps": 201}},
        )

    assert _errors(response) == STEPS_PAST


async def test_a_prompt_library_batch_names_the_saved_size_and_where_it_is_saved(
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
            "idempotency_key": "refused-setting-template",
            "name": "Refused setting",
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
    chat_id = await _chat(client)
    saved = await client.patch(
        f"/api/chats/{chat_id}", json={"generation_settings_json": {"image": TOO_WIDE}}
    )
    assert saved.status_code == 200, saved.text
    created_batch = await client.post(
        f"/api/chats/{chat_id}/prompt-batches",
        json={
            "idempotency_key": "refused-setting-preview",
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
                "idempotency_key": "refused-setting-queue",
                "expected_plan_version": batch["plan_version"],
                "expected_plan_sha256": batch["plan_sha256"],
            },
        )

    assert _errors(queued, "turn-size-unsupported") == WIDTH_SAVED


async def test_a_regenerate_names_the_setting_it_sends(client: AsyncClient) -> None:
    original = await _completed(client, await _chat(client), {})

    regenerated = await client.post(
        f"/api/messages/{original['assistant_message']['id']}/regenerate",
        json={"settings": {"steps": 201}},
    )

    assert _errors(regenerated) == STEPS_PAST


async def test_a_regenerate_says_a_setting_came_from_the_response_it_regenerates(
    client: AsyncClient,
) -> None:
    """The conversation chose another workflow since, which has no such setting."""

    first = _family("First family", {"type": "object", "properties": {"detail": DETAIL}})
    other = _family("Other family", {})
    chat_id = await _chat(client)
    await _choose(client, chat_id, first)
    original = await _completed(client, chat_id, {"detail": 3})
    await _choose(client, chat_id, other)

    regenerated = await client.post(
        f"/api/messages/{original['assistant_message']['id']}/regenerate",
        json={"settings": {}},
    )

    assert _errors(regenerated) == CARRIED_OVER


async def test_an_edit_names_the_setting_it_sends(app: FastAPI, client: AsyncClient) -> None:
    chat_id = await _chat(client)
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat_id}/turns", json={"text": TEXT, "mode": "image"}
        )
        assert source.status_code == 202, source.text
        edited = await client.post(
            f"/api/messages/{source.json()['user_message']['id']}/edits",
            json={
                "text": TEXT,
                "mode": "image",
                "idempotency_key": "refused-setting-edit",
                "settings": {"steps": 201},
            },
        )

    assert _errors(edited) == STEPS_PAST


@pytest.mark.parametrize("sent", [False, True], ids=["carried over", "sent"])
async def test_a_branch_names_the_setting_and_says_when_it_was_carried_over(
    app: FastAPI, client: AsyncClient, sent: bool
) -> None:
    first = _revision(
        "First workflow", input_schema_json={"type": "object", "properties": {"detail": DETAIL}}
    )
    other = _revision("Other workflow")
    profile_id = _profile("Branched model")
    chat_id = await _chat(client)
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": TEXT,
                "mode": "image",
                "workflow_revision_id": first,
                "profile_id": profile_id,
                "settings": {"detail": 3},
            },
        )
        assert source.status_code == 202, source.text
        branched = await client.post(
            f"/api/messages/{source.json()['user_message']['id']}/branch",
            json={
                "text": TEXT,
                "workflow_revision_id": other,
                "profile_id": profile_id,
                **({"settings": {"detail": 3}} if sent else {}),
            },
        )

    named = [{"code": "setting-unknown", "field": "detail"}]
    assert _errors(branched) == (named if sent else CARRIED_OVER)


async def test_a_remix_preview_and_queue_name_the_choice_they_cannot_apply(
    client: AsyncClient,
) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    choice = {
        "workflow_revision_id": _revision("Remix workflow"),
        "profile_id": _profile("Remix model"),
        # This workflow knows no sampler names, so the picture's cannot be applied.
        "apply": ["sampler"],
    }
    named = [{"code": "setting-not-applicable", "field": "sampler"}]

    previewed = await client.post(f"/api/artifacts/{artifact_id}/remix-preview", json=choice)
    queued = await client.post(
        f"/api/chats/{await _chat(client)}/remixes",
        json={**choice, "artifact_id": artifact_id, "review_digest": "sha256:" + "0" * 64},
    )

    assert _errors(previewed, "remix-choice-invalid") == named
    assert _errors(queued, "remix-choice-invalid") == named


async def test_an_enhance_preview_names_the_setting_beside_its_own_code(
    client: AsyncClient,
) -> None:
    chat_id, _revision_id, payload = await _selection(client, adjustable=True)

    response = await client.post(
        f"/api/chats/{chat_id}/upscale/preview", json={**payload, "settings": {"steps": 201}}
    )

    assert _errors(response, "upscale-preview-unavailable", 409) == STEPS_PAST


async def test_an_enhance_preview_names_the_saved_size_and_where_it_is_saved(
    client: AsyncClient,
) -> None:
    chat_id, revision_id, payload = await _selection(client, adjustable=False)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        schema = copy.deepcopy(revision.input_schema_json)
        schema["properties"]["width"] = dict(CAP)
        revision.input_schema_json = schema
        session.commit()
    saved = await client.patch(
        f"/api/chats/{chat_id}", json={"generation_settings_json": {"image": {"width": 2176}}}
    )
    assert saved.status_code == 200, saved.text

    response = await client.post(f"/api/chats/{chat_id}/upscale/preview", json=payload)

    assert _errors(response, "turn-size-unsupported") == WIDTH_SAVED


async def test_a_comparison_names_the_setting_in_its_refusal(client: AsyncClient) -> None:
    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    second = _arm("Second", _profile("Second model"), _revision("Second workflow"), steps=999)

    response = await client.post(PREFLIGHT, json=_request(first, second))

    assert response.status_code == 200, response.text
    [refusal] = response.json()["refusals"]
    assert (refusal["code"], refusal["setting"]) == ("arm-setting-invalid", "steps")
    for item in refusal["errors"]:
        assert isinstance(item.pop("message"), str)
    assert refusal["errors"] == STEPS_PAST


async def test_a_comparison_names_the_saved_size_and_where_it_is_saved(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a choice that keeps a picture's own size can meet it, so the resolver is made to."""

    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    refused_profile = _profile("Saved size model")
    second = _arm("Second", refused_profile, _revision("Second workflow"))
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
    [refusal] = response.json()["refusals"]
    assert (refusal["code"], refusal["setting"]) == ("arm-setting-invalid", "width")
    for item in refusal["errors"]:
        assert isinstance(item.pop("message"), str)
    assert refusal["errors"] == [
        {"code": "setting-above-maximum", "field": "width", "limit": 4096, "source": "profile"}
    ]
