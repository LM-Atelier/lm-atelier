"""A saved size the workflow cannot make on either side names both sides.

Width and height are each taken from the highest layer that names them, and
each is checked before anything is refused, so one refusal names every side
the workflow cannot make, with its limit and the layer it is saved in. The
detail is the first side's, as it always was, and nothing is queued.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_saved_sizes_a_workflow_cannot_make import SAVED, _capped, _save, _send
from test_upscale_preview_api import _selection
from test_workflow_output_geometry_api import _generation_row_counts
from test_workflow_output_geometry_api import geometry_runtime as geometry_runtime
from workflow_fixtures import seed_workflow_trust

from local_lm.db import SessionLocal
from local_lm.models import WorkflowRevision

#: The most the capped workflows make a side, in steps of 64.
CAP = {"type": "integer", "default": 1024, "minimum": 256, "maximum": 2048, "multipleOf": 64}
TOO_LARGE = {"width": 2176, "height": 2240}
WIDTH_DETAIL = (
    "The width of 2176 px saved for this conversation is not one this workflow makes "
    "(256 to 2048 px, in steps of 64). Change it there, or choose another workflow."
)


def _side(field: str, source: str, limit: int = 2048) -> dict[str, Any]:
    return {"code": "setting-above-maximum", "field": field, "limit": limit, "source": source}


def _refused(response: Any) -> tuple[str, list[dict[str, Any]]]:
    """The detail and the items of a saved size refusal, without the items' sentences."""

    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "turn-size-unsupported", body
    items: list[dict[str, Any]] = body["errors"]
    for item in items:
        message = item.pop("message")
        assert isinstance(message, str) and message.endswith("."), item
    detail: str = body["detail"]
    return detail, items


async def test_both_sides_saved_past_the_workflow_are_named_with_the_widths_detail(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)
    await _save(client, chat_id, profile_id, {"chat": TOO_LARGE})
    before = _generation_row_counts()

    response = await _send(app, client, chat_id, revision_id, profile_id)

    assert _generation_row_counts() == before, "nothing was queued"
    assert _refused(response) == (WIDTH_DETAIL, [_side("width", "chat"), _side("height", "chat")])


async def test_each_side_names_the_layer_it_is_saved_in(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)
    await _save(client, chat_id, profile_id, {"chat": {"width": 2176}, "project": {"height": 2240}})
    before = _generation_row_counts()

    response = await _send(app, client, chat_id, revision_id, profile_id)

    assert _generation_row_counts() == before
    assert _refused(response) == (
        WIDTH_DETAIL,
        [_side("width", "chat"), _side("height", "project")],
    )


async def test_a_side_the_workflow_makes_is_not_named(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)
    await _save(client, chat_id, profile_id, {"chat": {"width": 1024, "height": 2240}})
    before = _generation_row_counts()

    response = await _send(app, client, chat_id, revision_id, profile_id)

    assert _generation_row_counts() == before
    detail, items = _refused(response)
    assert detail.startswith(f"The height of 2240 px saved {SAVED['chat']} ")
    assert items == [_side("height", "chat")]


async def test_a_side_the_turn_sends_itself_outranks_the_saved_one(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision_id, profile_id, chat_id = await _capped(app, client, monkeypatch)
    await _save(client, chat_id, profile_id, {"chat": TOO_LARGE})
    before = _generation_row_counts()

    response = await _send(app, client, chat_id, revision_id, profile_id, settings={"width": 1024})

    assert _generation_row_counts() == before
    detail, items = _refused(response)
    assert detail.startswith(f"The height of 2240 px saved {SAVED['chat']} ")
    assert items == [_side("height", "chat")]


async def test_a_prompt_library_batch_names_both_saved_sides(
    app: FastAPI, client: AsyncClient
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Both sides batch workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "BothSidesBatchWorkflow"}},
            "input_schema": {"type": "object", "properties": {"width": CAP, "height": CAP}},
        },
    )
    assert created.status_code == 201, created.text
    revision_id = created.json()["current_revision_id"]
    seed_workflow_trust(revision_id)
    template = await client.post(
        "/api/prompt-templates",
        json={
            "idempotency_key": "both-sides-template",
            "name": "Both sides",
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
    chat = await client.post("/api/chats", json={"title": "Both sides batch"})
    assert chat.status_code == 201, chat.text
    chat_id = chat.json()["id"]
    saved = await client.patch(
        f"/api/chats/{chat_id}", json={"generation_settings_json": {"image": TOO_LARGE}}
    )
    assert saved.status_code == 200, saved.text
    created_batch = await client.post(
        f"/api/chats/{chat_id}/prompt-batches",
        json={
            "idempotency_key": "both-sides-preview",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": 1,
            "selection_seed": 3,
            "inputs": {"subject": ["gray square"]},
        },
    )
    assert created_batch.status_code in (200, 201), created_batch.text
    batch = created_batch.json()
    before = _generation_row_counts()

    async with app.state.services.scheduler.lease("primary"):
        queued = await client.post(
            f"/api/prompt-batches/{batch['id']}/queue",
            json={
                "idempotency_key": "both-sides-queue",
                "expected_plan_version": batch["plan_version"],
                "expected_plan_sha256": batch["plan_sha256"],
            },
        )

    assert _generation_row_counts() == before
    assert _refused(queued) == (WIDTH_DETAIL, [_side("width", "chat"), _side("height", "chat")])


async def test_an_enhance_preview_names_both_saved_sides(client: AsyncClient) -> None:
    chat_id, revision_id, payload = await _selection(client, adjustable=False)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        schema = copy.deepcopy(revision.input_schema_json)
        schema["properties"]["width"] = dict(CAP)
        schema["properties"]["height"] = dict(CAP)
        revision.input_schema_json = schema
        session.commit()
    saved = await client.patch(
        f"/api/chats/{chat_id}", json={"generation_settings_json": {"image": TOO_LARGE}}
    )
    assert saved.status_code == 200, saved.text

    response = await client.post(f"/api/chats/{chat_id}/upscale/preview", json=payload)

    assert _refused(response) == (WIDTH_DETAIL, [_side("width", "chat"), _side("height", "chat")])


async def test_a_saved_video_size_names_both_sides(client: AsyncClient) -> None:
    span = {"type": "integer", "default": 768, "minimum": 128, "maximum": 1536, "multipleOf": 32}
    workflow = await client.post(
        "/api/workflows",
        json={
            "name": "Both sides video",
            "operation": "text_to_video",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "Mock"}},
            "input_schema": {
                "type": "object",
                "properties": {
                    "width": dict(span),
                    "height": dict(span),
                    "frames": {"type": "integer", "default": 81, "minimum": 1},
                    "fps": {"type": "number", "default": 16, "minimum": 1},
                },
            },
        },
    )
    assert workflow.status_code == 201, workflow.text
    revision_id = workflow.json()["current_revision_id"]
    seed_workflow_trust(revision_id)
    project = await client.post(
        "/api/projects",
        json={"name": "Both sides video", "video_workflow_revision_id": revision_id},
    )
    assert project.status_code == 201, project.text
    chat = await client.post(
        "/api/chats", json={"title": "Both sides video", "project_id": project.json()["id"]}
    )
    assert chat.status_code == 201, chat.text
    saved = await client.patch(
        f"/api/chats/{chat.json()['id']}",
        json={"generation_settings_json": {"video": {"width": 1920, "height": 1600}}},
    )
    assert saved.status_code == 200, saved.text
    before = _generation_row_counts()

    response = await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={"text": "A slow camera move across a gray table", "mode": "video"},
    )

    assert _generation_row_counts() == before
    detail, items = _refused(response)
    assert detail.startswith(f"The width of 1920 px saved {SAVED['chat']} ")
    assert "(128 to 1536 px, in steps of 32)" in detail
    assert items == [_side("width", "chat", 1536), _side("height", "chat", 1536)]
