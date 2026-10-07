"""Deleted conversations cannot supply revisions, replayed requests or recipes."""

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import select
from test_chat_deletion import _text_exchange, wait_for_run
from test_chat_recovery import _history
from test_chat_recovery_replay import _change
from test_prompt_library_api import _create_payload

from local_lm.db import SessionLocal
from local_lm.models import EditTemplate, Message, PromptExpansionBatch, Run


async def _trash(client: AsyncClient, chat_id: str) -> dict[str, Any]:
    return await _change(
        client,
        f"/api/chats/{chat_id}/deletion-impact",
        f"/api/chats/{chat_id}/trash",
        "trash-reusable-inputs",
    )


async def _restore(client: AsyncClient, deletion: dict[str, Any]) -> None:
    deletion_id = deletion["deletion_id"]
    await _change(
        client,
        f"/api/recovery-items/{deletion_id}/impact",
        f"/api/recovery-items/{deletion_id}/restore",
        "restore-reusable-inputs",
    )


def _snapshot(chat_id: str) -> dict[str, Any]:
    with SessionLocal() as session:
        return {
            "history": _history(chat_id),
            "selected": list(
                session.execute(
                    select(Message.id, Message.active_response_revision_id)
                    .where(Message.chat_id == chat_id)
                    .order_by(Message.id)
                )
            ),
            "templates": list(session.scalars(select(EditTemplate.id).order_by(EditTemplate.id))),
            "batches": list(
                session.scalars(select(PromptExpansionBatch.id).order_by(PromptExpansionBatch.id))
            ),
        }


async def _exchange(client: AsyncClient) -> tuple[str, dict[str, Any]]:
    created = await client.post("/api/chats", json={"title": "Garden revisions"})
    assert created.status_code == 201
    chat_id = created.json()["id"]
    return chat_id, await _text_exchange(client, chat_id, "Keep the garden layout")


async def _regenerate(client: AsyncClient, message_id: str) -> dict[str, Any]:
    response = await client.post(
        f"/api/messages/{message_id}/regenerate",
        json={"settings": {}, "idempotency_key": "garden-regeneration"},
    )
    assert response.status_code == 202, response.text
    body: dict[str, Any] = response.json()
    await wait_for_run(client, body["run"]["id"])
    return body


async def test_selecting_a_deleted_response_refuses_without_changing_its_revision(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat_id, original = await _exchange(client)
    message_id = original["assistant_message"]["id"]
    with SessionLocal() as session:
        message = session.get(Message, message_id)
        assert message is not None
        revision_id = message.active_response_revision_id
        assert revision_id is not None
    await _regenerate(client, message_id)
    deletion = await _trash(client, chat_id)
    before = _snapshot(chat_id)
    path = f"/api/messages/{message_id}/revisions/{revision_id}/select"

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url=client.base_url,
        cookies=client.cookies,
        headers=client.headers,
    ) as reader:
        refused = await reader.post(path)
    assert refused.status_code == 404, refused.text
    assert refused.json()["code"] == "response-revision-not-found"
    assert _snapshot(chat_id) == before

    await _restore(client, deletion)
    selected = await client.post(path)
    assert selected.status_code == 200, selected.text
    assert selected.json()["id"] == message_id
    assert selected.json()["active_response_revision_id"] == revision_id
    assert _history(chat_id)["runs"] == before["history"]["runs"]


@pytest.mark.parametrize("replay", [False, True])
async def test_regeneration_cannot_replay_a_deleted_response(
    client: AsyncClient,
    replay: bool,
) -> None:
    chat_id, original = await _exchange(client)
    message_id = original["assistant_message"]["id"]
    regenerated = await _regenerate(client, message_id)
    deletion = await _trash(client, chat_id)
    before = _snapshot(chat_id)
    request: dict[str, object] = {
        "settings": {},
        "idempotency_key": "garden-regeneration" if replay else "another-garden-regeneration",
    }

    refused = await client.post(f"/api/messages/{message_id}/regenerate", json=request)

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "response-not-regenerable"
    assert _snapshot(chat_id) == before
    await _restore(client, deletion)
    replayed = await client.post(
        f"/api/messages/{message_id}/regenerate",
        json={"settings": {}, "idempotency_key": "garden-regeneration"},
    )
    assert replayed.status_code == 202, replayed.text
    assert replayed.json()["run"]["id"] == regenerated["run"]["id"]
    assert _snapshot(chat_id) == before


@pytest.mark.parametrize("replay", [False, True])
async def test_prompt_batch_creation_cannot_replay_a_deleted_conversation(
    client: AsyncClient,
    replay: bool,
) -> None:
    chat_id, _original = await _exchange(client)
    template = await client.post("/api/prompt-templates", json=_create_payload(name="Garden views"))
    assert template.status_code == 201, template.text
    revision = template.json()["revision"]
    payload = {
        "idempotency_key": "garden-batch",
        "template_revision_id": revision["id"],
        "contract_sha256": revision["contract_sha256"],
        "item_count": 2,
        "selection_seed": 41,
        "inputs": {"subject": ["garden", "river"]},
    }
    path = f"/api/chats/{chat_id}/prompt-batches"
    created = await client.post(path, json=payload)
    assert created.status_code == 201, created.text
    batch = created.json()
    deletion = await _trash(client, chat_id)
    before = _snapshot(chat_id)

    refused = await client.post(
        path, json=payload if replay else {**payload, "idempotency_key": "another-garden-batch"}
    )

    assert refused.status_code == 404, refused.text
    assert refused.json()["code"] == "chat-not-found"
    assert _snapshot(chat_id) == before
    await _restore(client, deletion)
    replayed = await client.post(path, json=payload)
    assert replayed.status_code == 201, replayed.text
    assert replayed.json() == {**batch, "replayed": True}
    assert _snapshot(chat_id) == before


async def test_a_recipe_cannot_copy_a_deleted_runs_settings(client: AsyncClient) -> None:
    chat_id, original = await _exchange(client)
    run_id = original["run"]["id"]
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        run.provenance_json = {
            **run.provenance_json,
            "resolved_settings": {"denoise": 0.42, "seed": 5},
        }
        session.commit()
    payload = {
        "name": "Garden wash",
        "instruction": "Use watercolor",
        "settings_json": {"denoise": 0.99},
        "from_run_id": run_id,
    }
    deletion = await _trash(client, chat_id)
    before = _snapshot(chat_id)

    refused = await client.post("/api/edit-templates", json=payload)

    assert refused.status_code == 404, refused.text
    assert refused.json()["code"] == "run-not-found"
    assert _snapshot(chat_id) == before
    await _restore(client, deletion)
    saved = await client.post("/api/edit-templates", json=payload)
    assert saved.status_code == 201, saved.text
    assert saved.json()["settings_json"] == {"denoise": 0.42}
    assert _history(chat_id) == before["history"]
