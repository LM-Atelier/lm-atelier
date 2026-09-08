from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import ResponseRevision, WorkPlan


async def _complete(client: AsyncClient, run_id: str) -> None:
    async with asyncio.timeout(10):
        while True:
            response = await client.get(f"/api/runs/{run_id}")
            assert response.status_code == 200, response.text
            status = response.json()["status"]
            if status == "complete":
                return
            assert status not in {"failed", "cancelled"}, response.text
            await asyncio.sleep(0.02)


async def _source(client: AsyncClient, chat_id: str | None = None) -> dict[str, Any]:
    if chat_id is None:
        chat = await client.post("/api/chats", json={"title": "Retryable regeneration"})
        assert chat.status_code == 201
        chat_id = chat.json()["id"]
    response = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={"text": "Explain how a paper boat floats.", "mode": "text"},
    )
    assert response.status_code == 202, response.text
    source = response.json()
    await _complete(client, source["run"]["id"])
    return source


@pytest.mark.parametrize("concurrent", [False, True])
async def test_regeneration_retry_returns_one_pending_revision(
    app: FastAPI, client: AsyncClient, concurrent: bool
) -> None:
    source = await _source(client)
    message_id = source["assistant_message"]["id"]
    url = f"/api/messages/{message_id}/regenerate"
    payload = {"settings": {"temperature": 0.25}, "idempotency_key": "one-regeneration"}
    async with app.state.services.scheduler.lease("primary"):
        if concurrent:
            first, second = await asyncio.gather(
                client.post(url, json=payload), client.post(url, json=payload)
            )
        else:
            first = await client.post(url, json=payload)
            second = await client.post(url, json=payload)
        assert first.status_code == 202, first.text
        assert second.status_code == 202, second.text
        assert first.json()["run"]["id"] == second.json()["run"]["id"]
        assert first.json()["assistant_message"]["id"] == second.json()["assistant_message"]["id"]
        assert first.json()["assistant_message"]["transcript_visible"] is False
        with SessionLocal() as session:
            plans = list(
                session.scalars(
                    select(WorkPlan).where(WorkPlan.chat_id == source["run"]["chat_id"])
                )
            )
            pending = list(
                session.scalars(
                    select(ResponseRevision).where(
                        ResponseRevision.message_id == message_id,
                        ResponseRevision.status == "pending",
                    )
                )
            )
            assert len(plans) == 2
            assert len(pending) == 1
        another = await client.post(
            url, json={**payload, "idempotency_key": "another-regeneration"}
        )
        assert another.status_code == 409, another.text
    await _complete(client, first.json()["run"]["id"])
    completed_retry = await client.post(url, json=payload)
    assert completed_retry.status_code == 202, completed_retry.text
    assert completed_retry.json()["run"]["id"] == first.json()["run"]["id"]


@pytest.mark.parametrize("conflict", ["settings", "message", "ordinary_turn"])
async def test_regeneration_key_is_bound_to_target_and_explicit_settings(
    app: FastAPI, client: AsyncClient, conflict: str
) -> None:
    source = await _source(client)
    other = await _source(client, source["run"]["chat_id"])
    url = f"/api/messages/{source['assistant_message']['id']}/regenerate"
    payload = {"settings": {"temperature": 0.25}, "idempotency_key": "bound-regeneration"}
    async with app.state.services.scheduler.lease("primary"):
        if conflict == "ordinary_turn":
            first = await client.post(
                f"/api/chats/{source['run']['chat_id']}/turns",
                json={
                    "text": "A distinct question",
                    "mode": "text",
                    "idempotency_key": payload["idempotency_key"],
                },
            )
        else:
            first = await client.post(url, json=payload)
        assert first.status_code == 202, first.text
        if conflict == "settings":
            payload["settings"] = {"temperature": 0.75}
        if conflict == "message":
            url = f"/api/messages/{other['assistant_message']['id']}/regenerate"
        refused = await client.post(url, json=payload)
        assert refused.status_code == 409, refused.text
        assert refused.json()["code"] == "regeneration-request-conflict"
        with SessionLocal() as session:
            plans = list(
                session.scalars(
                    select(WorkPlan).where(WorkPlan.chat_id == source["run"]["chat_id"])
                )
            )
            assert len(plans) == 3


async def test_regeneration_retry_does_not_replay_removed_source_content(
    app: FastAPI, client: AsyncClient
) -> None:
    from local_lm.domain import utcnow
    from local_lm.models import Message

    source = await _source(client)
    url = f"/api/messages/{source['assistant_message']['id']}/regenerate"
    payload = {"settings": {}, "idempotency_key": "removed-source-regeneration"}
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(url, json=payload)
        assert accepted.status_code == 202, accepted.text
        with SessionLocal() as session:
            message = session.get(Message, accepted.json()["user_message"]["id"])
            assert message is not None
            message.parts.clear()
            session.flush()
            message.content_removed_at = utcnow()
            session.commit()
        refused = await client.post(url, json=payload)
        assert refused.status_code == 409, refused.text
        assert refused.json()["code"] == "source-content-removed"
