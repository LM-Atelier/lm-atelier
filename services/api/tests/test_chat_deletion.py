"""The delete-a-turn cascade: service layer only, endpoint follows later."""

from __future__ import annotations

import asyncio

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.chat_deletion import (
    ExchangeBusy,
    ExchangeHasReplies,
    ExchangeNotFound,
    delete_exchange,
)
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Chat, Job, Message, Run


async def wait_for_run(client: AsyncClient, run_id: str) -> dict:  # type: ignore[type-arg]
    for _ in range(400):
        payload = (await client.get(f"/api/runs/{run_id}")).json()
        if payload["status"] in {"complete", "failed", "cancelled"}:
            return payload
        await asyncio.sleep(0.01)
    raise AssertionError("run did not finish in time")


async def _text_exchange(client: AsyncClient, chat_id: str, text: str) -> dict:  # type: ignore[type-arg]
    accepted = await client.post(f"/api/chats/{chat_id}/turns", json={"text": text, "mode": "text"})
    assert accepted.status_code == 202
    payload = accepted.json()
    await wait_for_run(client, payload["run"]["id"])
    return payload


async def _image_exchange(client: AsyncClient, chat_id: str, text: str) -> dict:  # type: ignore[type-arg]
    accepted = await client.post(
        f"/api/chats/{chat_id}/turns", json={"text": text, "mode": "image"}
    )
    assert accepted.status_code == 202
    payload = accepted.json()
    await wait_for_run(client, payload["run"]["id"])
    return payload


async def test_deleting_a_media_turn_removes_the_exchange_and_releases_its_image(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Deletion"})).json()
    kept = await _text_exchange(client, chat["id"], "Keep this turn")
    removed = await _image_exchange(client, chat["id"], "Create one image of a red barn")

    with SessionLocal() as session:
        result = delete_exchange(session, removed["user_message"]["id"])
        session.commit()

    assert result.chat_id == chat["id"]
    assert removed["user_message"]["id"] in result.message_ids
    assert len(result.run_ids) == 1
    assert result.released_artifact_ids, "the generated image should lose its last reference"

    with SessionLocal() as session:
        remaining_messages = set(
            session.scalars(select(Message.id).where(Message.chat_id == chat["id"]))
        )
        assert kept["user_message"]["id"] in remaining_messages
        assert not remaining_messages & set(result.message_ids)
        assert not session.scalars(select(Run.id).where(Run.id.in_(result.run_ids))).all()
        assert not session.scalars(select(Job.id).where(Job.id.in_(result.job_ids))).all()
        # The artifact row survives for the retention sweep; only references die.
        for artifact_id in result.released_artifact_ids:
            assert session.get(Artifact, artifact_id) is not None
        # The transcript head walked back to the surviving exchange.
        head = session.get(Chat, chat["id"]).active_head_message_id
        assert head not in set(result.message_ids)


async def test_a_turn_with_a_later_reply_is_refused(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Refusal"})).json()
    first = await _text_exchange(client, chat["id"], "First turn")
    await _text_exchange(client, chat["id"], "A reply that depends on the first")

    with SessionLocal() as session:
        with pytest.raises(ExchangeHasReplies):
            delete_exchange(session, first["user_message"]["id"])
        session.rollback()
        # Nothing was deleted by the refused attempt.
        assert session.get(Message, first["user_message"]["id"]) is not None


async def test_a_turn_with_live_jobs_is_refused(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Busy"})).json()
    exchange = await _text_exchange(client, chat["id"], "Finished turn")

    with SessionLocal() as session:
        run_id = session.scalar(
            select(Run.id).where(Run.user_message_id == exchange["user_message"]["id"])
        )
        session.add(
            Job(kind="chat", status="queued", phase="queued", run_id=run_id, payload_json={})
        )
        session.commit()

    with SessionLocal() as session, pytest.raises(ExchangeBusy):
        delete_exchange(session, exchange["user_message"]["id"])


async def test_an_artifact_still_referenced_elsewhere_is_retained(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Shared"})).json()
    generated = await _image_exchange(client, chat["id"], "Create one image of a green door")

    with SessionLocal() as session:
        artifact_id = session.scalar(
            select(Artifact.id).order_by(Artifact.created_at.desc()).limit(1)
        )
    assert artifact_id

    # The surviving reference must live on another branch of history - a
    # follow-up in the same chat would be a reply and correctly refuse the
    # deletion instead.
    other_chat = (await client.post("/api/chats", json={"title": "Reuses the image"})).json()
    follow_up = await client.post(
        f"/api/chats/{other_chat['id']}/turns",
        json={
            "text": "Describe the attached image",
            "mode": "text",
            "input_artifact_ids": [artifact_id],
        },
    )
    assert follow_up.status_code == 202
    await wait_for_run(client, follow_up.json()["run"]["id"])

    with SessionLocal() as session:
        result = delete_exchange(session, generated["user_message"]["id"])
        session.commit()

    # The follow-up still references the image, so it is retained, not released.
    assert artifact_id in result.retained_artifact_ids
    assert artifact_id not in result.released_artifact_ids


async def test_the_delete_endpoint_removes_an_exchange_with_typed_refusals(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Endpoint"})).json()
    kept = await _text_exchange(client, chat["id"], "Keep this turn")
    removed = await _image_exchange(client, chat["id"], "Create one image of a red kite")

    # Deleting the earlier turn while a later reply exists refuses, typed.
    refused = await client.delete(f"/api/messages/{kept['user_message']['id']}/exchange")
    assert refused.status_code == 409
    assert refused.json()["code"] == "exchange-has-replies"
    assert refused.json()["reply_count"] == 1

    deleted = await client.delete(f"/api/messages/{removed['user_message']['id']}/exchange")
    assert deleted.status_code == 200
    body = deleted.json()
    assert body["released_artifact_ids"], "the generated image loses its last reference"
    assert removed["user_message"]["id"] in body["message_ids"]

    detail = (await client.get(f"/api/chats/{chat['id']}")).json()
    remaining = {message["id"] for message in detail["messages"]}
    assert kept["user_message"]["id"] in remaining
    assert not remaining & set(body["message_ids"])

    again = await client.delete(f"/api/messages/{removed['user_message']['id']}/exchange")
    assert again.status_code == 404
    assert again.json()["code"] == "exchange-not-found"


async def test_deleting_twice_reports_not_found(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Idempotent"})).json()
    exchange = await _text_exchange(client, chat["id"], "Delete me twice")

    with SessionLocal() as session:
        delete_exchange(session, exchange["user_message"]["id"])
        session.commit()

    with SessionLocal() as session, pytest.raises(ExchangeNotFound):
        delete_exchange(session, exchange["user_message"]["id"])
