"""Forking a thread from a message."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Chat, Message


async def wait_for_run(client: AsyncClient, run_id: str) -> dict[str, Any]:
    for _ in range(400):
        payload: dict[str, Any] = (await client.get(f"/api/runs/{run_id}")).json()
        if payload["status"] in {"complete", "failed", "cancelled"}:
            return payload
        await asyncio.sleep(0.01)
    raise AssertionError("run did not finish in time")


async def _turn(client: AsyncClient, chat_id: str, text: str) -> dict[str, Any]:
    accepted = await client.post(f"/api/chats/{chat_id}/turns", json={"text": text, "mode": "text"})
    assert accepted.status_code == 202
    payload: dict[str, Any] = accepted.json()
    await wait_for_run(client, payload["run"]["id"])
    return payload


async def test_forking_copies_the_history_up_to_the_chosen_message(
    client: AsyncClient,
) -> None:
    source = (await client.post("/api/chats", json={"title": "Original"})).json()
    assert (
        await client.patch(f"/api/chats/{source['id']}", json={"routing_mode": "image"})
    ).status_code == 200
    first = await _turn(client, source["id"], "First question")
    await _turn(client, source["id"], "Second question")

    forked = await client.post(f"/api/messages/{first['assistant_message']['id']}/fork")
    assert forked.status_code == 201
    fork = forked.json()

    assert fork["id"] != source["id"]
    assert fork["title"] == "Original (thread)"
    # Settings travel so the fork behaves like the conversation it came from.
    assert fork["routing_mode"] == "image"
    assert fork["origin_json"]["forked_from_chat_id"] == source["id"]
    assert fork["origin_json"]["forked_from_message_id"] == first["assistant_message"]["id"]

    detail = (await client.get(f"/api/chats/{fork['id']}")).json()
    texts = [
        part["text"]
        for message in detail["messages"]
        for part in message["parts"]
        if part["type"] == "text"
    ]
    assert "First question" in texts
    # History stops at the forked message: the later turn stays behind.
    assert "Second question" not in texts
    assert detail["active_head_message_id"] == detail["messages"][-1]["id"]

    # The original is untouched - forking is not moving.
    original = (await client.get(f"/api/chats/{source['id']}")).json()
    assert len(original["messages"]) == 4
    assert original["origin_json"] == {}


async def test_the_fork_is_independently_editable(client: AsyncClient) -> None:
    source = (await client.post("/api/chats", json={"title": "Shared start"})).json()
    turn = await _turn(client, source["id"], "Common ground")
    fork = (await client.post(f"/api/messages/{turn['assistant_message']['id']}/fork")).json()

    with SessionLocal() as session:
        copied = session.scalars(select(Message.id).where(Message.chat_id == fork["id"])).all()
        original = session.scalars(select(Message.id).where(Message.chat_id == source["id"])).all()
    # New rows, not shared ones: editing the fork must not rewrite history.
    assert not set(copied) & set(original)

    await _turn(client, fork["id"], "Only in the fork")
    unchanged = (await client.get(f"/api/chats/{source['id']}")).json()
    assert len(unchanged["messages"]) == 2


async def test_forking_an_unknown_message_is_refused(client: AsyncClient) -> None:
    response = await client.post("/api/messages/msg_missing/fork")
    assert response.status_code == 404
    assert response.json()["code"] == "fork-source-not-found"


@pytest.mark.parametrize("title", ["A" * 240])
async def test_a_long_title_stays_within_its_bound(client: AsyncClient, title: str) -> None:
    source = (await client.post("/api/chats", json={"title": title})).json()
    turn = await _turn(client, source["id"], "Question")

    forked = await client.post(f"/api/messages/{turn['assistant_message']['id']}/fork")
    assert forked.status_code == 201
    with SessionLocal() as session:
        chat = session.get(Chat, forked.json()["id"])
        assert chat is not None
        assert len(chat.title) <= 240
