"""Forking a chat that contains a removed item, driven through the real routes.

The removal design deliberately allows forking a thread that contains a removed
item - refusing there would block forking any chat with one. Allowing the fork is
not the same as dropping the mark. A clone that is empty AND unmarked is not a
tombstone; it is a message that looks like it never had content, which is a
different claim about the conversation than the one the original makes.

Driven end to end rather than by calling the cloner: create a chat, remove an
ancestor's content through `POST /api/messages/{id}/remove-content`, then fork a
later message through `POST /api/messages/{id}/fork`, and read the clone back out
of the database. The removal must actually have happened for the fork to be
asked the question at all, so the first assertion is about the source.
"""

from __future__ import annotations

from typing import Any, cast

from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Chat, Message


async def _wait_for_run(client: AsyncClient, run_id: str) -> dict[str, Any]:
    async def read() -> dict[str, Any]:
        payload: dict[str, Any] = (await client.get(f"/api/runs/{run_id}")).json()
        return payload

    return cast(
        dict[str, Any],
        await wait_for_terminal_status(read, what=f"run {run_id}", expected=None),
    )


async def _turn(client: AsyncClient, chat_id: str, text: str) -> dict[str, Any]:
    accepted = await client.post(f"/api/chats/{chat_id}/turns", json={"text": text, "mode": "text"})
    assert accepted.status_code == 202
    payload: dict[str, Any] = accepted.json()
    await _wait_for_run(client, payload["run"]["id"])
    return payload


async def test_forking_past_a_removed_item_keeps_it_marked_as_removed(
    client: AsyncClient,
) -> None:
    source = (await client.post("/api/chats", json={"title": "Has a removal"})).json()
    first = await _turn(client, source["id"], "First question")
    second = await _turn(client, source["id"], "Second question")

    target = first["assistant_message"]["id"]
    preview = await client.get(f"/api/messages/{target}/removal-impact")
    assert preview.status_code == 200
    removed = await client.post(
        f"/api/messages/{target}/remove-content",
        json={
            "expected_message_id": target,
            "expected_revision_id": preview.json()["message_revision_id"],
            "operation_key": "fork-preserves-removal",
        },
    )
    assert removed.status_code == 200

    # The source really is tombstoned, or the fork is never asked the question.
    with SessionLocal() as session:
        original = session.get(Message, target)
        assert original is not None
        assert original.content_removed_at is not None
        assert original.parts == []

    forked = await client.post(f"/api/messages/{second['assistant_message']['id']}/fork")
    assert forked.status_code == 201
    fork_id = forked.json()["id"]

    with SessionLocal() as session:
        chat = session.get(Chat, fork_id)
        assert chat is not None
        clones = list(session.scalars(select(Message).where(Message.chat_id == fork_id)).all())
        tombstoned = [message for message in clones if message.content_removed_at is not None]
        empty = [message for message in clones if not message.parts]

        # The lineage travelled, so the removed ancestor is among the clones.
        assert len(clones) >= 4
        # Exactly the messages that carry no content are the ones marked removed.
        # Without this, a fork mints an empty message that does not say why it is
        # empty, and nothing downstream can tell a tombstone from a blank turn.
        assert [message.id for message in empty] == [message.id for message in tombstoned]
        assert len(tombstoned) == 1
