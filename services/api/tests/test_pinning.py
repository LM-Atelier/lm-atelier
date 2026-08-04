"""A pin says where someone works; archiving says they are done."""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient

pytestmark = pytest.mark.asyncio


async def _chat(client: AsyncClient, title: str) -> str:
    response = await client.post("/api/chats", json={"title": title})
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def test_pinned_chats_sort_first_and_keep_recency_among_themselves(
    client: AsyncClient,
) -> None:
    oldest = await _chat(client, "Oldest")
    middle = await _chat(client, "Middle")
    newest = await _chat(client, "Newest")

    for chat_id in (oldest, middle):
        assert (
            await client.patch(f"/api/chats/{chat_id}", json={"pinned": True})
        ).status_code == 200

    listed = (await client.get("/api/chats")).json()
    order = [item["id"] for item in listed]

    # Pinned first, but recency still orders them inside that group - pinning
    # two chats must not freeze their order relative to each other forever.
    assert order[:2] == [middle, oldest]
    assert newest in order[2:]


async def test_a_pin_survives_the_project_being_archived(client: AsyncClient) -> None:
    project = (await client.post("/api/projects", json={"name": "Studies"})).json()
    chat_id = (
        await client.post("/api/chats", json={"title": "Kept", "project_id": project["id"]})
    ).json()["id"]
    await client.patch(f"/api/chats/{chat_id}", json={"pinned": True})

    await client.patch(f"/api/projects/{project['id']}", json={"archived": True})

    # The pin is the person's statement about their own work, not the
    # project's, so archiving the project does not withdraw it.
    listed = (await client.get("/api/chats")).json()
    kept = next(item for item in listed if item["id"] == chat_id)
    assert kept["pinned"] is True


async def test_pinning_and_archiving_are_independent(client: AsyncClient) -> None:
    chat_id = await _chat(client, "Both")
    await client.patch(f"/api/chats/{chat_id}", json={"pinned": True, "archived": True})

    # No cap and no coupling: a pinned archived chat is a legitimate state,
    # and hiding it is what archived already means.
    hidden = (await client.get("/api/chats")).json()
    assert all(item["id"] != chat_id for item in hidden)

    shown = (await client.get("/api/chats", params={"include_archived": "true"})).json()
    found = next(item for item in shown if item["id"] == chat_id)
    assert found["pinned"] is True and found["archived"] is True


async def test_nothing_limits_how_many_things_are_pinned(client: AsyncClient) -> None:
    ids = [await _chat(client, f"Pinned {index}") for index in range(12)]
    for chat_id in ids:
        assert (
            await client.patch(f"/api/chats/{chat_id}", json={"pinned": True})
        ).status_code == 200

    listed = (await client.get("/api/chats")).json()
    assert sum(1 for item in listed if item["pinned"]) == 12
