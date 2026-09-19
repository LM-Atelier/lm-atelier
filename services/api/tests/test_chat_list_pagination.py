from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event

from local_lm.db import SessionLocal
from local_lm.models import Chat


async def _create_chat(client: AsyncClient, title: str) -> str:
    response = await client.post("/api/chats", json={"title": title})
    assert response.status_code == 201
    identity = response.json()["id"]
    assert isinstance(identity, str)
    return identity


async def test_chat_pages_bound_results_without_changing_the_unpaged_list(
    client: AsyncClient,
) -> None:
    for number in range(5):
        await _create_chat(client, f"Page navigation {number}")

    complete = await client.get("/api/chats", params={"query": "Page navigation"})
    page = await client.get(
        "/api/chats", params={"query": "Page navigation", "limit": 2, "offset": 1}
    )
    end = await client.get(
        "/api/chats", params={"query": "Page navigation", "limit": 2, "offset": 5}
    )

    assert complete.status_code == page.status_code == end.status_code == 200
    assert len(complete.json()) == 5
    assert page.json() == complete.json()[1:3]
    assert end.json() == []


async def test_chat_pages_apply_archive_and_title_filters_before_the_limit(
    client: AsyncClient,
) -> None:
    kept = await _create_chat(client, "Page filter match")
    archived = await _create_chat(client, "Page filter archived")
    await _create_chat(client, "Unrelated navigation title")
    response = await client.patch(f"/api/chats/{archived}", json={"archived": True})
    assert response.status_code == 200

    page = await client.get("/api/chats", params={"query": "Page filter", "limit": 1})
    inclusive = await client.get(
        "/api/chats",
        params={"query": "Page filter", "include_archived": "true", "limit": 1},
    )

    assert page.status_code == inclusive.status_code == 200
    assert [item["id"] for item in page.json()] == [kept]
    assert [item["id"] for item in inclusive.json()] == [archived]


async def test_chat_pages_use_a_stable_tiebreaker_after_pin_and_recency(
    client: AsyncClient,
) -> None:
    identities = [await _create_chat(client, f"Page tie {number}") for number in range(4)]
    with SessionLocal() as session:
        for identity in identities:
            row = session.get(Chat, identity)
            assert row is not None
            row.updated_at = datetime(2026, 1, 1, tzinfo=UTC)
            row.pinned = identity == identities[0]
        session.commit()

    expected = [identities[0], *sorted(identities[1:], reverse=True)]
    actual: list[str] = []
    for offset in (0, 2):
        page = await client.get(
            "/api/chats", params={"query": "Page tie", "limit": 2, "offset": offset}
        )
        assert page.status_code == 200
        actual.extend(item["id"] for item in page.json())
    assert actual == expected


@pytest.mark.parametrize(
    "parameters", [{"limit": 0}, {"limit": -1}, {"limit": 201}, {"offset": -1}]
)
async def test_chat_pages_refuse_invalid_bounds(
    client: AsyncClient, parameters: dict[str, int]
) -> None:
    response = await client.get("/api/chats", params=parameters)
    assert response.status_code == 422


async def test_workspace_search_matches_project_names_before_paging(client: AsyncClient) -> None:
    project = await client.post("/api/projects", json={"name": "Harbor notebook"})
    assert project.status_code == 201
    created = await client.post(
        "/api/chats", json={"title": "Unrelated title", "project_id": project.json()["id"]}
    )
    assert created.status_code == 201
    await _create_chat(client, "Another unrelated title")

    legacy = await client.get("/api/chats", params={"query": "Harbor"})
    page = await client.get(
        "/api/chats", params={"query": "Harbor", "search_projects": True, "limit": 1}
    )
    assert legacy.status_code == page.status_code == 200
    assert legacy.json() == []
    assert [row["id"] for row in page.json()] == [created.json()["id"]]


async def test_workspace_search_treats_wildcard_characters_as_text(client: AsyncClient) -> None:
    kept = await _create_chat(client, "Progress 10%_complete")
    await _create_chat(client, "Progress 100 complete")
    response = await client.get(
        "/api/chats", params={"query": "%_", "search_projects": True, "limit": 1}
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [kept]


@pytest.mark.parametrize("field", ["title", "project"])
async def test_workspace_search_preserves_unicode_case_matching(
    client: AsyncClient, field: str
) -> None:
    project = await client.post(
        "/api/projects", json={"name": "CAFÉ sketches" if field == "project" else "Sketches"}
    )
    assert project.status_code == 201
    created = await client.post(
        "/api/chats",
        json={
            "title": "CAFÉ scene" if field == "title" else "Scene",
            "project_id": project.json()["id"],
        },
    )
    assert created.status_code == 201
    page = await client.get(
        "/api/chats", params={"query": "café", "search_projects": True, "limit": 1}
    )
    assert page.status_code == 200
    assert [row["id"] for row in page.json()] == [created.json()["id"]]


async def test_workspace_search_hydrates_only_the_requested_page(client: AsyncClient) -> None:
    for number in range(6):
        await _create_chat(client, f"CAFÉ sketch {number}")
        await _create_chat(client, f"Other sketch {number}")
    loaded: list[str] = []

    def record_load(chat: Chat, _context: object) -> None:
        loaded.append(chat.id)

    event.listen(Chat, "load", record_load)
    try:
        response = await client.get(
            "/api/chats",
            params={"query": "café", "search_projects": True, "limit": 2, "offset": 2},
        )
    finally:
        event.remove(Chat, "load", record_load)
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert loaded == [row["id"] for row in response.json()]
