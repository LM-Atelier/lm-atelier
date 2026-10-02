"""Manual Media Library tags are created through the shipped API."""

from __future__ import annotations

from httpx import AsyncClient


async def test_a_manual_tag_can_be_created(client: AsyncClient) -> None:
    created = await client.post(
        "/api/media-tags",
        json={"label": "Still life", "color": "#6b705c"},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["label"] == "Still life"
    assert body["slug"] == "still-life"
    assert body["color"] == "#6b705c"
    assert body["version"] == 1
    assert body["id"].startswith("mediatag_")

    listed = await client.get("/api/media-tags")
    assert listed.status_code == 200
    assert any(item["id"] == body["id"] for item in listed.json()["tags"])


async def test_a_blank_tag_label_is_refused(client: AsyncClient) -> None:
    response = await client.post("/api/media-tags", json={"label": "  "})
    assert response.status_code == 422
    assert response.json()["code"] == "media-tag-invalid"

    listed = await client.get("/api/media-tags")
    assert listed.json()["tags"] == []


async def test_a_repeated_tag_slug_is_refused(client: AsyncClient) -> None:
    created = await client.post("/api/media-tags", json={"label": "Still life"})
    assert created.status_code == 201

    repeated = await client.post("/api/media-tags", json={"label": "still life"})
    assert repeated.status_code == 409
    assert repeated.json()["code"] == "media-tag-conflict"

    listed = await client.get("/api/media-tags")
    assert [item["id"] for item in listed.json()["tags"]] == [created.json()["id"]]


async def test_an_unreadable_tag_body_is_refused(client: AsyncClient) -> None:
    response = await client.post(
        "/api/media-tags",
        content=b"\xff",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "media-tag-invalid"
