"""Manual Media Library collections are created through the shipped API."""

from __future__ import annotations

from httpx import AsyncClient


async def test_a_manual_collection_can_be_created(client: AsyncClient) -> None:
    created = await client.post(
        "/api/media-collections",
        json={"name": "Still life", "description": "A neutral fixture."},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "Still life"
    assert body["description"] == "A neutral fixture."
    assert body["kind"] == "manual"
    assert body["version"] == 1
    assert body["id"].startswith("collection_")

    listed = await client.get("/api/media-collections")
    assert listed.status_code == 200
    assert any(item["id"] == body["id"] for item in listed.json()["collections"])


async def test_a_blank_collection_name_is_refused(client: AsyncClient) -> None:
    response = await client.post("/api/media-collections", json={"name": "  "})
    assert response.status_code == 422
    assert response.json()["code"] == "media-collection-invalid"
