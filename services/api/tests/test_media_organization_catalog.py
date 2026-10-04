"""Catalog choices stay bounded and refuse pages from an earlier revision."""

from __future__ import annotations

import base64
import json
import time
from typing import cast

import pytest
from httpx2 import AsyncClient
from sqlalchemy import text

from local_lm import db


async def _create(client: AsyncClient, kind: str, index: int) -> str:
    path = "/api/media-collections" if kind == "albums" else "/api/media-tags"
    payload = {"name": f"Study {index}"} if kind == "albums" else {"label": f"Study {index}"}
    response = await client.post(path, json=payload)
    assert response.status_code == 201
    return cast(str, response.json()["id"])


async def _page(client: AsyncClient, kind: str, **params: object) -> dict[str, object]:
    response = await client.get(
        f"/api/media-organization/catalog/{kind}", params=cast(dict[str, str], params)
    )
    assert response.status_code == 200
    return cast(dict[str, object], response.json())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["albums", "tags"])
async def test_catalog_pages_visit_each_identity_once_with_bounded_results(
    client: AsyncClient, kind: str
) -> None:
    ids = [await _create(client, kind, index) for index in range(5)]
    first = await _page(client, kind, limit=2)
    assert set(first) == {"items", "next_cursor", "revision"}
    assert len(cast(list[object], first["items"])) == 2
    assert type(first["revision"]) is int
    pages = [first]
    while pages[-1]["next_cursor"] is not None:
        pages.append(await _page(client, kind, limit=2, cursor=pages[-1]["next_cursor"]))
        assert pages[-1]["revision"] == first["revision"]
        assert len(cast(list[object], pages[-1]["items"])) <= 2
        assert len(pages) <= 3
    seen = [row["id"] for page in pages for row in cast(list[dict[str, object]], page["items"])]
    assert seen == ids


@pytest.mark.asyncio
async def test_catalog_keyset_breaks_creation_time_ties_with_identity(client: AsyncClient) -> None:
    await _page(client, "albums")
    ids = ["collection_" + digit * 32 for digit in ("c", "a", "b")]
    with db.SessionLocal() as session:
        for identity in ids:
            session.execute(
                text(
                    "INSERT INTO media_collections "
                    "(id,kind,name,description,version,created_at,updated_at) VALUES "
                    "(:id,'manual','Same name','',1,'2026-10-03 12:00:00.000000',"
                    "'2026-10-03 12:00:00.000000')"
                ),
                {"id": identity},
            )
        session.commit()
    first = await _page(client, "albums", limit=1)
    second = await _page(client, "albums", limit=1, cursor=first["next_cursor"])
    third = await _page(client, "albums", limit=1, cursor=second["next_cursor"])
    assert [
        cast(list[dict[str, object]], page["items"])[0]["id"] for page in (first, second, third)
    ] == sorted(ids)
    assert third["next_cursor"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["albums", "tags"])
@pytest.mark.parametrize("mutation", ["insert", "rename", "delete", "membership-version"])
async def test_catalog_changes_refuse_the_next_page_before_returning_choices(
    client: AsyncClient, kind: str, mutation: str
) -> None:
    ids = [await _create(client, kind, index) for index in range(3)]
    first = await _page(client, kind, limit=1)
    table = "media_collections" if kind == "albums" else "media_tags"
    if mutation == "insert":
        await _create(client, kind, 4)
    else:
        with db.SessionLocal() as session:
            if mutation == "delete":
                statement = f"DELETE FROM {table} WHERE id=:id"
            elif mutation == "rename":
                changes = "name='Renamed'" if kind == "albums" else "label='Renamed',slug='renamed'"
                statement = f"UPDATE {table} SET {changes},version=version+1 WHERE id=:id"
            else:
                statement = f"UPDATE {table} SET version=version+1 WHERE id=:id"
            session.execute(text(statement), {"id": ids[-1]})
            session.commit()
    response = await client.get(
        f"/api/media-organization/catalog/{kind}",
        params={"limit": 1, "cursor": str(first["next_cursor"])},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "media-catalog-stale"
    assert "items" not in response.json()
    fresh = await _page(client, kind, limit=1)
    assert cast(int, fresh["revision"]) > cast(int, first["revision"])


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["query", "limit", "kind", "signature"])
async def test_catalog_cursors_are_bound_to_the_exact_page_request(
    client: AsyncClient, changed: str
) -> None:
    for index in range(3):
        await _create(client, "albums", index)
    first = await _page(client, "albums", limit=1)
    cursor = cast(str, first["next_cursor"])
    params: dict[str, object] = {"limit": 1, "cursor": cursor}
    kind = "albums"
    if changed == "query":
        params["query"] = "Study"
    elif changed == "limit":
        params["limit"] = 2
    elif changed == "kind":
        kind = "tags"
    else:
        parts = cursor.split(".")
        params["cursor"] = parts[0] + "." + ("a" if parts[1][0] != "a" else "b") + parts[1][1:]
    response = await client.get(
        f"/api/media-organization/catalog/{kind}", params=cast(dict[str, str], params)
    )
    assert response.status_code == 422
    assert response.json()["code"] == "media-catalog-invalid"
    assert "items" not in response.json()


@pytest.mark.asyncio
async def test_catalog_search_escapes_wildcards_and_keeps_queries_out_of_cursors(
    client: AsyncClient,
) -> None:
    for name in ("50% Study", "50% Sketch", "500 Study", "Different"):
        assert (await client.post("/api/media-collections", json={"name": name})).status_code == 201
    first = await _page(client, "albums", query=" 50% ", limit=1)
    assert cast(list[dict[str, object]], first["items"])[0]["name"] == "50% Study"
    cursor = cast(str, first["next_cursor"])
    payload = base64.urlsafe_b64decode(
        cursor.split(".")[0] + "=" * (-len(cursor.split(".")[0]) % 4)
    )
    assert b"50%" not in payload
    assert len(json.loads(payload)["filters"]["query"]) == 64
    second = await _page(client, "albums", query="50%", limit=1, cursor=cursor)
    assert [item["name"] for item in cast(list[dict[str, object]], second["items"])] == [
        "50% Sketch"
    ]
    assert second["next_cursor"] is None


@pytest.mark.asyncio
async def test_catalog_expiry_requires_a_fresh_first_page(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(2):
        await _create(client, "albums", index)
    first = await _page(client, "albums", limit=1)
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + 901)
    response = await client.get(
        "/api/media-organization/catalog/albums",
        params={"limit": 1, "cursor": str(first["next_cursor"])},
    )
    assert response.status_code == 422
    assert (await _page(client, "albums", limit=1))["items"] == first["items"]


@pytest.mark.asyncio
async def test_catalog_keyset_uses_the_creation_identity_index(client: AsyncClient) -> None:
    await _page(client, "albums")
    with db.SessionLocal() as session:
        for table, index in (
            ("media_collections", "ix_media_collection_catalog_order"),
            ("media_tags", "ix_media_tag_catalog_order"),
        ):
            plan = session.execute(
                text(
                    f"EXPLAIN QUERY PLAN SELECT id FROM {table} "
                    "WHERE (created_at,id) > ('2026-10-03','') ORDER BY created_at,id LIMIT 51"
                )
            ).all()
            description = " ".join(str(row[-1]) for row in plan)
            assert "SEARCH" in description
            assert index in description
            assert "TEMP B-TREE" not in description
