"""Creation retries confirm their original write without duplicating choices."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from httpx2 import AsyncClient
from sqlalchemy import text

from local_lm import db
from local_lm.models import MediaOrganizationCreation

KEY = "ab" * 16


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["albums", "tags"])
async def test_invalid_names_leave_no_creation_receipt_and_allow_a_corrected_request(
    client: AsyncClient, kind: str
) -> None:
    refused = await client.post(_path(kind), json=_request(kind, name="Invalid\nname"))
    assert refused.status_code == 422
    assert refused.json()["code"] == (
        "media-collection-invalid" if kind == "albums" else "media-tag-invalid"
    )
    key = "collections" if kind == "albums" else "tags"
    assert (await client.get(_path(kind))).json()[key] == []
    with db.SessionLocal() as session:
        assert session.get(MediaOrganizationCreation, KEY) is None
    corrected = await client.post(_path(kind), json=_request(kind))
    assert corrected.status_code == 201
    assert (await client.post(_path(kind), json=_request(kind))).json() == corrected.json()


@pytest.mark.asyncio
async def test_a_duplicate_tag_refusal_leaves_no_new_receipt_and_allows_correction(
    client: AsyncClient,
) -> None:
    first = await client.post(_path("tags"), json=_request("tags"))
    assert first.status_code == 201
    refused_key = "cd" * 16
    refused = await client.post(_path("tags"), json=_request("tags", key=refused_key))
    assert refused.status_code == 409
    assert refused.json()["code"] == "media-tag-conflict"
    assert (await client.get(_path("tags"))).json()["tags"] == [first.json()]
    with db.SessionLocal() as session:
        assert session.get(MediaOrganizationCreation, refused_key) is None
    corrected = await client.post(
        _path("tags"), json=_request("tags", key=refused_key, name="Different")
    )
    assert corrected.status_code == 201
    assert (await client.post(_path("tags"), json=_request("tags"))).json() == first.json()


def _request(kind: str, *, key: object = KEY, name: str = "Studies") -> dict[str, object]:
    return {"operation_key": key, **({"name": name} if kind == "albums" else {"label": name})}


def _path(kind: str) -> str:
    return "/api/media-collections" if kind == "albums" else "/api/media-tags"


async def _replayed(client: AsyncClient, kind: str) -> dict[str, object]:
    first = await client.post(_path(kind), json=_request(kind))
    assert first.status_code == 201
    second = await client.post(_path(kind), json=_request(kind))
    assert second.status_code == 201
    assert second.json() == first.json()
    return cast(dict[str, object], first.json())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["albums", "tags"])
async def test_creation_retries_return_the_same_identity_and_store_one_choice(
    client: AsyncClient, kind: str
) -> None:
    created = await _replayed(client, kind)
    key = "collections" if kind == "albums" else "tags"
    response = await client.get(_path(kind))
    assert response.status_code == 200
    assert [row["id"] for row in response.json()[key]] == [created["id"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["albums", "tags"])
async def test_creation_keys_refuse_a_changed_request_without_another_write(
    client: AsyncClient, kind: str
) -> None:
    first = await client.post(_path(kind), json=_request(kind))
    assert first.status_code == 201
    changed = await client.post(_path(kind), json=_request(kind, name="Different"))
    assert changed.status_code == 409
    assert changed.json()["code"] == "media-creation-conflict"
    key = "collections" if kind == "albums" else "tags"
    assert len((await client.get(_path(kind))).json()[key]) == 1


@pytest.mark.asyncio
async def test_creation_keys_cannot_be_reused_for_a_different_catalog(client: AsyncClient) -> None:
    assert (await client.post(_path("albums"), json=_request("albums"))).status_code == 201
    response = await client.post(_path("tags"), json=_request("tags"))
    assert response.status_code == 409
    assert response.json()["code"] == "media-creation-conflict"
    assert (await client.get(_path("tags"))).json() == {"tags": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["albums", "tags"])
async def test_concurrent_creation_retries_commit_one_choice(
    client: AsyncClient, kind: str
) -> None:
    responses = await asyncio.gather(
        client.post(_path(kind), json=_request(kind)),
        client.post(_path(kind), json=_request(kind)),
    )
    assert [response.status_code for response in responses] == [201, 201]
    assert responses[0].json() == responses[1].json()
    key = "collections" if kind == "albums" else "tags"
    assert len((await client.get(_path(kind))).json()[key]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [None, True, 4, "", "short", "x" * 129])
async def test_invalid_creation_keys_refuse_before_writing(
    client: AsyncClient, key: object
) -> None:
    response = await client.post(_path("albums"), json=_request("albums", key=key))
    assert response.status_code == 422
    assert (await client.get(_path("albums"))).json() == {"collections": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["albums", "tags"])
async def test_creation_replay_does_not_resurrect_a_removed_choice(
    client: AsyncClient, kind: str
) -> None:
    first = await _replayed(client, kind)
    table = "media_collections" if kind == "albums" else "media_tags"
    with db.SessionLocal() as session:
        session.execute(text(f"DELETE FROM {table} WHERE id=:id"), {"id": first["id"]})
        session.commit()
    again = await client.post(_path(kind), json=_request(kind))
    assert again.status_code == 201
    assert again.json() == first
    key = "collections" if kind == "albums" else "tags"
    assert (await client.get(_path(kind))).json()[key] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["albums", "tags"])
@pytest.mark.parametrize(
    "damage", ["extra-field", "wrong-id", "wrong-name", "wrong-version", "seal"]
)
async def test_damaged_creation_receipts_refuse_replay_without_a_new_write(
    client: AsyncClient, kind: str, damage: str
) -> None:
    first = await _replayed(client, kind)
    response = dict(first)
    if damage == "extra-field":
        response["unexpected"] = "neutral stored marker"
    elif damage == "wrong-id":
        prefix = "collection_" if kind == "albums" else "mediatag_"
        response["id"] = prefix + "0" * 32
    elif damage == "wrong-name":
        response["name" if kind == "albums" else "label"] = "Different"
    elif damage == "wrong-version":
        response["version"] = 99
    with db.SessionLocal() as session:
        if damage == "seal":
            session.execute(
                text(
                    "UPDATE media_organization_creations SET response_sha256=:digest "
                    "WHERE operation_key=:key"
                ),
                {"key": KEY, "digest": "0" * 64},
            )
        else:
            session.execute(
                text(
                    "UPDATE media_organization_creations SET response_json=:response "
                    "WHERE operation_key=:key"
                ),
                {"key": KEY, "response": json.dumps(response)},
            )
        session.commit()
    replay = await client.post(_path(kind), json=_request(kind))
    assert replay.status_code == 409
    assert replay.json()["code"] == "media-creation-conflict"
    assert "neutral stored marker" not in replay.text
    key = "collections" if kind == "albums" else "tags"
    assert len((await client.get(_path(kind))).json()[key]) == 1
