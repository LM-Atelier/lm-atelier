"""Selected library items change only through a versioned, replayable impact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from httpx2 import AsyncClient
from sqlalchemy import delete, select

from local_lm import db
from local_lm.artifact_library import ensure_library_entry
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.domain import ArtifactKind
from local_lm.models import ArtifactLibraryEntry, MediaCollectionMembership, MediaTagAssignment


@dataclass(frozen=True)
class SelectedMedia:
    entry_id: str
    artifact_id: str
    version: int
    content: bytes

    def selection(self) -> dict[str, object]:
        return {"id": self.entry_id, "version": self.version}


def _media(settings: Settings, count: int = 2) -> list[SelectedMedia]:
    store = ArtifactStore(settings)
    result = []
    with db.SessionLocal() as session:
        for index in range(count):
            content = f"neutral organization image {index}".encode("ascii")
            artifact = store.ingest_bytes(
                session,
                content,
                kind=ArtifactKind.IMAGE,
                media_type="image/png",
                original_name=f"study-{index}.png",
            )
            entry = ensure_library_entry(session, artifact)
            assert entry is not None
            session.flush()
            result.append(SelectedMedia(entry.id, artifact.id, entry.version, content))
        session.commit()
    return result


async def _album(client: AsyncClient) -> dict[str, object]:
    response = await client.post("/api/media-collections", json={"name": "Studies"})
    assert response.status_code == 201
    return cast(dict[str, object], response.json())


async def _tag(client: AsyncClient) -> dict[str, object]:
    response = await client.post("/api/media-tags", json={"label": "Study"})
    assert response.status_code == 201
    return cast(dict[str, object], response.json())


async def _impact(
    client: AsyncClient,
    action: str,
    target: dict[str, object],
    media: list[SelectedMedia],
) -> dict[str, object]:
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": action,
            "target": {"id": target["id"], "version": target["version"]},
            "entries": [entry.selection() for entry in media],
        },
    )
    assert response.status_code == 201, response.text
    body = cast(dict[str, object], response.json())
    assert body["selected_count"] == len(media)
    assert body["size_bytes"] == sum(len(entry.content) for entry in media)
    return body


async def _apply(client: AsyncClient, impact: dict[str, object], key: str) -> dict[str, object]:
    response = await client.post(
        f"/api/media-organization/impacts/{impact['id']}/apply",
        json={"operation_key": key},
    )
    assert response.status_code == 200, response.text
    return cast(dict[str, object], response.json())


async def test_album_impact_is_read_only_until_applied_and_replays_exactly(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings)
    album = await _album(client)
    impact = await _impact(client, "add-to-album", album, media)
    assert impact["changed_count"] == 2
    with db.SessionLocal() as session:
        assert session.scalars(select(MediaCollectionMembership)).all() == []

    applied = await _apply(client, impact, "album-add-once")
    replay = await _apply(client, impact, "album-add-once")
    assert replay == applied
    with db.SessionLocal() as session:
        memberships = session.scalars(
            select(MediaCollectionMembership).order_by(MediaCollectionMembership.position)
        ).all()
        assert [(item.entry_id, item.position) for item in memberships] == [
            (media[0].entry_id, 0),
            (media[1].entry_id, 1),
        ]
        assert set(session.scalars(select(ArtifactLibraryEntry.id)).all()) == {
            item.entry_id for item in media
        }
    for item in media:
        response = await client.get(f"/api/artifacts/{item.artifact_id}/content")
        assert response.status_code == 200
        assert response.content == item.content


@pytest.mark.parametrize("kind", ["album", "tag"])
async def test_removing_organization_preserves_selected_artifacts(
    client: AsyncClient,
    settings: Settings,
    kind: str,
) -> None:
    media = _media(settings)
    target = await (_album(client) if kind == "album" else _tag(client))
    add_action = "add-to-album" if kind == "album" else "add-tag"
    remove_action = "remove-from-album" if kind == "album" else "remove-tag"
    added = await _apply(client, await _impact(client, add_action, target, media), "add-selected")
    target["version"] = added["target_version"]
    removed = await _apply(
        client, await _impact(client, remove_action, target, media), "remove-selected"
    )
    assert removed["changed_count"] == 2
    with db.SessionLocal() as session:
        assert session.scalars(select(MediaCollectionMembership)).all() == []
        assert session.scalars(select(MediaTagAssignment)).all() == []
        assert len(session.scalars(select(ArtifactLibraryEntry)).all()) == 2
    for item in media:
        response = await client.get(f"/api/artifacts/{item.artifact_id}/content")
        assert response.content == item.content


async def test_an_album_changed_after_preview_refuses_the_whole_batch(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings, 3)
    album = await _album(client)
    original = await _impact(client, "add-to-album", album, media[:2])
    competing = await _impact(client, "add-to-album", album, media[2:])
    await _apply(client, competing, "competing-album-write")
    response = await client.post(
        f"/api/media-organization/impacts/{original['id']}/apply",
        json={"operation_key": "stale-album-write"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "media-organization-stale"
    with db.SessionLocal() as session:
        assert session.scalars(select(MediaCollectionMembership.entry_id)).all() == [
            media[2].entry_id
        ]


async def test_a_selected_entry_changed_after_preview_refuses_every_membership(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings)
    album = await _album(client)
    impact = await _impact(client, "add-to-album", album, media)
    response = await client.patch(f"/api/artifacts/{media[1].artifact_id}", json={"favorite": True})
    assert response.status_code == 200
    response = await client.post(
        f"/api/media-organization/impacts/{impact['id']}/apply",
        json={"operation_key": "stale-selected-entry"},
    )
    assert response.status_code == 409
    with db.SessionLocal() as session:
        assert session.scalars(select(MediaCollectionMembership)).all() == []


async def test_an_operation_key_cannot_apply_another_materialized_selection(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings)
    album = await _album(client)
    first = await _impact(client, "add-to-album", album, media[:1])
    second = await _impact(client, "add-to-album", album, media[1:])
    await _apply(client, first, "shared-operation-key")
    response = await client.post(
        f"/api/media-organization/impacts/{second['id']}/apply",
        json={"operation_key": "shared-operation-key"},
    )
    assert response.status_code == 409
    with db.SessionLocal() as session:
        assert session.scalars(select(MediaCollectionMembership.entry_id)).all() == [
            media[0].entry_id
        ]


@pytest.mark.parametrize("mutation", ["duplicate", "unknown-action", "unknown-field", "stale"])
async def test_invalid_selection_never_creates_an_organization_change(
    client: AsyncClient,
    settings: Settings,
    mutation: str,
) -> None:
    media = _media(settings)
    album = await _album(client)
    entries = [entry.selection() for entry in media]
    payload: dict[str, object] = {
        "action": "add-to-album",
        "target": {"id": album["id"], "version": album["version"]},
        "entries": entries,
    }
    if mutation == "duplicate":
        entries.append(entries[0])
    elif mutation == "unknown-action":
        payload["action"] = "delete-bytes"
    elif mutation == "unknown-field":
        payload["sql"] = "SELECT 1"
    else:
        entries[0]["version"] = 999
    response = await client.post("/api/media-organization/impacts", json=payload)
    assert response.status_code == (409 if mutation == "stale" else 422)
    with db.SessionLocal() as session:
        assert session.scalars(select(MediaCollectionMembership)).all() == []


async def test_reordering_an_album_preserves_the_media_and_membership_order(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings, 3)
    album = await _album(client)
    added = await _apply(client, await _impact(client, "add-to-album", album, media), "add-order")
    album["version"] = added["target_version"]
    reordered = await _apply(
        client, await _impact(client, "reorder-album", album, [media[2], media[0]]), "change-order"
    )
    assert reordered["changed_count"] == 2
    with db.SessionLocal() as session:
        members = session.scalars(
            select(MediaCollectionMembership).order_by(MediaCollectionMembership.position)
        ).all()
        assert [item.entry_id for item in members] == [
            media[2].entry_id,
            media[1].entry_id,
            media[0].entry_id,
        ]
        assert [item.position for item in members] == [0, 1, 2]
        assert len(session.scalars(select(ArtifactLibraryEntry)).all()) == 3


@pytest.mark.parametrize("kind", ["album", "tag"])
async def test_deleting_organization_previews_its_exact_references_and_keeps_media(
    client: AsyncClient,
    settings: Settings,
    kind: str,
) -> None:
    media = _media(settings)
    target = await (_album(client) if kind == "album" else _tag(client))
    added = await _apply(
        client,
        await _impact(client, "add-to-album" if kind == "album" else "add-tag", target, media),
        "prepare-organized-media",
    )
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": f"delete-{kind}",
            "target": {"id": target["id"], "version": added["target_version"]},
        },
    )
    assert response.status_code == 201, response.text
    impact = response.json()
    assert impact["selected_count"] == 2
    assert impact["size_bytes"] == sum(len(item.content) for item in media)
    applied = await _apply(client, impact, "delete-organization")
    assert await _apply(client, impact, "delete-organization") == applied
    listing = await client.get("/api/media-collections" if kind == "album" else "/api/media-tags")
    assert listing.json()["collections" if kind == "album" else "tags"] == []
    with db.SessionLocal() as session:
        assert len(session.scalars(select(ArtifactLibraryEntry)).all()) == 2
    for item in media:
        assert (
            await client.get(f"/api/artifacts/{item.artifact_id}/content")
        ).content == item.content


@pytest.mark.parametrize("kind", ["album", "tag"])
async def test_rename_is_confirmed_against_the_organization_version(
    client: AsyncClient,
    kind: str,
) -> None:
    target = await (_album(client) if kind == "album" else _tag(client))
    changes = (
        {"name": "Landscape studies", "description": "Selected examples"}
        if kind == "album"
        else {"label": "Landscape study", "color": "#a1b2c3"}
    )
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": f"rename-{kind}",
            "target": {"id": target["id"], "version": target["version"]},
            "changes": changes,
        },
    )
    assert response.status_code == 201, response.text
    applied = await _apply(client, response.json(), "rename-organization")
    assert applied["target_version"] == 2
    listing = await client.get("/api/media-collections" if kind == "album" else "/api/media-tags")
    updated = listing.json()["collections" if kind == "album" else "tags"][0]
    assert updated["id"] == target["id"]
    assert updated["version"] == 2
    for field, value in changes.items():
        assert updated[field] == value
    if kind == "tag":
        assert updated["slug"] == "landscape-study"


async def test_a_batch_favorite_uses_one_impact_and_changes_only_selected_entries(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings, 3)
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": "set-favorite",
            "favorite": True,
            "entries": [item.selection() for item in media[:2]],
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["changed_count"] == 2
    applied = await _apply(client, response.json(), "favorite-selected")
    assert await _apply(client, response.json(), "favorite-selected") == applied
    with db.SessionLocal() as session:
        selected = [session.get(ArtifactLibraryEntry, item.entry_id) for item in media]
        assert all(item is not None for item in selected)
        assert [item.favorite for item in selected if item is not None] == [True, True, False]


async def test_merging_tags_previews_both_versions_and_preserves_overlapping_assignments(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings, 3)
    source = await _tag(client)
    created = await client.post("/api/media-tags", json={"label": "Landscape"})
    assert created.status_code == 201
    destination = created.json()
    source_applied = await _apply(
        client, await _impact(client, "add-tag", source, media[:2]), "tag-source"
    )
    destination_applied = await _apply(
        client, await _impact(client, "add-tag", destination, media[1:]), "tag-destination"
    )
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": "merge-tags",
            "target": {"id": source["id"], "version": source_applied["target_version"]},
            "destination": {
                "id": destination["id"],
                "version": destination_applied["target_version"],
            },
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["selected_count"] == 3
    assert response.json()["size_bytes"] == sum(len(item.content) for item in media)
    applied = await _apply(client, response.json(), "merge-tag-selection")
    assert await _apply(client, response.json(), "merge-tag-selection") == applied
    listing = await client.get("/api/media-tags")
    assert [item["id"] for item in listing.json()["tags"]] == [destination["id"]]
    with db.SessionLocal() as session:
        assignments = session.scalars(select(MediaTagAssignment)).all()
        assert {(item.tag_id, item.entry_id) for item in assignments} == {
            (destination["id"], item.entry_id) for item in media
        }
        assert len(session.scalars(select(ArtifactLibraryEntry)).all()) == 3


async def test_album_and_tag_filters_are_applied_before_the_page_limit(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings, 4)
    album = await _album(client)
    tag = await _tag(client)
    await _apply(client, await _impact(client, "add-to-album", album, media[:3]), "filtered-album")
    await _apply(client, await _impact(client, "add-tag", tag, media[1:]), "filtered-tag")
    parameters: dict[str, str | int] = {
        "collection_id": cast(str, album["id"]),
        "tag_id": cast(str, tag["id"]),
        "limit": 1,
    }
    first = await client.get("/api/artifact-library", params=parameters)
    assert first.status_code == 200
    assert [item["id"] for item in first.json()["items"]] == [media[1].entry_id]
    assert first.json()["items"][0]["collection_position"] == 1
    cursor = first.json()["next_cursor"]
    assert isinstance(cursor, str)
    second = await client.get("/api/artifact-library", params={**parameters, "cursor": cursor})
    assert second.status_code == 200
    assert [item["id"] for item in second.json()["items"]] == [media[2].entry_id]
    assert second.json()["items"][0]["collection_position"] == 2
    assert second.json()["next_cursor"] is None
    cross_filter = await client.get(
        "/api/artifact-library",
        params={"limit": 1, "cursor": cursor, "collection_id": cast(str, album["id"])},
    )
    assert cross_filter.status_code == 422


async def test_a_cursor_refuses_an_album_changed_between_pages(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings, 3)
    album = await _album(client)
    added = await _apply(client, await _impact(client, "add-to-album", album, media), "page-album")
    parameters: dict[str, str | int] = {"collection_id": cast(str, album["id"]), "limit": 1}
    first = await client.get("/api/artifact-library", params=parameters)
    assert first.status_code == 200
    cursor = first.json()["next_cursor"]
    assert isinstance(cursor, str)
    album["version"] = added["target_version"]
    await _apply(
        client, await _impact(client, "remove-from-album", album, media[:1]), "change-page-album"
    )
    response = await client.get("/api/artifact-library", params={**parameters, "cursor": cursor})
    assert response.status_code == 422
    assert response.json()["code"] == "artifact-library-cursor-invalid"


async def test_album_pages_follow_saved_positions_after_a_reviewed_reorder(
    client: AsyncClient, settings: Settings
) -> None:
    media = _media(settings, 3)
    album = await _album(client)
    added = await _apply(
        client, await _impact(client, "add-to-album", album, media), "ordered-album-add"
    )
    parameters: dict[str, str | int] = {"collection_id": cast(str, album["id"]), "limit": 1}
    first = await client.get("/api/artifact-library", params=parameters)
    assert first.status_code == 200
    assert [item["id"] for item in first.json()["items"]] == [media[0].entry_id]
    cursor = first.json()["next_cursor"]
    assert isinstance(cursor, str)
    album["version"] = added["target_version"]
    desired = [media[1], media[2], media[0]]
    await _apply(
        client, await _impact(client, "reorder-album", album, desired), "ordered-album-reorder"
    )
    stale = await client.get("/api/artifact-library", params={**parameters, "cursor": cursor})
    assert stale.status_code == 422
    assert stale.json()["code"] == "artifact-library-cursor-invalid"
    observed = []
    positions = []
    next_cursor = None
    for index in range(3):
        page_parameters = {**parameters, **({"cursor": next_cursor} if next_cursor else {})}
        response = await client.get("/api/artifact-library", params=page_parameters)
        assert response.status_code == 200
        rows = response.json()["items"]
        assert len(rows) == 1
        observed.append(rows[0]["id"])
        positions.append(rows[0]["collection_position"])
        next_cursor = response.json()["next_cursor"]
        assert (next_cursor is not None) == (index < 2)
    assert observed == [item.entry_id for item in desired]
    assert positions == [0, 1, 2]
    all_media = await client.get("/api/artifact-library")
    assert [item["id"] for item in all_media.json()["items"]] == [
        item.entry_id for item in reversed(media)
    ]
    assert all("collection_position" not in item for item in all_media.json()["items"])


async def test_album_pages_refuse_positions_that_cannot_be_represented_exactly(
    client: AsyncClient, settings: Settings
) -> None:
    media = _media(settings, 1)
    album = await _album(client)
    await _apply(client, await _impact(client, "add-to-album", album, media), "position-bound")
    with db.SessionLocal() as session:
        session.execute(
            delete(MediaCollectionMembership).where(
                MediaCollectionMembership.collection_id == album["id"]
            )
        )
        session.add(
            MediaCollectionMembership(
                collection_id=cast(str, album["id"]),
                entry_id=media[0].entry_id,
                position=9_007_199_254_740_992,
            )
        )
        session.commit()
    response = await client.get(
        "/api/artifact-library", params={"collection_id": cast(str, album["id"])}
    )
    assert response.status_code == 409
    assert response.json()["code"] == "artifact-library-conflict"
    assert "9007199254740992" not in response.text
