"""Organizing a deleted conversation's media changes neither its size nor how it leaves."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import Result, delete, func, select
from sqlalchemy.orm import Session
from test_chat_recovery import _chat, _command, _impact
from test_chat_recovery_graph import _graph, _readers
from test_chat_recovery_graph import graph_session as graph_session
from test_chat_recovery_media_atomic import _two_generated_images
from test_chat_recovery_transitions import _trash as _trash_garden
from test_media_organization_batches import SelectedMedia
from test_media_organization_batches import _apply as _apply_organization
from test_media_organization_batches import _impact as _organization_impact
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm import chat_recovery_graph, recovery_maintenance
from local_lm.db import SessionLocal
from local_lm.domain import new_id
from local_lm.models import (
    ArtifactLibraryEntry,
    Chat,
    MediaCollection,
    MediaCollectionMembership,
    MediaTag,
    MediaTagAssignment,
    RecoveryItem,
)
from local_lm.recovery_v1 import RecoveryCountsV1

GARDEN_ENTRY = "libentry:sha256:" + f"{1:064x}"


def _album(session: Session, index: int) -> None:
    album = MediaCollection(id=new_id("collection"), kind="manual", name=f"album {index}")
    session.add(album)
    session.flush()
    session.add(
        MediaCollectionMembership(collection_id=album.id, entry_id=GARDEN_ENTRY, position=0)
    )
    session.flush()


def _tag(session: Session, index: int) -> None:
    tag = MediaTag(id=new_id("mediatag"), slug=f"tag-{index}", label=f"tag {index}")
    session.add(tag)
    session.flush()
    session.add(MediaTagAssignment(tag_id=tag.id, entry_id=GARDEN_ENTRY))
    session.flush()


def _held(session: Session) -> list[int | None]:
    return [
        session.scalar(select(func.count()).where(column == GARDEN_ENTRY))
        for column in (MediaCollectionMembership.entry_id, MediaTagAssignment.entry_id)
    ]


def test_organizing_a_deleted_conversations_media_never_grows_its_graph(
    graph_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _trash_garden(graph_session)
    readers = _readers(monkeypatch)

    def measured() -> tuple[int, int, int, RecoveryCountsV1]:
        counts = _graph(graph_session).snapshot.counts
        graph_session.commit()
        reader = readers[-1]
        return reader.row_count, reader.byte_count, reader.read_bytes, counts

    before = measured()
    for index in range(1, 4):
        _album(graph_session, index)
        _tag(graph_session, index)
    graph_session.commit()
    assert _held(graph_session) == [3, 3]
    assert measured() == before


def test_only_a_change_in_protection_changes_the_private_revision(graph_session: Session) -> None:
    def fingerprint() -> str:
        value = _graph(graph_session).snapshot.fingerprint
        graph_session.commit()
        return value

    unorganized = fingerprint()
    _tag(graph_session, 1)
    graph_session.commit()
    protected = fingerprint()
    assert protected != unorganized
    _tag(graph_session, 2)
    _album(graph_session, 1)
    graph_session.commit()
    assert fingerprint() == protected
    graph_session.execute(delete(MediaCollectionMembership))
    graph_session.execute(delete(MediaTagAssignment))
    graph_session.commit()
    assert _held(graph_session) == [0, 0]
    assert fingerprint() == unorganized


def test_a_fact_named_like_a_column_refuses_the_read(graph_session: Session) -> None:
    # A fact adds to a row; standing in for one of its columns would hide a change to it.
    reader = chat_recovery_graph._GraphReader(graph_session)
    with pytest.raises(chat_recovery_graph.ChatRecoveryGraphError):
        reader.read(
            "artifact_library_entries",
            ArtifactLibraryEntry.id == GARDEN_ENTRY,
            facts=lambda _row: {"id": 0},
        )


def test_a_read_refused_partway_leaves_no_statement_open(
    graph_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    results: list[Result[Any]] = []
    execute = graph_session.execute

    def recording(*args: Any, **kwargs: Any) -> Result[Any]:
        result = execute(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(graph_session, "execute", recording)
    monkeypatch.setattr(chat_recovery_graph, "MAX_GRAPH_ROWS", 0)
    reader = chat_recovery_graph._GraphReader(graph_session)
    # Two rows match, so the refusal at the first leaves the second unread.
    with pytest.raises(chat_recovery_graph.ChatRecoveryGraphTooLarge):
        reader.read("chats", Chat.id.in_(["chat-garden", "chat-other"]))
    assert len(results) == 1
    assert results[0].closed


async def _trash(
    client: AsyncClient, chat_id: str, key: str, *, generated_media: bool = False
) -> dict[str, Any]:
    query = "?delete_generated_media=true" if generated_media else ""
    preview = await _impact(client, f"/api/chats/{chat_id}/deletion-impact{query}")
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash",
        json={**_command(preview, key), "delete_generated_media": generated_media},
    )
    assert trashed.status_code == 200, trashed.text
    result: dict[str, Any] = trashed.json()
    return result


async def _admitted_with_the_least_room(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, chat_id: str
) -> int:
    """Bound conversations so this one only just enters Recently Deleted; return its room."""
    with monkeypatch.context() as patch:
        readers = _readers(patch)
        await _impact(client, f"/api/chats/{chat_id}/deletion-impact?delete_generated_media=true")
    rows: int = readers[-1].row_count
    bound = -(-16 * rows // 15)
    monkeypatch.setattr(chat_recovery_graph, "MAX_GRAPH_ROWS", bound)
    return bound - rows


async def _tag_through_the_library(
    client: AsyncClient, images: dict[str, tuple[str, Path, bytes]], count: int
) -> str:
    """Give the first image `count` tags, each its own Media Library change."""
    artifact_id, (entry_id, _path, content) = next(iter(images.items()))
    with SessionLocal() as session:
        entry = session.get(ArtifactLibraryEntry, entry_id)
        assert entry is not None
        selected = SelectedMedia(entry_id, artifact_id, entry.version, content)
    for index in range(1, count + 1):
        created = await client.post("/api/media-tags", json={"label": f"tag {index}"})
        assert created.status_code == 201, created.text
        impact = await _organization_impact(client, "add-tag", created.json(), [selected])
        await _apply_organization(client, impact, f"tag-{index}")
    with SessionLocal() as session:
        assigned = session.scalar(
            select(func.count()).where(MediaTagAssignment.entry_id == entry_id)
        )
    assert assigned == count
    return entry_id


@pytest.mark.parametrize("removal", ["purge", "batch_purge", "restore_then_delete_again"])
async def test_a_deleted_conversation_organized_past_its_size_bound_stays_listed_and_can_be_removed(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, removal: str
) -> None:
    chat_id, images = await _two_generated_images(client, app)
    room = await _admitted_with_the_least_room(client, monkeypatch, chat_id)
    plain = await _trash(client, await _chat(client), "trash-plain")
    organized = await _trash(client, chat_id, "trash-organized", generated_media=True)
    # Counted as rows, these tags and the deletion record would put it one row past its bound.
    tagged = await _tag_through_the_library(client, images, room)

    page = await client.get("/api/recovery-items")
    assert page.status_code == 200, page.text
    assert {item["deletion_id"]: item for item in page.json()["items"]} == {
        item["deletion_id"]: item for item in (organized, plain)
    }
    deletion_id = organized["deletion_id"]
    if removal == "purge":
        impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
        purged = await client.post(
            f"/api/recovery-items/{deletion_id}/purge",
            json={**_command(impact, "purge-organized"), "acknowledgement": "permanently-delete"},
        )
        assert purged.status_code == 200, purged.text
        with SessionLocal() as session:
            assert session.get(Chat, chat_id) is None
            for entry_id, _path, _content in images.values():
                kept = session.get(ArtifactLibraryEntry, entry_id) is not None
                assert kept == (entry_id == tagged)
    elif removal == "batch_purge":
        selection = {"deletion_ids": [deletion_id, plain["deletion_id"]], "action": "purge"}
        preview = await client.post("/api/recovery-items/batches", json=selection)
        assert preview.status_code == 200, preview.text
        assert preview.json()["available"] is True
        applied = await client.post(
            f"/api/recovery-items/batches/{preview.json()['batch_id']}/apply",
            json={
                **_command(preview.json(), "purge-both"),
                "acknowledgement": "permanently-delete",
            },
        )
        assert applied.status_code == 200, applied.text
        with SessionLocal() as session:
            assert session.get(Chat, chat_id) is None
            assert session.get(Chat, plain["subject_id"]) is None
    else:
        impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
        restored = await client.post(
            f"/api/recovery-items/{deletion_id}/restore", json=_command(impact, "restore-organized")
        )
        assert restored.status_code == 200, restored.text
        await _trash(client, chat_id, "trash-organized-again", generated_media=True)


async def test_expiry_purges_a_conversation_organized_past_its_size_bound(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, images = await _two_generated_images(client, app)
    room = await _admitted_with_the_least_room(client, monkeypatch, chat_id)
    organized = await _trash(client, chat_id, "trash-organized", generated_media=True)
    await _tag_through_the_library(client, images, room)
    deadline = datetime.fromisoformat(organized["purge_after"])

    expired = await recovery_maintenance.expire_recovery_batch(app.state.services, deadline)
    assert (expired.examined, expired.purged, expired.deferred) == (1, 1, 0)
    with SessionLocal() as session:
        item = session.get(RecoveryItem, organized["deletion_id"])
        assert item is not None and item.state == "purged"
        assert session.get(Chat, chat_id) is None
    again = await recovery_maintenance.expire_recovery_batch(app.state.services, deadline)
    assert again.examined == 0
