"""A failed media purge leaves the whole chat recoverable with its original deadline."""

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_chat_deletion import _image_exchange
from test_chat_recovery import _chat, _command, _history, _impact
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm import chat_recovery
from local_lm.artifact_library import ensure_library_entry, referenced_artifact_ids
from local_lm.chat_media_purge import release_generated_membership
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind, new_id
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    MediaCollection,
    MediaCollectionMembership,
    MediaTag,
    MediaTagAssignment,
    MessagePart,
    RecoveryItem,
    RecoveryOperation,
    RecoveryPreviewRecord,
)
from local_lm.recovery_previews import RecoveryPreviewConflict


async def _two_generated_images(
    client: AsyncClient, app: FastAPI
) -> tuple[str, dict[str, tuple[str, Path, bytes]]]:
    chat_id = await _chat(client)
    await _image_exchange(client, chat_id, "Create a picture of garden beds")
    parts = _history(chat_id)["parts"]
    part = next(part for part in parts if part[-1])
    position = max(item[2] for item in parts if item[1] == part[1]) + 1
    images: dict[str, tuple[str, Path, bytes]] = {}
    store = app.state.services.artifacts
    with SessionLocal() as session:
        first = session.get(Artifact, part[-1])
        assert first is not None
        second = store.ingest_bytes(
            session,
            store.resolve(first).read_bytes() + b"another garden image",
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
        )
        session.add(
            MessagePart(message_id=part[1], position=position, type="image", artifact_id=second.id)
        )
        for artifact in (first, second):
            entry = ensure_library_entry(session, artifact)
            assert entry is not None
            path = store.resolve(artifact)
            images[artifact.id] = (entry.id, path, path.read_bytes())
        session.commit()
    return chat_id, images


async def _trash(client: AsyncClient, chat_id: str) -> str:
    preview = await _impact(
        client, f"/api/chats/{chat_id}/deletion-impact?delete_generated_media=true"
    )
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash",
        json={**_command(preview, "trash-two-garden-images"), "delete_generated_media": True},
    )
    assert trashed.status_code == 200, trashed.text
    return str(trashed.json()["deletion_id"])


def _rows() -> dict[str, list[Any]]:
    with SessionLocal() as session:
        return {
            model.__tablename__: list(session.execute(select(model.__table__)).all())
            for model in (
                ArtifactLibraryEntry,
                RecoveryItem,
                RecoveryOperation,
                RecoveryPreviewRecord,
            )
        }


async def test_chat_media_purge_rolls_back_library_removal_and_replays_exactly(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, images = await _two_generated_images(client, app)
    before = _history(chat_id)
    deletion_id = await _trash(client, chat_id)
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    command = {
        **_command(preview, "purge-two-garden-images"),
        "acknowledgement": "permanently-delete",
    }
    collaborator: object = getattr(chat_recovery, "release_generated_membership", None)
    assert collaborator is release_generated_membership
    original = release_generated_membership
    rows = _rows()
    sequence = app.state.services.events.sequence

    def fail_after_release(session: Session, ids: frozenset[str], item: RecoveryItem) -> None:
        original(session, ids, item)
        assert ids == frozenset(images)
        assert all(
            session.get(ArtifactLibraryEntry, values[0]) is None for values in images.values()
        )
        raise RecoveryPreviewConflict("recovery-snapshot-invalid")

    monkeypatch.setattr(chat_recovery, "release_generated_membership", fail_after_release)
    refused = await client.post(f"/api/recovery-items/{deletion_id}/purge", json=command)
    assert refused.status_code == 409, refused.text
    assert _history(chat_id) == before
    assert _rows() == rows
    assert app.state.services.events.sequence == sequence
    assert all(path.read_bytes() == content for _, path, content in images.values())

    monkeypatch.setattr(chat_recovery, "release_generated_membership", original)
    purged = await client.post(f"/api/recovery-items/{deletion_id}/purge", json=command)
    assert purged.status_code == 200, purged.text
    assert purged.json()["reclaimed_bytes"] == 0
    with SessionLocal() as session:
        original_item = session.get(RecoveryItem, deletion_id)
        assert original_item is not None
        members = list(
            session.scalars(select(RecoveryItem).where(RecoveryItem.kind == "media_library_entry"))
        )
        assert len(members) == 2
        assert all(item.state == "purged" for item in members)
        assert all(
            item.deleted_at == original_item.deleted_at
            and item.purge_after == original_item.purge_after
            for item in members
        )
    rows = _rows()
    replayed = await client.post(f"/api/recovery-items/{deletion_id}/purge", json=command)
    assert replayed.status_code == 200 and replayed.content == purged.content
    assert _rows() == rows
    assert all(path.read_bytes() == content for _, path, content in images.values())


@pytest.mark.parametrize("organization", ["collection", "tag"])
async def test_new_library_organization_invalidates_chat_media_purge_preview(
    client: AsyncClient, app: FastAPI, organization: str
) -> None:
    chat_id, images = await _two_generated_images(client, app)
    before = _history(chat_id)
    deletion_id = await _trash(client, chat_id)
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    command = {
        **_command(preview, "purge-organized-garden"),
        "acknowledgement": "permanently-delete",
    }
    protected_entry = next(iter(images.values()))[0]
    with SessionLocal() as session:
        if organization == "collection":
            collection = MediaCollection(id=new_id("collection"), kind="manual", name="Garden")
            session.add(collection)
            session.flush()
            session.add(
                MediaCollectionMembership(
                    collection_id=collection.id, entry_id=protected_entry, position=0
                )
            )
        else:
            tag = MediaTag(id=new_id("mediatag"), slug="garden", label="Garden")
            session.add(tag)
            session.flush()
            session.add(MediaTagAssignment(tag_id=tag.id, entry_id=protected_entry))
        session.commit()
    rows = _rows()
    refused = await client.post(f"/api/recovery-items/{deletion_id}/purge", json=command)
    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "recovery-impact-stale"
    assert _rows() == rows and _history(chat_id) == before
    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    result = await client.post(
        f"/api/recovery-items/{deletion_id}/purge",
        json={
            **_command(preview, "purge-organized-garden-checked"),
            "acknowledgement": "permanently-delete",
        },
    )
    assert result.status_code == 200, result.text
    with SessionLocal() as session:
        for entry_id, _, _ in images.values():
            assert (session.get(ArtifactLibraryEntry, entry_id) is not None) == (
                entry_id == protected_entry
            )


async def test_a_membership_release_preview_cannot_authorize_byte_deletion(
    client: AsyncClient, app: FastAPI
) -> None:
    _, images = await _two_generated_images(client, app)
    with SessionLocal() as session, pytest.raises(ValueError, match="cannot authorize deletion"):
        referenced_artifact_ids(
            session, exclude_library_membership_for=frozenset(images), for_deletion=True
        )
    with SessionLocal() as session:
        assert all(
            session.get(ArtifactLibraryEntry, entry_id) is not None
            for entry_id, _, _ in images.values()
        )
    assert all(path.read_bytes() == content for _, path, content in images.values())
