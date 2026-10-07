"""Chat media purge intent releases only exclusive, unprotected library membership."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_deletion import _image_exchange
from test_chat_recovery import _chat, _command, _history, _impact
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm.artifact_library import ensure_library_entry, set_library_favorite
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind, new_id
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    MediaCollection,
    MediaCollectionMembership,
    MediaTag,
    MediaTagAssignment,
    Message,
    MessagePart,
    ReferenceAsset,
    ReferenceSubject,
)


@pytest.mark.parametrize("delete_generated_media", [False, True])
@pytest.mark.parametrize(
    "retained_by",
    [
        "none",
        "favorite",
        "other_chat",
        "reference",
        "poster_parent",
        "collection",
        "tag",
        "uploaded",
    ],
)
async def test_chat_media_intent_waits_for_purge_and_respects_independent_retention(
    client: AsyncClient, app: FastAPI, delete_generated_media: bool, retained_by: str
) -> None:
    chat_id = await _chat(client)
    await _image_exchange(client, chat_id, "Create a picture of garden beds")
    history = _history(chat_id)
    artifact_id = next(part[-1] for part in history["parts"] if part[-1])
    store = app.state.services.artifacts
    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact is not None
        entry = session.scalar(
            select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == artifact_id)
        )
        assert entry is not None
        entry_id = entry.id
        path = store.resolve(artifact)
        content = path.read_bytes()
        if retained_by == "favorite":
            set_library_favorite(session, artifact, True)
        elif retained_by == "other_chat":
            other = Chat(title="Another garden notebook")
            session.add(other)
            session.flush()
            message = Message(chat_id=other.id, role="assistant", status="complete")
            session.add(message)
            session.flush()
            session.add(
                MessagePart(
                    message_id=message.id, position=0, type="image", artifact_id=artifact_id
                )
            )
        elif retained_by == "reference":
            subject = ReferenceSubject(
                name="Garden beds", mention_slug="garden-beds", kind="object"
            )
            session.add(subject)
            session.flush()
            session.add(
                ReferenceAsset(
                    reference_subject_id=subject.id, artifact_id=artifact_id, purpose="other"
                )
            )
        elif retained_by == "poster_parent":
            parent = store.ingest_bytes(
                session,
                content + b"independent garden parent",
                kind=ArtifactKind.IMAGE,
                media_type="image/png",
                metadata={"poster_artifact_id": artifact_id},
            )
            assert ensure_library_entry(session, parent) is not None
        elif retained_by == "collection":
            collection = MediaCollection(
                id=new_id("collection"), kind="manual", name="Garden collection"
            )
            session.add(collection)
            session.flush()
            session.add(
                MediaCollectionMembership(
                    collection_id=collection.id, entry_id=entry_id, position=0
                )
            )
        elif retained_by == "tag":
            tag = MediaTag(id=new_id("mediatag"), slug="garden", label="Garden")
            session.add(tag)
            session.flush()
            session.add(MediaTagAssignment(tag_id=tag.id, entry_id=entry_id))
        session.commit()

    if retained_by == "uploaded":
        uploaded = await client.post(
            "/api/artifacts", files={"file": ("garden-copy.png", content, "image/png")}
        )
        assert uploaded.status_code == 201 and uploaded.json()["id"] == artifact_id

    preview = await _impact(
        client,
        f"/api/chats/{chat_id}/deletion-impact?delete_generated_media={str(delete_generated_media).lower()}",
    )
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash",
        json={
            **_command(preview, "trash-garden-media-intent"),
            "delete_generated_media": delete_generated_media,
        },
    )
    assert trashed.status_code == 200, trashed.text
    deletion_id = trashed.json()["deletion_id"]
    with SessionLocal() as session:
        assert session.get(ArtifactLibraryEntry, entry_id) is not None
    assert _history(chat_id) == history
    assert path.read_bytes() == content

    preview = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    purged = await client.post(
        f"/api/recovery-items/{deletion_id}/purge",
        json={
            **_command(preview, "purge-garden-media-intent"),
            "acknowledgement": "permanently-delete",
        },
    )
    assert purged.status_code == 200, purged.text
    assert purged.json()["reclaimed_bytes"] == 0
    released = delete_generated_media and retained_by == "none"
    with SessionLocal() as session:
        assert (session.get(ArtifactLibraryEntry, entry_id) is None) is released
        assert session.get(Artifact, artifact_id) is not None
        if retained_by == "favorite":
            entry = session.get(ArtifactLibraryEntry, entry_id)
            assert entry is not None and entry.favorite
        later = datetime.now(UTC) + timedelta(days=120)
        store.cleanup_retention(
            session, retention_days=1, temporary_hours=1, dry_run=False, now=later
        )
        session.commit()
        assert path.read_bytes() == content
        store.cleanup_retention(
            session,
            retention_days=1,
            temporary_hours=1,
            dry_run=False,
            now=later + timedelta(days=1, microseconds=1),
        )
        session.commit()
        assert (session.get(Artifact, artifact_id) is None) is released
    if released:
        assert not path.exists()
    else:
        assert path.read_bytes() == content
