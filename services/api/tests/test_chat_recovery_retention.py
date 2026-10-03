"""Recovery keeps real media bytes pinned until canonical references leave."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_recovery import _command, _impact

from local_lm.artifact_library import ensure_library_entry, set_library_favorite
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    Message,
    MessagePart,
    ReferenceAsset,
    ReferenceSubject,
)


@pytest.mark.parametrize("retained_by", ["other_chat", "reference", "favorite", "poster_parent"])
async def test_recovery_and_retention_preserve_independently_retained_media_bytes(
    client: AsyncClient, settings: Settings, retained_by: str
) -> None:
    response = await client.post("/api/chats", json={"title": "Garden media to recover"})
    assert response.status_code == 201
    chat_id = str(response.json()["id"])
    store = ArtifactStore(settings)
    content = b"\x89PNG\r\n\x1a\nconstructed garden media "
    with SessionLocal() as session:
        media = [
            store.ingest_bytes(
                session, content + name.encode(), kind=ArtifactKind.IMAGE, media_type="image/png"
            )
            for name in ("retained", "only-in-chat")
        ]
        protected, private = media
        message = Message(chat_id=chat_id, role="assistant", status="complete")
        session.add(message)
        session.flush()
        session.add_all(
            MessagePart(
                message_id=message.id, position=index, type="image", artifact_id=artifact.id
            )
            for index, artifact in enumerate(media)
        )
        if retained_by == "other_chat":
            other = Chat(title="Independent archived garden chat", archived=True)
            session.add(other)
            session.flush()
            other_message = Message(chat_id=other.id, role="assistant", status="complete")
            session.add(other_message)
            session.flush()
            session.add(
                MessagePart(
                    message_id=other_message.id, position=0, type="image", artifact_id=protected.id
                )
            )
        elif retained_by == "reference":
            subject = ReferenceSubject(
                name="Garden structure", mention_slug="garden-structure", kind="object"
            )
            session.add(subject)
            session.flush()
            session.add(
                ReferenceAsset(
                    reference_subject_id=subject.id, artifact_id=protected.id, purpose="other"
                )
            )
        elif retained_by == "favorite":
            set_library_favorite(session, protected, True)
        else:
            parent = store.ingest_bytes(
                session,
                content + b"retained-parent",
                kind=ArtifactKind.IMAGE,
                media_type="image/png",
                metadata={"poster_artifact_id": protected.id},
            )
            assert ensure_library_entry(session, parent) is not None
        session.commit()
        ids = [artifact.id for artifact in media]
        paths = [store.resolve(artifact) for artifact in media]
        bytes_before = [path.read_bytes() for path in paths]
        assert (
            session.scalar(
                select(ArtifactLibraryEntry.id).where(
                    ArtifactLibraryEntry.artifact_id == private.id
                )
            )
            is None
        )

    impact = await _impact(
        client, f"/api/chats/{chat_id}/deletion-impact?delete_generated_media=true"
    )
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash",
        json={
            **_command(impact, "trash-retained-garden-media"),
            "delete_generated_media": True,
        },
    )
    assert trashed.status_code == 200, trashed.text
    assert (await client.get(f"/api/chats/{chat_id}")).status_code == 404
    later = datetime.now(UTC) + timedelta(days=120)
    with SessionLocal() as session:
        retained = store.cleanup_retention(
            session, retention_days=1, temporary_hours=1, dry_run=False, now=later
        )
        session.commit()
        assert retained.marked_count == retained.removed_count == retained.reclaimed_bytes == 0
        assert [path.read_bytes() for path in paths] == bytes_before
        assert session.get(Chat, chat_id) is not None

    deletion_id = str(trashed.json()["deletion_id"])
    purge_impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    purged = await client.post(
        f"/api/recovery-items/{deletion_id}/purge",
        json={
            **_command(purge_impact, "purge-retained-garden-media"),
            "acknowledgement": "permanently-delete",
        },
    )
    assert purged.status_code == 200, purged.text
    assert purged.json()["reclaimed_bytes"] == 0
    assert [path.read_bytes() for path in paths] == bytes_before
    with SessionLocal() as session:
        marked = store.cleanup_retention(
            session, retention_days=1, temporary_hours=1, dry_run=False, now=later
        )
        session.commit()
        assert marked.marked_count == 1
        assert marked.removed_count == marked.reclaimed_bytes == 0
        assert [path.read_bytes() for path in paths] == bytes_before
        swept = store.cleanup_retention(
            session,
            retention_days=1,
            temporary_hours=1,
            dry_run=False,
            now=later + timedelta(days=1, microseconds=1),
        )
        session.commit()
        assert swept.removed_count == 1
        assert swept.reclaimed_bytes == len(bytes_before[1])
        assert session.get(Artifact, ids[1]) is None
        assert not paths[1].exists()
        assert session.get(Artifact, ids[0]) is not None
        assert paths[0].read_bytes() == bytes_before[0]
