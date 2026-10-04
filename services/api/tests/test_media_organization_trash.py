"""A selected library trash operation keeps the canonical recovery contract."""

from dataclasses import replace

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from test_media_organization_batches import _album, _apply, _impact, _media, _tag

from local_lm import db
from local_lm.artifact_library import set_library_favorite
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.domain import utcnow
from local_lm.media_recovery import preview_media_recovery, restore_media
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    MediaCollectionMembership,
    MediaOrganizationImpact,
    MediaTagAssignment,
    RecoveryItem,
    RecoveryOperation,
)
from local_lm.recovery_v1 import RestoreRecoveryV1


async def test_selected_trash_keeps_favorites_organization_bytes_and_restorable_identity(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings, 3)
    with db.SessionLocal() as session:
        for index, selected in enumerate(media):
            artifact = session.get(Artifact, selected.artifact_id)
            assert artifact is not None
            set_library_favorite(session, artifact, True)
            session.flush()
            entry = session.get(ArtifactLibraryEntry, selected.entry_id)
            assert entry is not None
            media[index] = replace(selected, version=entry.version)
        session.commit()
    album, tag = await _album(client), await _tag(client)
    await _apply(client, await _impact(client, "add-to-album", album, media), "organize-album")
    await _apply(client, await _impact(client, "add-tag", tag, media), "organize-tag")
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": "trash",
            "entries": [item.selection() for item in media[:2]],
        },
    )
    assert response.status_code == 201, response.text
    impact = response.json()
    assert impact["selected_count"] == impact["changed_count"] == 2
    with db.SessionLocal() as session:
        assert session.scalars(select(RecoveryItem)).all() == []
    applied = await _apply(client, impact, "trash-selected-once")
    assert await _apply(client, impact, "trash-selected-once") == applied
    deletion_ids = applied["deletion_ids"]
    assert isinstance(deletion_ids, list) and all(isinstance(item, str) for item in deletion_ids)
    assert len(set(deletion_ids)) == 2
    assert "display_label" not in str(applied)
    store = ArtifactStore(settings)
    with db.SessionLocal() as session:
        for index, selected in enumerate(media):
            entry = session.get(ArtifactLibraryEntry, selected.entry_id)
            artifact = session.get(Artifact, selected.artifact_id)
            assert entry is not None and artifact is not None
            assert entry.state == ("trashed" if index < 2 else "visible")
            assert entry.favorite and artifact.favorite
            assert store.resolve(artifact).read_bytes() == selected.content
            assert session.get(MediaCollectionMembership, (album["id"], entry.id)) is not None
            assert session.get(MediaTagAssignment, (tag["id"], entry.id)) is not None
        for index, deletion_id in enumerate(deletion_ids):
            now = utcnow()
            preview = preview_media_recovery(session, deletion_id, now)
            session.commit()
            restored = restore_media(
                session,
                deletion_id,
                RestoreRecoveryV1(
                    expected_revision=preview.revision,
                    impact_sha256=preview.impact_sha256,
                    operation_key=f"restore-selected-{index}",
                ),
                now,
            )
            session.commit()
            assert restored.subject_id == media[index].entry_id
            entry = session.get(ArtifactLibraryEntry, restored.subject_id)
            assert entry is not None and entry.state == "visible" and entry.favorite


async def test_a_later_changed_recovery_graph_rolls_back_the_entire_trash_selection(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings)
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": "trash",
            "entries": [item.selection() for item in media],
        },
    )
    assert response.status_code == 201, response.text
    impact = response.json()
    album = await _album(client)
    await _apply(
        client, await _impact(client, "add-to-album", album, media[1:]), "change-last-graph"
    )
    response = await client.post(
        f"/api/media-organization/impacts/{impact['id']}/apply",
        json={
            "operation_key": "refuse-stale-trash",
        },
    )
    assert response.status_code == 409, response.text
    with db.SessionLocal() as session:
        assert session.scalars(select(RecoveryItem)).all() == []
        assert session.scalars(select(RecoveryOperation)).all() == []
        entries = session.scalars(select(ArtifactLibraryEntry)).all()
        assert len(entries) == 2 and all(entry.state == "visible" for entry in entries)
        record = session.get(MediaOrganizationImpact, impact["id"])
        assert record is not None and record.operation_key is None
        assert record.response_json is None
        assert session.scalars(select(MediaCollectionMembership.entry_id)).all() == [
            media[1].entry_id
        ]


async def test_another_hidden_selection_member_refuses_before_any_recovery_write(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings)
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": "trash",
            "entries": [item.selection() for item in media],
        },
    )
    assert response.status_code == 201, response.text
    original = response.json()
    competing = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": "trash",
            "entries": [media[1].selection()],
        },
    )
    assert competing.status_code == 201, competing.text
    await _apply(client, competing.json(), "hide-second-only")
    response = await client.post(
        f"/api/media-organization/impacts/{original['id']}/apply",
        json={
            "operation_key": "refuse-hidden-member",
        },
    )
    assert response.status_code == 409
    with db.SessionLocal() as session:
        items = session.scalars(select(RecoveryItem)).all()
        assert len(items) == 1 and items[0].subject_id == media[1].entry_id
        entry = session.get(ArtifactLibraryEntry, media[0].entry_id)
        assert entry is not None and entry.state == "visible"


@pytest.mark.parametrize("proof", [None, {"invalid": {}}, {"expected_revision": "not-a-digest"}])
async def test_malformed_stored_recovery_proofs_never_hide_selected_media(
    client: AsyncClient,
    settings: Settings,
    proof: object,
) -> None:
    media = _media(settings, 1)
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": "trash",
            "entries": [media[0].selection()],
        },
    )
    assert response.status_code == 201, response.text
    impact = response.json()
    with db.SessionLocal() as session:
        record = session.get(MediaOrganizationImpact, impact["id"])
        assert record is not None
        record.recovery_json = None if proof is None else {media[0].entry_id: proof}
        session.commit()
    response = await client.post(
        f"/api/media-organization/impacts/{impact['id']}/apply",
        json={
            "operation_key": "refuse-malformed-proof",
        },
    )
    assert response.status_code == 409
    with db.SessionLocal() as session:
        assert session.scalars(select(RecoveryItem)).all() == []
        entry = session.get(ArtifactLibraryEntry, media[0].entry_id)
        assert entry is not None and entry.state == "visible"
