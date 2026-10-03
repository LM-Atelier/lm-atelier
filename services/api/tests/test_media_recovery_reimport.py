"""Explicit imports recover existing library identity without renewing expired deletion."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_chat_recovery import _command, _impact
from test_media_recovery import CONTENT, _seed

from local_lm.db import SessionLocal
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Job,
    MediaCollectionMembership,
    MediaTagAssignment,
    RecoveryItem,
    RecoveryOperation,
    RecoveryPreviewRecord,
)
from local_lm.recovery_previews import RecoveryPreviewConflict


async def _deleted(app: FastAPI, client: AsyncClient):
    with SessionLocal() as session:
        artifact, entry, collection, tag = _seed(app.state.services.artifacts, session)
        identities = artifact.id, entry.id, collection.id, tag.id
    preview = await _impact(client, f"/api/artifact-library/{identities[1]}/deletion-impact")
    response = await client.post(
        f"/api/artifact-library/{identities[1]}/trash",
        json=_command(preview, "trash-for-import"),
    )
    assert response.status_code == 200
    return identities, response.json()


@pytest.mark.parametrize("kind", ["image", "video", "input"])
async def test_import_restores_the_same_membership_and_organization_after_trash(
    app: FastAPI, client: AsyncClient, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    (artifact_id, entry_id, collection_id, tag_id), item = await _deleted(app, client)
    observed: list[str] = []
    publish = app.state.services.events.publish

    async def observe(event_type: str, entity_id: str | None = None, payload: dict | None = None):
        if event_type == "recovery.updated":
            assert entity_id == item["deletion_id"] and payload == {}
            with SessionLocal() as session:
                assert session.get(RecoveryItem, entity_id) is None
                assert session.get(ArtifactLibraryEntry, entry_id).state == "visible"
            observed.append(entity_id)
        return await publish(event_type, entity_id, payload)

    monkeypatch.setattr(app.state.services.events, "publish", observe)
    for _ in range(2):
        imported = await client.post(
            f"/api/artifacts?kind={kind}",
            files={"file": ("Reimported garden.png", CONTENT, "image/png")},
        )
        assert imported.status_code == 201, imported.text
        assert imported.json()["id"] == artifact_id
        assert imported.json()["favorite"] is True
        assert (await client.get("/api/recovery-items")).json()["items"] == []
        with SessionLocal() as session:
            entry = session.get(ArtifactLibraryEntry, entry_id)
            assert entry is not None and entry.state == "visible"
            assert entry.favorite and entry.display_name == "Garden layout"
            assert entry.deleted_at is None and entry.recovery_id is None
            assert session.scalar(select(func.count()).select_from(Artifact)) == 1
            assert session.scalar(select(func.count()).select_from(ArtifactLibraryEntry)) == 1
            assert session.scalar(select(func.count()).select_from(Job)) == 0
            membership = session.scalar(select(MediaCollectionMembership))
            assert (membership.collection_id, membership.entry_id, membership.note) == (
                collection_id,
                entry_id,
                "Keep original spacing",
            )
            assignment = session.scalar(select(MediaTagAssignment))
            assert (assignment.tag_id, assignment.entry_id) == (tag_id, entry_id)
            artifact = session.get(Artifact, artifact_id)
            assert artifact.metadata_json["uploaded"] is True
            assert app.state.services.artifacts.resolve(artifact).read_bytes() == CONTENT
    assert observed == [item["deletion_id"]]


async def test_import_cannot_restore_expired_membership_or_extend_its_deadline(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    (artifact_id, entry_id, _collection_id, _tag_id), item = await _deleted(app, client)
    deadline = datetime.fromisoformat(item["purge_after"]).astimezone(UTC)
    monkeypatch.setattr("local_lm.api.utcnow", lambda: deadline + timedelta(seconds=1))
    with SessionLocal() as session:
        before_metadata = dict(session.get(Artifact, artifact_id).metadata_json)
        before_operations = session.scalar(select(func.count()).select_from(RecoveryOperation))
    imported = await client.post(
        "/api/artifacts?kind=image",
        files={"file": ("Expired garden.png", CONTENT, "image/png")},
    )
    assert imported.status_code == 409, imported.text
    assert imported.json()["code"] == "recovery-window-expired"
    with SessionLocal() as session:
        entry = session.get(ArtifactLibraryEntry, entry_id)
        assert entry.state == "trashed" and entry.recovery_id == item["deletion_id"]
        recovery = session.get(RecoveryItem, item["deletion_id"])
        assert recovery.state == "recoverable"
        assert recovery.purge_after.replace(tzinfo=UTC) == deadline
        artifact = session.get(Artifact, artifact_id)
        assert artifact.metadata_json == before_metadata
        assert app.state.services.artifacts.resolve(artifact).read_bytes() == CONTENT
        assert (
            session.scalar(select(func.count()).select_from(RecoveryOperation)) == before_operations
        )


async def test_import_rolls_back_membership_restore_and_its_receipt_on_failure(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import media_reimport

    (artifact_id, _entry_id, _collection_id, _tag_id), _item = await _deleted(app, client)

    def rows():
        with SessionLocal() as session:
            return {
                model.__tablename__: session.execute(select(model.__table__)).all()
                for model in (
                    Artifact,
                    ArtifactLibraryEntry,
                    RecoveryItem,
                    RecoveryOperation,
                    RecoveryPreviewRecord,
                    MediaCollectionMembership,
                    MediaTagAssignment,
                    Job,
                )
            }

    original = media_reimport.restore_media

    def fail_after_restore(*args, **kwargs):
        original(*args, **kwargs)
        raise RecoveryPreviewConflict("recovery-impact-stale")

    before = rows()
    monkeypatch.setattr(media_reimport, "restore_media", fail_after_restore)
    response = await client.post(
        "/api/artifacts?kind=image",
        files={"file": ("Recoverable garden.png", CONTENT, "image/png")},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "recovery-impact-stale"
    assert rows() == before
    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert app.state.services.artifacts.resolve(artifact).read_bytes() == CONTENT
