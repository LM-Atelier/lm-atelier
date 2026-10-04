"""Organization refusals identify recoverable causes before any partial write."""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from test_media_organization_batches import _apply, _impact, _media, _tag

from local_lm import db
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.models import Artifact, MediaOrganizationImpact, MediaTag, MediaTagAssignment


async def _destination(client: AsyncClient) -> dict[str, object]:
    response = await client.post("/api/media-tags", json={"label": "Landscape"})
    assert response.status_code == 201
    return response.json()


@pytest.mark.parametrize("label", ["landscape", "LANDSCAPE"])
async def test_a_taken_normalized_tag_name_is_refused_before_preview(
    client: AsyncClient, label: str
) -> None:
    source = await _tag(client)
    destination = await _destination(client)
    for _ in range(2):
        response = await client.post(
            "/api/media-organization/impacts",
            json={
                "action": "rename-tag",
                "target": {"id": source["id"], "version": source["version"]},
                "changes": {"label": label, "color": None},
            },
        )
        assert response.status_code == 409
        assert response.json()["code"] == "media-tag-conflict"
    with db.SessionLocal() as session:
        assert session.scalars(select(MediaOrganizationImpact)).all() == []
        assert session.get(MediaTag, source["id"]).label == "Study"
        assert session.get(MediaTag, destination["id"]).label == "Landscape"


async def test_a_name_taken_after_preview_has_the_same_specific_refusal(
    client: AsyncClient,
) -> None:
    source = await _tag(client)
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": "rename-tag",
            "target": {"id": source["id"], "version": source["version"]},
            "changes": {"label": "Landscape", "color": None},
        },
    )
    assert response.status_code == 201
    impact = response.json()
    await _destination(client)
    response = await client.post(
        f"/api/media-organization/impacts/{impact['id']}/apply",
        json={"operation_key": "name-taken-after-review"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "media-tag-conflict"
    with db.SessionLocal() as session:
        source_tag = session.get(MediaTag, source["id"])
        assert source_tag is not None and source_tag.label == "Study" and source_tag.version == 1
        saved = session.get(MediaOrganizationImpact, impact["id"])
        assert saved is not None and saved.operation_key is None and saved.response_json is None


async def test_renaming_a_tag_with_its_own_normalized_name_remains_allowed(
    client: AsyncClient,
) -> None:
    source = await _tag(client)
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": "rename-tag",
            "target": {"id": source["id"], "version": source["version"]},
            "changes": {"label": "STUDY", "color": None},
        },
    )
    assert response.status_code == 201
    assert (await _apply(client, response.json(), "same-normalized-name"))["target_version"] == 2


@pytest.mark.parametrize("overlap", [False, True])
async def test_a_tag_merge_explains_when_a_trashed_source_needs_restoring(
    client: AsyncClient, settings: Settings, overlap: bool
) -> None:
    media = _media(settings)
    source = await _tag(client)
    destination = await _destination(client)
    source["version"] = (
        await _apply(client, await _impact(client, "add-tag", source, media), "source-tags")
    )["target_version"]
    if overlap:
        destination["version"] = (
            await _apply(
                client,
                await _impact(client, "add-tag", destination, media[1:]),
                "destination-tags",
            )
        )["target_version"]
    response = await client.post(
        "/api/media-organization/impacts",
        json={"action": "trash", "entries": [media[1].selection()]},
    )
    assert response.status_code == 201
    await _apply(client, response.json(), "trash-source-carrier")
    command = {
        "action": "merge-tags",
        "target": {"id": source["id"], "version": source["version"]},
        "destination": {"id": destination["id"], "version": destination["version"]},
    }
    with db.SessionLocal() as session:
        before = len(session.scalars(select(MediaOrganizationImpact)).all())
        assignments_before = set(
            session.execute(select(MediaTagAssignment.tag_id, MediaTagAssignment.entry_id)).all()
        )
    for _ in range(2):
        response = await client.post("/api/media-organization/impacts", json=command)
        assert response.status_code == 409
        assert response.json()["code"] == "media-tag-merge-trash"
    with db.SessionLocal() as session:
        assert len(session.scalars(select(MediaOrganizationImpact)).all()) == before
        assert session.get(MediaTag, source["id"]) is not None
        assert (
            set(
                session.execute(
                    select(MediaTagAssignment.tag_id, MediaTagAssignment.entry_id)
                ).all()
            )
            == assignments_before
        )
    store = ArtifactStore(settings)
    with db.SessionLocal() as session:
        for item in media:
            artifact = session.get(Artifact, item.artifact_id)
            assert artifact is not None
            assert store.resolve(artifact).read_bytes() == item.content
