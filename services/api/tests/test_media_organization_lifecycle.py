"""Organization previews expire without losing committed replay or large selections."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import timedelta
from typing import cast

from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_media_organization_batches import _apply, _media, _tag

from local_lm import db
from local_lm.config import Settings
from local_lm.domain import utcnow
from local_lm.media_organization_batches import preview_organization
from local_lm.models import (
    ArtifactLibraryEntry,
    MediaOrganizationImpact,
    MediaTag,
    MediaTagAssignment,
)


async def test_an_expired_preview_refuses_before_changing_any_selected_media(
    client: AsyncClient, settings: Settings
) -> None:
    media = _media(settings)
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": "set-favorite",
            "favorite": True,
            "entries": [m.selection() for m in media],
        },
    )
    assert response.status_code == 201
    impact_id = response.json()["id"]
    with db.SessionLocal() as session:
        impact = session.get(MediaOrganizationImpact, impact_id)
        assert impact is not None
        impact.expires_at = utcnow() - timedelta(seconds=1)
        session.commit()
    refused = await client.post(
        f"/api/media-organization/impacts/{impact_id}/apply",
        json={"operation_key": "expired-selection"},
    )
    assert refused.status_code == 409
    with db.SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(ArtifactLibraryEntry)) == 2
        assert session.scalars(select(ArtifactLibraryEntry.favorite)).all() == [False, False]
        stored = session.get(MediaOrganizationImpact, impact_id)
        assert stored is not None and stored.operation_key is None


async def test_preview_retirement_is_bounded_and_keeps_committed_replays(
    client: AsyncClient, settings: Settings
) -> None:
    media = _media(settings, 1)
    request = {"action": "set-favorite", "favorite": True, "entries": [media[0].selection()]}
    created = await client.post("/api/media-organization/impacts", json=request)
    assert created.status_code == 201
    committed = await _apply(client, created.json(), "kept-committed-result")
    now = utcnow()
    with db.SessionLocal() as session:
        kept = session.get(MediaOrganizationImpact, created.json()["id"])
        assert kept is not None
        kept.expires_at = now - timedelta(hours=1)
        entry = session.get(ArtifactLibraryEntry, media[0].entry_id)
        assert entry is not None
        selected = replace(media[0], version=entry.version)
        for index in range(1002):
            session.add(
                MediaOrganizationImpact(
                    id=f"orgimp_{index:032x}",
                    request_json=request,
                    preview_json={},
                    fingerprint="a" * 64,
                    input_sha256="a" * 64,
                    expires_at=now - timedelta(minutes=1),
                )
            )
        future_id = "orgimp_" + "f" * 32
        session.add(
            MediaOrganizationImpact(
                id=future_id,
                request_json=request,
                preview_json={},
                fingerprint="b" * 64,
                input_sha256="b" * 64,
                expires_at=now + timedelta(hours=1),
            )
        )
        session.commit()
    next_preview = await client.post(
        "/api/media-organization/impacts",
        json={**request, "entries": [selected.selection()]},
    )
    assert next_preview.status_code == 201
    with db.SessionLocal() as session:
        # Two older unused previews, one future preview, one committed receipt,
        # and the newly requested preview remain after one bounded retirement.
        assert session.scalar(select(func.count()).select_from(MediaOrganizationImpact)) == 5
        assert session.get(MediaOrganizationImpact, future_id) is not None
        assert session.get(MediaOrganizationImpact, created.json()["id"]) is not None
    assert await _apply(client, created.json(), "kept-committed-result") == committed


async def test_a_large_tag_preview_respects_the_database_bind_limit(
    client: AsyncClient, settings: Settings
) -> None:
    media = _media(settings, 1001)
    tag = await _tag(client)
    with db.SessionLocal() as session:
        session.add_all(
            [
                MediaTagAssignment(tag_id=tag["id"], entry_id=item.entry_id, added_at=utcnow())
                for item in media
            ]
        )
        session.commit()
        stored = session.get(MediaTag, tag["id"])
        assert stored is not None
        version = stored.version
    with db.SessionLocal() as session:
        connection = cast(sqlite3.Connection, session.connection().connection.driver_connection)
        prior_limit = connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        try:
            response = preview_organization(
                session,
                {"action": "delete-tag", "target": {"id": tag["id"], "version": version}},
            )
            session.commit()
        finally:
            connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, prior_limit)
    assert response["selected_count"] == 1001
    assert response["size_bytes"] == sum(len(item.content) for item in media)
    with db.SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(MediaTagAssignment)) == 1001
