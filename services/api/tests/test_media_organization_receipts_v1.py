"""Saved batch corruption cannot change media or disclose stored result fields."""

from __future__ import annotations

from typing import cast

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from test_media_organization_batches import _album, _apply, _impact, _media

from local_lm import db
from local_lm.config import Settings
from local_lm.media_organization_receipts_v1 import (
    organization_input_digest,
    organization_result_digest,
)
from local_lm.models import ArtifactLibraryEntry, MediaOrganizationImpact


async def _favorite_preview(client: AsyncClient, settings: Settings) -> dict[str, object]:
    media = _media(settings)
    response = await client.post(
        "/api/media-organization/impacts",
        json={
            "action": "set-favorite",
            "favorite": True,
            "entries": [item.selection() for item in media],
        },
    )
    assert response.status_code == 201
    return cast(dict[str, object], response.json())


@pytest.mark.parametrize(
    "damage",
    [
        "changed-request",
        "changed-proof",
        "changed-fingerprint",
        "changed-preview",
        "wrong-input-digest",
        "extra-preview-field",
        "wrong-preview-count",
        "wrong-preview-size",
        "wrong-preview-action",
        "wrong-preview-expiry",
        "naive-preview-expiry",
    ],
)
async def test_a_damaged_saved_preview_refuses_without_changing_media(
    client: AsyncClient,
    settings: Settings,
    damage: str,
) -> None:
    preview = await _favorite_preview(client, settings)
    with db.SessionLocal() as session:
        impact = session.get(MediaOrganizationImpact, preview["id"])
        assert impact is not None
        if damage == "changed-request":
            impact.request_json = {**impact.request_json, "favorite": False}
        elif damage == "changed-proof":
            impact.recovery_json = {"unexpected": {"value": "neutral stored marker"}}
        elif damage == "changed-fingerprint":
            impact.fingerprint = "0" * 64
        elif damage == "wrong-input-digest":
            impact.input_sha256 = "0" * 64
        else:
            mutations: dict[str, dict[str, object]] = {
                "changed-preview": {"changed_count": 0},
                "extra-preview-field": {"unexpected": "neutral stored marker"},
                "wrong-preview-count": {"selected_count": True},
                "wrong-preview-size": {"size_bytes": 999},
                "wrong-preview-action": {"action": "add-to-album"},
                "wrong-preview-expiry": {"expires_at": "invalid"},
                "naive-preview-expiry": {"expires_at": "2026-10-03T12:00:00"},
            }
            impact.preview_json = {**impact.preview_json, **mutations[damage]}
            if damage != "changed-preview":
                impact.input_sha256 = organization_input_digest(impact)
        session.commit()
    refused = await client.post(
        f"/api/media-organization/impacts/{preview['id']}/apply",
        json={"operation_key": "damaged-preview"},
    )
    assert refused.status_code == 409
    assert refused.json() == {
        "detail": "The selected media or organization changed. Review the selection again.",
        "code": "media-organization-stale",
    }
    with db.SessionLocal() as session:
        assert session.scalars(select(ArtifactLibraryEntry.favorite)).all() == [False, False]
        stored = session.get(MediaOrganizationImpact, preview["id"])
        assert stored is not None and stored.operation_key is None
        assert stored.response_json is None and stored.response_sha256 is None


@pytest.mark.parametrize(
    "damage",
    [
        "changed-result",
        "extra-result-field",
        "wrong-id",
        "wrong-action",
        "wrong-count",
        "boolean-count",
        "wrong-changed-count",
        "wrong-target-version",
        "unexpected-deletion-ids",
        "missing-result",
        "changed-request",
        "changed-operation-key",
    ],
)
async def test_a_damaged_committed_result_refuses_without_replaying_writes(
    client: AsyncClient,
    settings: Settings,
    damage: str,
) -> None:
    preview = await _favorite_preview(client, settings)
    committed = await _apply(client, preview, "kept-result")
    with db.SessionLocal() as session:
        before = session.execute(
            select(
                ArtifactLibraryEntry.id, ArtifactLibraryEntry.version, ArtifactLibraryEntry.favorite
            )
        ).all()
        impact = session.get(MediaOrganizationImpact, preview["id"])
        assert impact is not None and impact.response_json is not None
        if damage == "changed-request":
            impact.request_json = {"invalid": "neutral stored marker"}
        elif damage == "changed-operation-key":
            impact.operation_key = "altered-result"
        elif damage == "missing-result":
            impact.response_json = None
        else:
            mutations: dict[str, dict[str, object]] = {
                "changed-result": {"changed_count": 0},
                "extra-result-field": {"unexpected": "neutral stored marker"},
                "wrong-id": {"id": "orgimp_" + "f" * 32},
                "wrong-action": {"action": "trash"},
                "wrong-count": {"selected_count": 3},
                "boolean-count": {"selected_count": True},
                "wrong-changed-count": {"changed_count": 1},
                "wrong-target-version": {"target_version": 10},
                "unexpected-deletion-ids": {"deletion_ids": ["recover_" + "a" * 32]},
            }
            impact.response_json = {**impact.response_json, **mutations[damage]}
            if damage != "changed-result":
                impact.response_sha256 = organization_result_digest(impact)
        session.commit()
    refused = await client.post(
        f"/api/media-organization/impacts/{preview['id']}/apply",
        json={
            "operation_key": "altered-result"
            if damage == "changed-operation-key"
            else "kept-result"
        },
    )
    assert refused.status_code == 409
    assert "neutral stored marker" not in refused.text
    with db.SessionLocal() as session:
        assert (
            session.execute(
                select(
                    ArtifactLibraryEntry.id,
                    ArtifactLibraryEntry.version,
                    ArtifactLibraryEntry.favorite,
                )
            ).all()
            == before
        )
    assert committed["selected_count"] == 2


async def test_a_replay_cannot_claim_another_album_version_even_with_a_valid_result_digest(
    client: AsyncClient,
    settings: Settings,
) -> None:
    media = _media(settings)
    album = await _album(client)
    preview = await _impact(client, "add-to-album", album, media)
    applied = await _apply(client, preview, "album-version-result")
    assert applied["target_version"] == 3
    with db.SessionLocal() as session:
        impact = session.get(MediaOrganizationImpact, preview["id"])
        assert impact is not None and impact.response_json is not None
        impact.response_json = {**impact.response_json, "target_version": 9}
        impact.response_sha256 = organization_result_digest(impact)
        session.commit()
    refused = await client.post(
        f"/api/media-organization/impacts/{preview['id']}/apply",
        json={"operation_key": "album-version-result"},
    )
    assert refused.status_code == 409


@pytest.mark.parametrize("damage", ["duplicate", "missing", "invalid"])
async def test_a_trash_replay_requires_each_exact_recovery_identity_once(
    client: AsyncClient,
    settings: Settings,
    damage: str,
) -> None:
    media = _media(settings)
    response = await client.post(
        "/api/media-organization/impacts",
        json={"action": "trash", "entries": [item.selection() for item in media]},
    )
    assert response.status_code == 201
    preview = response.json()
    await _apply(client, preview, "trash-result-identities")
    with db.SessionLocal() as session:
        impact = session.get(MediaOrganizationImpact, preview["id"])
        assert impact is not None and impact.response_json is not None
        ids = impact.response_json["deletion_ids"]
        assert isinstance(ids, list)
        replacements = {
            "duplicate": [ids[0], ids[0]],
            "missing": [ids[0]],
            "invalid": [ids[0], "neutral stored marker"],
        }
        impact.response_json = {**impact.response_json, "deletion_ids": replacements[damage]}
        impact.response_sha256 = organization_result_digest(impact)
        session.commit()
    refused = await client.post(
        f"/api/media-organization/impacts/{preview['id']}/apply",
        json={"operation_key": "trash-result-identities"},
    )
    assert refused.status_code == 409
    assert "neutral stored marker" not in refused.text
    with db.SessionLocal() as session:
        assert session.scalars(select(ArtifactLibraryEntry.state)).all() == ["trashed", "trashed"]
