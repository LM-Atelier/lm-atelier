"""Human review through the API, backed by retained artifact bytes."""

from __future__ import annotations

import io
import os
from contextlib import suppress

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select

from local_lm import api as api_module
from local_lm import artifacts as artifacts_module
from local_lm.db import SessionLocal
from local_lm.models import Artifact, ReferenceAsset, ReferenceAssetReviewEvent


def png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 110, 130)).save(buffer, format="PNG")
    return buffer.getvalue()


async def attached(client: AsyncClient, name: str, image: bytes) -> tuple[str, str]:
    subject = await client.post("/api/references", json={"name": name, "kind": "person"})
    assert subject.status_code == 201, subject.text
    subject_id = subject.json()["id"]

    upload = await client.post(
        "/api/artifacts", files={"file": (f"{name}.png", image, "image/png")}
    )
    assert upload.status_code in (200, 201), upload.text
    asset = await client.post(
        f"/api/references/{subject_id}/assets",
        json={"artifact_id": upload.json()["id"], "purpose": "identity"},
    )
    assert asset.status_code == 201, asset.text
    return subject_id, asset.json()["asset"]["id"]


async def test_review_promotes_and_records_exact_immutable_evidence(
    client: AsyncClient,
) -> None:
    image = png(1024, 768)
    subject_id, asset_id = await attached(client, "Reviewed", image)

    reviewed = await client.post(
        f"/api/references/{subject_id}/assets/{asset_id}/review",
        json={"outcome": "usable"},
    )

    assert reviewed.status_code == 200, reviewed.text
    body = reviewed.json()
    assert body["asset"]["validation_state"] == "usable"
    assert (body["width"], body["height"], body["review_version"]) == (1024, 768, 2)
    with SessionLocal() as session:
        asset = session.get(ReferenceAsset, asset_id)
        event = session.scalar(
            select(ReferenceAssetReviewEvent).where(
                ReferenceAssetReviewEvent.reference_asset_id == asset_id
            )
        )
        assert asset is not None
        assert event is not None
        artifact = session.get(Artifact, asset.artifact_id)
        assert artifact is not None
        assert event.artifact_id == artifact.id
        assert event.artifact_sha256 == artifact.sha256
        assert event.expected_version == 1
        assert event.result_version == 2
        assert event.decision == "usable"
        assert (event.width, event.height) == (1024, 768)


async def test_rejection_requires_a_reason_and_records_it(client: AsyncClient) -> None:
    subject_id, asset_id = await attached(client, "Rejected", png(800, 800))

    bare = await client.post(
        f"/api/references/{subject_id}/assets/{asset_id}/review",
        json={"outcome": "rejected"},
    )
    assert bare.status_code == 422
    assert bare.json()["code"] == "reference-review-refused"

    explained = await client.post(
        f"/api/references/{subject_id}/assets/{asset_id}/review",
        json={"outcome": "rejected", "reasons": ["wrong person"]},
    )
    assert explained.status_code == 200, explained.text
    assert explained.json()["asset"]["validation_state"] == "rejected"
    with SessionLocal() as session:
        event = session.scalar(
            select(ReferenceAssetReviewEvent).where(
                ReferenceAssetReviewEvent.reference_asset_id == asset_id
            )
        )
        assert event is not None
        assert event.reasons_json == ["wrong person"]


async def test_unknown_review_outcome_is_refused(client: AsyncClient) -> None:
    subject_id, asset_id = await attached(client, "Unknown", png(800, 800))

    refused = await client.post(
        f"/api/references/{subject_id}/assets/{asset_id}/review",
        json={"outcome": "probably usable"},
    )

    assert refused.status_code == 422
    assert refused.json()["code"] == "reference-review-outcome-unsupported"


async def test_size_refusal_reports_measurement_without_settling(client: AsyncClient) -> None:
    subject_id, asset_id = await attached(client, "Tiny", png(64, 64))

    refused = await client.post(
        f"/api/references/{subject_id}/assets/{asset_id}/review",
        json={"outcome": "usable"},
    )

    assert refused.status_code == 422
    assert refused.json()["code"] == "reference-review-refused"
    assert "64x64" in refused.json()["detail"]
    with SessionLocal() as session:
        asset = session.get(ReferenceAsset, asset_id)
        assert asset is not None
        assert asset.validation_state == "unchecked"
        assert (
            session.scalar(
                select(ReferenceAssetReviewEvent).where(
                    ReferenceAssetReviewEvent.reference_asset_id == asset_id
                )
            )
            is None
        )


async def test_review_cannot_reach_another_subject_and_cannot_settle_twice(
    client: AsyncClient,
) -> None:
    owner_id, asset_id = await attached(client, "Owner", png(800, 800))
    other_id, _ = await attached(client, "Other", png(800, 800))

    crossed = await client.post(
        f"/api/references/{other_id}/assets/{asset_id}/review",
        json={"outcome": "usable"},
    )
    assert crossed.status_code == 404
    assert crossed.json()["code"] == "reference-asset-not-attached"

    first = await client.post(
        f"/api/references/{owner_id}/assets/{asset_id}/review",
        json={"outcome": "usable"},
    )
    assert first.status_code == 200, first.text
    second = await client.post(
        f"/api/references/{owner_id}/assets/{asset_id}/review",
        json={"outcome": "rejected", "reasons": ["changed mind"]},
    )
    assert second.status_code == 409
    assert second.json()["code"] == "reference-review-already-settled"


async def test_review_read_is_bounded_before_materializing_the_artifact(
    client: AsyncClient,
    monkeypatch,
) -> None:
    image = png(800, 800)
    subject_id, asset_id = await attached(client, "Bounded", image)
    monkeypatch.setattr(api_module, "_MAX_REFERENCE_REVIEW_BYTES", len(image) - 1)

    refused = await client.post(
        f"/api/references/{subject_id}/assets/{asset_id}/review",
        json={"outcome": "usable"},
    )

    assert refused.status_code == 422
    assert refused.json()["code"] == "reference-review-refused"
    with SessionLocal() as session:
        asset = session.get(ReferenceAsset, asset_id)
        assert asset is not None
        assert asset.validation_state == "unchecked"


async def test_changed_retained_bytes_cannot_support_a_review(
    client: AsyncClient,
    app: FastAPI,
) -> None:
    image = png(800, 800)
    subject_id, asset_id = await attached(client, "Changed", image)
    with SessionLocal() as session:
        asset = session.get(ReferenceAsset, asset_id)
        assert asset is not None
        artifact = session.get(Artifact, asset.artifact_id)
        assert artifact is not None
        path = app.state.services.artifacts.resolve(artifact)
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    refused = await client.post(
        f"/api/references/{subject_id}/assets/{asset_id}/review",
        json={"outcome": "usable"},
    )

    assert refused.status_code == 422
    assert refused.json()["code"] == "reference-review-refused"
    with SessionLocal() as session:
        asset = session.get(ReferenceAsset, asset_id)
        assert asset is not None
        assert asset.validation_state == "unchecked"
    path.write_bytes(original)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("id", "sha256:" + "0" * 64, "artifact identity is invalid"),
        ("relative_path", "not/the/canonical/path", "artifact path is not canonical"),
        ("size_bytes", -1, "artifact file size does not match its record"),
    ],
)
async def test_verified_bytes_refuses_untrusted_artifact_record_fields(
    client: AsyncClient,
    app: FastAPI,
    field: str,
    replacement: str | int,
    message: str,
) -> None:
    image = png(800, 800)
    _subject_id, asset_id = await attached(client, f"Record-{field}", image)
    with SessionLocal() as session:
        asset = session.get(ReferenceAsset, asset_id)
        assert asset is not None
        artifact = session.get(Artifact, asset.artifact_id)
        assert artifact is not None
        setattr(artifact, field, replacement)

        with pytest.raises(ValueError, match=message):
            app.state.services.artifacts.verified_bytes(artifact, maximum_bytes=len(image))


async def test_verified_bytes_refuses_a_recorded_size_mismatch_before_reading(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = png(800, 800)
    _subject_id, asset_id = await attached(client, "Recorded-size", image)
    with SessionLocal() as session:
        asset = session.get(ReferenceAsset, asset_id)
        assert asset is not None
        artifact = session.get(Artifact, asset.artifact_id)
        assert artifact is not None
        path = app.state.services.artifacts.resolve(artifact)
        source_descriptor = os.open(path, os.O_RDONLY)
        monkeypatch.setattr(
            artifacts_module,
            "open_entry",
            lambda _anchor, _name: os.dup(source_descriptor),
        )
        artifact.size_bytes += 1
        try:
            with pytest.raises(ValueError, match="file size does not match its record"):
                app.state.services.artifacts.verified_bytes(artifact, maximum_bytes=len(image) + 1)
            assert os.lseek(source_descriptor, 0, os.SEEK_CUR) == 0
        finally:
            with suppress(OSError):
                os.close(source_descriptor)


async def test_verified_bytes_refuses_same_size_content_with_the_wrong_digest(
    client: AsyncClient,
    app: FastAPI,
) -> None:
    image = png(800, 800)
    _subject_id, asset_id = await attached(client, "Digest", image)
    with SessionLocal() as session:
        asset = session.get(ReferenceAsset, asset_id)
        assert asset is not None
        artifact = session.get(Artifact, asset.artifact_id)
        assert artifact is not None
        path = app.state.services.artifacts.resolve(artifact)
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        try:
            with pytest.raises(ValueError, match="checksum does not match"):
                app.state.services.artifacts.verified_bytes(artifact, maximum_bytes=len(image))
        finally:
            path.write_bytes(original)


async def test_verified_bytes_refuses_a_non_regular_opened_descriptor(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = png(800, 800)
    _subject_id, asset_id = await attached(client, "Non-regular", image)
    with SessionLocal() as session:
        asset = session.get(ReferenceAsset, asset_id)
        assert asset is not None
        artifact = session.get(Artifact, asset.artifact_id)
        assert artifact is not None
        read_descriptor, write_descriptor = os.pipe()
        monkeypatch.setattr(artifacts_module, "open_entry", lambda _anchor, _name: read_descriptor)
        try:
            with pytest.raises(ValueError, match="not a regular file"):
                app.state.services.artifacts.verified_bytes(artifact, maximum_bytes=len(image))
        finally:
            for descriptor in (read_descriptor, write_descriptor):
                with suppress(OSError):
                    os.close(descriptor)


async def test_review_materializes_the_descriptor_opened_through_the_held_shard(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject_id, asset_id = await attached(client, "Held", png(800, 800))
    opened: list[int] = []
    materialized: list[int] = []
    real_open_entry = artifacts_module.open_entry
    real_fdopen = artifacts_module.os.fdopen

    def tracked_open_entry(anchor, name: str) -> int | None:
        descriptor = real_open_entry(anchor, name)
        if descriptor is not None:
            opened.append(descriptor)
        return descriptor

    def tracked_fdopen(descriptor: int, *args, **kwargs):
        materialized.append(descriptor)
        return real_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(artifacts_module, "open_entry", tracked_open_entry)
    monkeypatch.setattr(artifacts_module.os, "fdopen", tracked_fdopen)

    reviewed = await client.post(
        f"/api/references/{subject_id}/assets/{asset_id}/review",
        json={"outcome": "usable"},
    )

    assert reviewed.status_code == 200, reviewed.text
    assert len(opened) == 1
    assert materialized == opened
