from __future__ import annotations

import os

from httpx2 import AsyncClient

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import Artifact


async def test_spoofed_inline_media_is_delivered_as_a_download(
    client: AsyncClient,
) -> None:
    content = b"<html><script>alert('not an image')</script></html>"
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": (r"C:\private\spoofed.png", content, "image/png")},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["original_name"] == "spoofed.png"

    response = await client.get(f"/api/artifacts/{uploaded.json()['id']}/content")

    assert response.status_code == 200
    assert response.content == content
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert response.headers["content-security-policy"] == "sandbox; default-src 'none'"


async def test_corrupt_cas_content_returns_gone_instead_of_wrong_immutable_bytes(
    client: AsyncClient,
    settings: Settings,
) -> None:
    content = b"\x89PNG\r\n\x1a\noriginal"
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("image.png", content, "image/png")},
    )
    artifact_id = uploaded.json()["id"]
    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact
        relative_path = artifact.relative_path

    artifact_path = settings.artifact_dir / relative_path
    artifact_path.write_bytes(b"\x89PNG\r\n\x1a\nmodified")
    future = artifact_path.stat().st_mtime + 2
    os.utime(artifact_path, (future, future))

    response = await client.get(f"/api/artifacts/{artifact_id}/content")

    assert response.status_code == 410
    assert response.json()["detail"] == "artifact file is missing or corrupt"
