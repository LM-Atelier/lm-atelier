"""Opening the studio on a picture the person brought in themselves.

`kind` records where an artifact came from, not what it holds. A generated
picture is `image`; one a person uploaded is `input`. The studio asked for
`kind == image`, which asked "did we make this" and refused every uploaded
picture - most of the reason to open the studio at all.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from httpx2 import AsyncClient

pytestmark = pytest.mark.asyncio

# A one-pixel PNG, enough to be ingested as real image bytes.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
    "00000049454e44ae426082"
)


async def _upload(
    client: AsyncClient, name: str, media_type: str, payload: bytes
) -> dict[str, Any]:
    response = await client.post(
        "/api/artifacts",
        files={"file": (name, io.BytesIO(payload), media_type)},
    )
    assert response.status_code == 201, response.text
    uploaded: dict[str, Any] = response.json()
    return uploaded


async def test_the_studio_opens_on_an_uploaded_image(client: AsyncClient) -> None:
    uploaded = await _upload(client, "photo.png", "image/png", _PNG)

    # Stored as `input` because a person brought it, not because of what it is.
    assert uploaded["kind"] == "input"

    opened = await client.post("/api/studio/sessions", json={"source_artifact_id": uploaded["id"]})

    assert opened.status_code == 200, opened.text
    assert opened.json()["id"]


async def test_reopening_an_uploaded_image_resumes_the_same_session(
    client: AsyncClient,
) -> None:
    uploaded = await _upload(client, "photo.png", "image/png", _PNG)
    first = await client.post("/api/studio/sessions", json={"source_artifact_id": uploaded["id"]})
    second = await client.post("/api/studio/sessions", json={"source_artifact_id": uploaded["id"]})

    assert first.json()["id"] == second.json()["id"]


async def test_something_uploaded_that_is_not_an_image_is_still_refused(
    client: AsyncClient,
) -> None:
    """Accepting `input` must not become accepting anything a person uploads."""
    uploaded = await _upload(client, "notes.txt", "text/plain", b"not a picture")

    refused = await client.post("/api/studio/sessions", json={"source_artifact_id": uploaded["id"]})

    assert refused.status_code == 422
    assert refused.json()["code"] == "studio-image-only"
