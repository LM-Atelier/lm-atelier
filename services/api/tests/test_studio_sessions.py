"""Studio sessions: one durable hidden chat per source image."""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient

pytestmark = pytest.mark.asyncio


def _seed_image(suffix: str = "a") -> str:
    from local_lm.db import SessionLocal
    from local_lm.domain import ArtifactKind
    from local_lm.models import Artifact

    digest = suffix * 64
    with SessionLocal() as session:
        session.add(
            Artifact(
                id=f"sha256:{digest}",
                sha256=digest,
                kind=ArtifactKind.IMAGE.value,
                media_type="image/png",
                size_bytes=8,
                relative_path=f"{digest[:2]}/{digest[2:4]}/{digest}",
                original_name="harbor.png",
            )
        )
        session.commit()
    return f"sha256:{digest}"


async def test_opening_twice_resumes_the_same_session(client: AsyncClient) -> None:
    artifact_id = _seed_image()

    first = await client.post("/api/studio/sessions", json={"source_artifact_id": artifact_id})
    assert first.status_code == 200
    body = first.json()
    assert body["title"] == "Studio - harbor.png"
    assert body["routing_mode"] == "image"

    second = await client.post("/api/studio/sessions", json={"source_artifact_id": artifact_id})
    assert second.json()["id"] == body["id"]

    fetched = await client.get(f"/api/studio/sessions/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["messages"] == []


async def test_sessions_stay_out_of_the_chat_sidebar_and_chat_api(client: AsyncClient) -> None:
    artifact_id = _seed_image("b")
    opened = await client.post("/api/studio/sessions", json={"source_artifact_id": artifact_id})
    session_id = opened.json()["id"]

    chats = await client.get("/api/chats")
    assert all(chat["id"] != session_id for chat in chats.json())
    # The standard chat endpoint refuses the studio scope; the studio's own
    # endpoint is the only reader.
    assert (await client.get(f"/api/chats/{session_id}")).status_code == 404


async def test_a_chat_entry_carries_its_settings_snapshot(client: AsyncClient) -> None:
    artifact_id = _seed_image("c")
    chat = (await client.post("/api/chats", json={"title": "Source", "project_id": None})).json()
    await client.patch(
        f"/api/chats/{chat['id']}",
        json={"generation_settings_json": {"image": {"steps": 12}}},
    )

    opened = await client.post(
        "/api/studio/sessions",
        json={"source_artifact_id": artifact_id, "source_chat_id": chat["id"]},
    )
    assert opened.status_code == 200
    assert opened.json()["generation_settings_json"]["image"] == {"steps": 12}


async def test_refusals_are_typed(client: AsyncClient) -> None:
    absent = await client.post("/api/studio/sessions", json={"source_artifact_id": "sha256:absent"})
    assert absent.status_code == 404
    assert absent.json()["code"] == "artifact-not-found"

    from local_lm.db import SessionLocal
    from local_lm.domain import ArtifactKind
    from local_lm.models import Artifact

    with SessionLocal() as session:
        session.add(
            Artifact(
                id=f"sha256:{'d' * 64}",
                sha256="d" * 64,
                kind=ArtifactKind.VIDEO.value,
                media_type="video/mp4",
                size_bytes=8,
                relative_path=f"dd/dd/{'d' * 64}",
            )
        )
        session.commit()
    video = await client.post(
        "/api/studio/sessions", json={"source_artifact_id": f"sha256:{'d' * 64}"}
    )
    assert video.status_code == 422
    assert video.json()["code"] == "studio-image-only"

    missing = await client.get("/api/studio/sessions/absent")
    assert missing.status_code == 404
    assert missing.json()["code"] == "studio-session-not-found"
