from __future__ import annotations

from types import SimpleNamespace

from httpx2 import AsyncClient

from local_lm.artifacts import ArtifactStore


async def test_browser_safe_video_does_not_need_a_proxy(settings) -> None:  # type: ignore[no-untyped-def]
    store = ArtifactStore(settings)
    artifact = SimpleNamespace(media_type="video/mp4")
    assert await store.browser_video_proxy(artifact) is None  # type: ignore[arg-type]


async def test_missing_ffmpeg_leaves_incompatible_original_available(
    settings,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    store = ArtifactStore(settings)
    monkeypatch.setattr("local_lm.artifacts.shutil.which", lambda _name: None)
    artifact = SimpleNamespace(media_type="image/gif")
    assert await store.browser_video_proxy(artifact) is None  # type: ignore[arg-type]


async def test_artifact_content_supports_browser_byte_ranges(client: AsyncClient) -> None:
    content = b"0123456789"
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("clip.mp4", content, "video/mp4")},
    )
    artifact_id = uploaded.json()["id"]
    partial = await client.get(
        f"/api/artifacts/{artifact_id}/content", headers={"Range": "bytes=2-5"}
    )
    assert partial.status_code == 206
    assert partial.content == b"2345"
    assert partial.headers["content-range"] == "bytes 2-5/10"
    assert partial.headers["accept-ranges"] == "bytes"
