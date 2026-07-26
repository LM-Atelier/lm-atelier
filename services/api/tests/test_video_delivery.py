from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx2 import AsyncClient

from local_lm.artifacts import _MAX_VIDEO_POSTER_BYTES, ArtifactStore


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


async def test_video_proxy_remains_file_backed_until_its_owner_discards_it(
    settings,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    store = ArtifactStore(settings)
    source = settings.data_dir / "source-video.mkv"
    source.write_bytes(b"source")
    proxy_content = b"file-backed-browser-proxy"

    class FakeProcess:
        returncode = 0

        async def wait(self) -> int:
            return self.returncode

    async def create_process(*arguments, **_kwargs):  # type: ignore[no-untyped-def]
        Path(arguments[-1]).write_bytes(proxy_content)
        return FakeProcess()

    monkeypatch.setattr("local_lm.artifacts.shutil.which", lambda _name: "ffmpeg")
    monkeypatch.setattr("local_lm.artifacts.asyncio.create_subprocess_exec", create_process)
    monkeypatch.setattr(store, "resolve", lambda _artifact: source)
    artifact = SimpleNamespace(
        media_type="video/x-matroska",
        original_name="source.mkv",
    )

    staged = await store.browser_video_proxy(artifact)  # type: ignore[arg-type]

    assert staged is not None
    assert staged.path.parent == store.root
    assert staged.path.read_bytes() == proxy_content
    assert staged.original_name == "source.mkv.proxy.mp4"
    staged.discard()
    assert not staged.path.exists()


async def test_video_poster_rejects_oversized_ffmpeg_output(
    settings,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    store = ArtifactStore(settings)
    source = settings.data_dir / "source-video.mp4"
    source.write_bytes(b"source")

    class OversizedStdout:
        returned = False

        async def read(self, _size: int) -> bytes:
            if self.returned:
                return b""
            self.returned = True
            return b"x" * (_MAX_VIDEO_POSTER_BYTES + 1)

    class FakeProcess:
        returncode: int | None = None
        stdout = OversizedStdout()
        killed = False

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    process = FakeProcess()

    async def create_process(*_arguments, **_kwargs):  # type: ignore[no-untyped-def]
        return process

    monkeypatch.setattr("local_lm.artifacts.shutil.which", lambda _name: "ffmpeg")
    monkeypatch.setattr("local_lm.artifacts.asyncio.create_subprocess_exec", create_process)
    monkeypatch.setattr(store, "resolve", lambda _artifact: source)
    artifact = SimpleNamespace(media_type="video/mp4")

    assert await store.video_poster(artifact) is None  # type: ignore[arg-type]
    assert process.killed is True


async def test_cancelling_video_poster_reaps_ffmpeg(
    settings,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    store = ArtifactStore(settings)
    source = settings.data_dir / "cancelled-source-video.mp4"
    source.write_bytes(b"source")
    started = asyncio.Event()

    class BlockingStdout:
        async def read(self, _size: int) -> bytes:
            started.set()
            await asyncio.Event().wait()
            return b""

    class FakeProcess:
        returncode: int | None = None
        stdout = BlockingStdout()
        killed = False

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    process = FakeProcess()

    async def create_process(*_arguments, **_kwargs):  # type: ignore[no-untyped-def]
        return process

    monkeypatch.setattr("local_lm.artifacts.shutil.which", lambda _name: "ffmpeg")
    monkeypatch.setattr("local_lm.artifacts.asyncio.create_subprocess_exec", create_process)
    monkeypatch.setattr(store, "resolve", lambda _artifact: source)
    artifact = SimpleNamespace(media_type="video/mp4")
    task = asyncio.create_task(store.video_poster(artifact))  # type: ignore[arg-type]
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed is True


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
    assert partial.headers["etag"] == f'"{artifact_id.removeprefix("sha256:")}"'
    assert partial.headers["content-security-policy"] == "sandbox; default-src 'none'"
    assert partial.headers["x-content-type-options"] == "nosniff"

    suffix = await client.get(
        f"/api/artifacts/{artifact_id}/content", headers={"Range": "bytes=-3"}
    )
    assert suffix.status_code == 206
    assert suffix.content == b"789"
    assert suffix.headers["content-range"] == "bytes 7-9/10"

    invalid = await client.get(
        f"/api/artifacts/{artifact_id}/content", headers={"Range": "bytes=50-60"}
    )
    assert invalid.status_code == 416
    assert invalid.headers["content-range"] == "bytes */10"

    complete = await client.get(f"/api/artifacts/{artifact_id}/content")
    assert complete.status_code == 200
    assert complete.content == content
    assert complete.headers["content-length"] == "10"
