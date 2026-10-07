"""Reading what a stored video is, within fixed bounds, before a video utility touches it.

The fixtures are tiny test patterns made by the real ffmpeg that hosted CI
installs, and the cases that probe a stored video run the real ffprobe found
beside it. The rest read hand-built answers or stop before ffprobe would run.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm import artifacts, media_process, media_tools, video_probe, video_utility_api
from local_lm.artifacts import ArtifactStore
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.filesystem_links import AnchoredDirectoryError
from local_lm.media_process import ToolResult
from local_lm.media_tools import MediaTool, MediaToolUnavailable, find_media_tool
from local_lm.models import Artifact
from local_lm.video_probe import VideoProbeRefused, describe

PATTERN = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10"]
TONE = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000"]
H264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "5"]


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    assert executable, "these cases need a real ffmpeg on PATH, as hosted CI provides"
    return executable


def _make(directory: Path, name: str, *arguments: str) -> bytes:
    target = directory / name
    subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *arguments, str(target)],
        check=True,
        timeout=60,
    )
    return target.read_bytes()


def _store(app: FastAPI, content: bytes, media_type: str = "video/mp4") -> str:
    kind = ArtifactKind.VIDEO if media_type.startswith("video/") else ArtifactKind.IMAGE
    with SessionLocal() as session:
        artifact = app.state.services.artifacts.ingest_bytes(
            session, content, kind=kind, media_type=media_type, original_name="clip"
        )
        session.commit()
        return str(artifact.id)


async def _probe(client: AsyncClient, artifact_id: str) -> Any:
    return await client.get(f"/api/artifacts/{artifact_id}/video-probe")


def _stored(app: FastAPI, artifact_id: str) -> tuple[Artifact, Path]:
    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact is not None
        return artifact, app.state.services.artifacts.resolve(artifact)


def _replace_keeping_size_and_time(path: Path, content: bytes) -> None:
    status = path.stat()
    path.write_bytes(content)
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (status.st_size, status.st_mtime_ns)


def _copies_left(app: FastAPI) -> list[Path]:
    return list(app.state.services.artifacts.root.glob("tool-input-*"))


def _link_directory(link: Path, target: Path) -> None:
    """Point a directory name at another directory: a junction on Windows, a symlink elsewhere."""

    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
        )
    else:
        os.symlink(target, link, target_is_directory=True)


async def test_a_stored_video_is_described_with_what_the_utilities_may_do(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(tmp_path, "clip.mp4", *PATTERN, *TONE, "-t", "1", *H264, "-c:a", "aac")
    artifact_id = _store(app, content)

    response = await _probe(client, artifact_id)

    assert response.status_code == 200, response.text
    probe = response.json()
    assert probe["version"] == 1
    assert probe["artifact_id"] == artifact_id
    assert probe["artifact_sha256"] == hashlib.sha256(content).hexdigest()
    assert probe["container"] == "mp4"
    assert probe["duration_seconds"] == pytest.approx(1.0, abs=0.1)
    assert probe["start_seconds"] == pytest.approx(0.0, abs=0.05)
    assert probe["video"] == {
        "index": 0,
        "codec": "h264",
        "width": 64,
        "height": 48,
        "display_width": 64,
        "display_height": 48,
        "sample_aspect": None,
        "rotation": 0,
        "frame_rate": "10/1",
        "frame_rate_form": "constant",
        "time_base": probe["video"]["time_base"],
    }
    assert re.fullmatch(r"1/\d+", probe["video"]["time_base"])
    assert probe["audio"] == [{"index": 1, "codec": "aac", "channels": 1, "sample_rate": 8000}]
    assert probe["omitted_streams"] == 0
    assert (probe["can_save_frame"], probe["can_trim"], probe["can_keep_audio"]) == (
        True,
        True,
        True,
    )
    assert probe["limits"] == []
    # The ffprobe found on the system path, by its version and file digest, never its path.
    tool = probe["tool"]
    assert set(tool) == {"name", "version", "sha256", "origin"}
    assert (tool["name"], tool["origin"]) == ("ffprobe", "system")
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None
    assert tool["sha256"] == hashlib.sha256(Path(ffprobe).read_bytes()).hexdigest()


async def test_a_turned_video_with_wide_pixels_is_shown_at_its_displayed_size(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    _make(tmp_path, "wide.mp4", *PATTERN, "-t", "1", "-vf", "setsar=2/1", *H264)
    wide = str(tmp_path / "wide.mp4")
    content = _make(tmp_path, "turned.mp4", "-display_rotation", "90", "-i", wide, "-c", "copy")

    probe = (await _probe(client, _store(app, content))).json()

    video = probe["video"]
    assert (video["width"], video["height"], video["sample_aspect"]) == (64, 48, "2:1")
    assert video["rotation"] in (90, 270)
    # Twice as wide per pixel, then turned a quarter: 128 wide becomes 128 tall.
    assert (video["display_width"], video["display_height"]) == (48, 128)
    assert probe["can_save_frame"] is True


async def test_a_matroska_video_with_audio_it_cannot_copy_can_still_be_trimmed_without_it(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(tmp_path, "clip.mkv", *PATTERN, *TONE, "-t", "1", *H264, "-c:a", "pcm_s16le")

    probe = (await _probe(client, _store(app, content, "video/x-matroska"))).json()

    assert probe["container"] == "matroska"
    assert probe["audio"][0]["codec"] == "pcm_s16le"
    assert (probe["can_save_frame"], probe["can_trim"], probe["can_keep_audio"]) == (
        True,
        True,
        False,
    )
    assert probe["limits"] == ["audio-codec-unsupported"]


async def test_a_video_codec_outside_the_list_is_described_but_not_offered(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(tmp_path, "clip.mkv", *PATTERN, "-t", "1", "-c:v", "mjpeg")

    probe = (await _probe(client, _store(app, content, "video/x-matroska"))).json()

    assert probe["video"]["codec"] == "mjpeg"
    assert (probe["can_save_frame"], probe["can_trim"]) == (False, False)
    assert probe["limits"] == ["video-codec-unsupported"]


async def test_a_list_that_names_other_files_is_refused(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    # A concatenation list stored as a video names a real video beside it. Only
    # the stored file itself is ever read as a video.
    _make(tmp_path, "elsewhere.mp4", *PATTERN, "-t", "1", *H264)
    listing = f"ffconcat version 1.0\nfile '{(tmp_path / 'elsewhere.mp4').as_posix()}'\n"

    response = await _probe(client, _store(app, listing.encode()))

    assert response.status_code == 422
    assert response.json()["code"] == "video-probe-unreadable"


async def test_a_container_outside_mp4_and_matroska_is_refused(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(tmp_path, "clip.avi", *PATTERN, "-t", "1", "-c:v", "mpeg4")

    response = await _probe(client, _store(app, content, "video/x-msvideo"))

    assert response.status_code == 422
    assert response.json()["code"] == "video-probe-unreadable"


async def test_a_file_with_no_video_stream_is_refused(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(tmp_path, "tone.mp4", *TONE, "-t", "1", "-c:a", "aac")

    response = await _probe(client, _store(app, content))

    assert response.status_code == 422
    assert response.json()["code"] == "video-probe-no-video-stream"


async def test_a_picture_is_refused_without_running_ffprobe(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _make(tmp_path, "still.png", *PATTERN, "-frames:v", "1")
    artifact_id = _store(app, content, "image/png")

    async def no_probe(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ffprobe ran for a picture")

    # Not even looked for: the answer is the same with no ffprobe installed.
    monkeypatch.setattr(video_probe, "run_tool", no_probe)
    monkeypatch.setattr(media_tools, "run_tool", no_probe)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    response = await _probe(client, artifact_id)

    assert response.status_code == 422
    assert response.json()["code"] == "video-probe-not-a-video"


async def test_a_missing_or_stored_video_answers_with_its_code(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unknown = await _probe(client, "sha256:" + "0" * 64)
    assert (unknown.status_code, unknown.json()["code"]) == (404, "artifact-not-found")

    artifact_id = _store(app, _make(tmp_path, "clip.mp4", *PATTERN, "-t", "1", *H264))
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    missing = await _probe(client, artifact_id)
    assert (missing.status_code, missing.json()["code"]) == (503, "media-tool-missing")


async def test_a_corrupted_stored_video_is_never_handed_to_ffprobe(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id = _store(app, _make(tmp_path, "clip.mp4", *PATTERN, "-t", "1", *H264))
    _artifact_row, path = _stored(app, artifact_id)
    original = path.read_bytes()
    # Same size, one bit different: only the digest can tell.
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    async def no_probe(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ffprobe ran on bytes that do not match the record")

    monkeypatch.setattr(video_probe, "run_tool", no_probe)
    response = await _probe(client, artifact_id)

    assert (response.status_code, response.json()["code"]) == (410, "artifact-file-unreadable")


async def test_a_video_replaced_after_an_earlier_read_is_refused_though_it_kept_size_and_time(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _make(tmp_path, "clip.mp4", *PATTERN, "-t", "1", *H264)
    artifact_id = _store(app, content)
    artifact, path = _stored(app, artifact_id)
    # Read and checked once, as anything serving the file would have done.
    app.state.services.artifacts.verified_path(artifact)
    assert (await _probe(client, artifact_id)).status_code == 200
    _replace_keeping_size_and_time(path, bytes([content[0] ^ 1]) + content[1:])

    async def no_probe(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ffprobe ran on bytes that do not match the record")

    monkeypatch.setattr(video_probe, "run_tool", no_probe)
    response = await _probe(client, artifact_id)

    assert (response.status_code, response.json()["code"]) == (410, "artifact-file-unreadable")
    assert _copies_left(app) == []


async def test_a_stored_video_reached_through_a_link_is_refused_before_ffprobe_runs(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _make(tmp_path, "clip.mp4", *PATTERN, "-t", "1", *H264)
    artifact_id = _store(app, content)
    _artifact_row, stored = _stored(app, artifact_id)
    # The very same bytes, moved out of the store and linked back in at their shard.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / stored.name).write_bytes(stored.read_bytes())
    stored.unlink()
    stored.parent.rmdir()
    _link_directory(stored.parent, elsewhere)
    assert stored.read_bytes() == content

    async def no_probe(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ffprobe ran on a file reached through a link")

    monkeypatch.setattr(video_probe, "run_tool", no_probe)
    response = await _probe(client, artifact_id)

    assert (response.status_code, response.json()["code"]) == (410, "artifact-file-unreadable")
    assert _copies_left(app) == []
    assert (elsewhere / stored.name).read_bytes() == content


async def test_ffprobe_reads_a_private_copy_of_exactly_the_verified_bytes(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _make(tmp_path, "clip.mp4", *PATTERN, "-t", "1", *H264)
    artifact_id = _store(app, content)
    _artifact_row, stored = _stored(app, artifact_id)
    run_tool = media_process.run_tool
    read: list[Path] = []

    async def reading(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        copy = Path(arguments[-1].removeprefix("file:"))
        read.append(copy)
        assert hashlib.sha256(copy.read_bytes()).hexdigest() == hashlib.sha256(content).hexdigest()
        # Replacing the stored file while ffprobe runs changes nothing it reads.
        _replace_keeping_size_and_time(stored, bytes([content[0] ^ 1]) + content[1:])
        return await run_tool(executable, arguments, **bounds)

    monkeypatch.setattr(video_probe, "run_tool", reading)
    response = await _probe(client, artifact_id)

    assert response.status_code == 200, response.text
    assert response.json()["artifact_sha256"] == hashlib.sha256(content).hexdigest()
    assert response.json()["video"]["codec"] == "h264"
    [copy] = read
    assert copy != stored
    assert copy.parent == app.state.services.artifacts.root
    assert not copy.exists()
    assert _copies_left(app) == []


@pytest.mark.parametrize("failure", ["no room", "store refuses the copy"])
async def test_a_copy_that_cannot_be_made_answers_with_its_code(
    client: AsyncClient,
    app: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    artifact_id = _store(app, _make(tmp_path, "clip.mp4", *PATTERN, "-t", "1", *H264))
    if failure == "no room":
        monkeypatch.setattr(artifacts, "available_bytes", lambda _anchor: 0)
    else:

        def refused(*_args: object) -> int:
            raise AnchoredDirectoryError(errno.EROFS)

        monkeypatch.setattr(artifacts, "create_entry", refused)

    async def no_probe(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ffprobe ran without a copy")

    monkeypatch.setattr(video_probe, "run_tool", no_probe)
    response = await _probe(client, artifact_id)

    # The stored video is intact, so it is never reported missing or corrupt.
    assert (response.status_code, response.json()["code"]) == (503, "video-probe-unavailable")
    assert _copies_left(app) == []


async def test_an_ffprobe_gone_since_it_was_found_is_reported_as_missing(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id = _store(app, _make(tmp_path, "clip.mp4", *PATTERN, "-t", "1", *H264))
    gone = MediaTool("ffprobe", tmp_path / "removed" / "ffprobe", "8.1.2", "0" * 64)

    async def found(_name: str) -> MediaTool:
        return gone

    monkeypatch.setattr(video_utility_api, "find_media_tool", found)
    response = await _probe(client, artifact_id)

    assert (response.status_code, response.json()["code"]) == (503, "media-tool-missing")
    assert _copies_left(app) == []


async def test_a_probe_cancelled_while_it_copies_waits_for_the_copy_and_leaves_none(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id = _store(app, _make(tmp_path, "clip.mp4", *PATTERN, "-t", "1", *H264))
    artifact, _path = _stored(app, artifact_id)
    ffprobe = await find_media_tool("ffprobe")
    copying, release, ended = threading.Event(), threading.Event(), threading.Event()
    copy_verified = ArtifactStore._copy_verified

    def held(self: ArtifactStore, *args: Any) -> str:
        copying.set()
        try:
            assert release.wait(30)
            return copy_verified(self, *args)
        finally:
            ended.set()

    monkeypatch.setattr(ArtifactStore, "_copy_verified", held)
    probing = asyncio.create_task(
        video_probe.probe_video(app.state.services.artifacts, artifact, ffprobe)
    )
    assert await asyncio.to_thread(copying.wait, 30)
    probing.cancel()
    await asyncio.sleep(0.2)

    # Cancelled, the probe still waits for its copy, which stops and removes itself.
    assert not probing.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await probing
    assert ended.is_set()
    assert _copies_left(app) == []


async def test_a_cancel_scope_around_a_copying_probe_ends_the_scope_and_not_the_task(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id = _store(app, _make(tmp_path, "clip.mp4", *PATTERN, "-t", "1", *H264))
    artifact, _path = _stored(app, artifact_id)
    ffprobe = await find_media_tool("ffprobe")
    release = threading.Event()
    copy_verified = ArtifactStore._copy_verified

    def held(self: ArtifactStore, *args: Any) -> str:
        assert release.wait(30)
        return copy_verified(self, *args)

    monkeypatch.setattr(ArtifactStore, "_copy_verified", held)
    # The copy ends a moment after the scope's deadline, while the probe waits for it.
    timer = threading.Timer(0.5, release.set)
    timer.start()
    try:
        with anyio.move_on_after(0.1) as scope:
            await video_probe.probe_video(app.state.services.artifacts, artifact, ffprobe)
    finally:
        timer.cancel()
        release.set()

    # The scope knows the cancellation for its own, so only the scope ends.
    assert scope.cancelled_caught
    assert _copies_left(app) == []


async def test_a_probe_cancelled_while_ffprobe_reads_leaves_no_copy_behind(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id = _store(app, _make(tmp_path, "clip.mp4", *PATTERN, "-t", "1", *H264))
    artifact, _path = _stored(app, artifact_id)
    ffprobe = await find_media_tool("ffprobe")
    run_tool = media_process.run_tool
    reading = asyncio.Event()

    async def signalling(*args: Any, **bounds: Any) -> Any:
        reading.set()
        return await run_tool(*args, **bounds)

    monkeypatch.setattr(video_probe, "run_tool", signalling)
    probing = asyncio.create_task(
        video_probe.probe_video(app.state.services.artifacts, artifact, ffprobe)
    )
    async with asyncio.timeout(20):
        await reading.wait()
    probing.cancel()

    with pytest.raises(asyncio.CancelledError):
        await probing
    assert _copies_left(app) == []


async def test_the_identity_is_the_version_reported_and_the_program_file_digest() -> None:
    tool = await find_media_tool("ffprobe")

    located = shutil.which("ffprobe")
    assert located is not None
    assert tool.executable == Path(located)
    assert tool.sha256 == hashlib.sha256(Path(located).read_bytes()).hexdigest()
    assert re.fullmatch(r"[0-9A-Za-z._+~-]{1,80}", tool.version)
    assert tool.record() == {
        "name": "ffprobe",
        "version": tool.version,
        "sha256": tool.sha256,
        "origin": "system",
    }


async def test_a_program_replaced_in_place_is_measured_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The program file is measured, never trusted from an earlier call. Its
    # version answer is given here, since a real program changed in place may
    # not start at all, and a launcher found on the path may not start from a
    # copy elsewhere.
    program = tmp_path / "ffprobe"
    program.write_bytes(b"first program")
    monkeypatch.setattr(shutil, "which", lambda _name: str(program))

    async def answers_as_ffprobe(*_args: object, **_kwargs: object) -> ToolResult:
        return ToolResult(0, b"ffprobe version 8.1.2 Copyright\n", b"")

    monkeypatch.setattr(media_tools, "run_tool", answers_as_ffprobe)
    first = await find_media_tool("ffprobe")
    _replace_keeping_size_and_time(program, b"other program")
    second = await find_media_tool("ffprobe")

    assert first.sha256 == hashlib.sha256(b"first program").hexdigest()
    assert second.sha256 == hashlib.sha256(b"other program").hexdigest()
    assert second.version == "8.1.2"


async def test_a_program_that_does_not_answer_as_the_tool_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The interpreter running these tests is a real program that rejects
    # ``-version``, so it stands in for anything else found under the tool's name.
    monkeypatch.setattr(shutil, "which", lambda _name: sys.executable)

    with pytest.raises(MediaToolUnavailable) as refused:
        await find_media_tool("ffprobe")

    assert refused.value.code == "media-tool-unreadable"


def _tool() -> MediaTool:
    return MediaTool("ffprobe", Path("ffprobe"), "1.0", "0" * 64)


def _artifact() -> Artifact:
    return Artifact(id="sha256:" + "a" * 64, sha256="a" * 64, media_type="video/mp4")


def _payload(
    *streams: object, duration: str | None = "2.5", start: str | None = None
) -> dict[str, Any]:
    container: dict[str, Any] = {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"}
    if duration is not None:
        container["duration"] = duration
    if start is not None:
        container["start_time"] = start
    return {"format": container, "streams": list(streams)}


def _video(**facts: Any) -> dict[str, Any]:
    return {
        "index": 0,
        "codec_type": "video",
        "codec_name": "h264",
        "width": 640,
        "height": 360,
        "r_frame_rate": "30/1",
        "avg_frame_rate": "30/1",
        **facts,
    }


def test_the_description_names_each_limit_it_finds() -> None:
    cover = _video(index=1, disposition={"attached_pic": 1})
    subtitle = {"index": 2, "codec_type": "subtitle", "codec_name": "mov_text"}
    probe = describe(
        _payload(
            _video(avg_frame_rate="24000/1001", side_data_list=[{"rotation": 45}]),
            cover,
            subtitle,
            duration=None,
            start="-0.021",
        ),
        _artifact(),
        _tool(),
    )

    # An edit list can start a video's timestamps just before zero.
    assert probe.start_seconds == -0.021
    # The cover picture is not the video, and both it and the subtitle are left out.
    assert probe.video.index == 0
    assert probe.omitted_streams == 2
    assert probe.video.frame_rate == "24000/1001"
    assert probe.video.frame_rate_form == "variable"
    assert probe.limits == ["video-duration-unknown", "video-rotation-unsupported"]
    assert (probe.can_save_frame, probe.can_trim, probe.can_keep_audio) == (False, False, True)


def test_a_long_video_or_a_huge_frame_is_described_but_not_offered() -> None:
    probe = describe(
        _payload(_video(width=9000, codec_name="H.264!"), duration=str(5 * 60 * 60)),
        _artifact(),
        _tool(),
    )

    assert probe.video.codec == "unknown"
    assert probe.limits == [
        "video-codec-unsupported",
        "video-duration-too-long",
        "video-frame-too-large",
    ]
    assert probe.can_save_frame is False


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ([], "video-probe-unreadable"),
        ({"format": {"format_name": "concat"}, "streams": []}, "video-probe-unreadable"),
        (_payload(_video(width=None)), "video-probe-unreadable"),
        (_payload(_video(), "not a stream"), "video-probe-unreadable"),
        (_payload(*[_video(index=index) for index in range(17)]), "video-probe-too-many-streams"),
        (
            _payload({"index": 0, "codec_type": "audio", "codec_name": "aac"}),
            "video-probe-no-video-stream",
        ),
    ],
)
def test_a_malformed_or_oversized_answer_is_refused(payload: object, code: str) -> None:
    with pytest.raises(VideoProbeRefused) as refused:
        describe(payload, _artifact(), _tool())

    assert refused.value.code == code
