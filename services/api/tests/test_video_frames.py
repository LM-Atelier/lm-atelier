"""Saving one frame of a stored video as a new picture, as a standalone durable job.

Fixtures are tiny test patterns made by the real ffmpeg that hosted CI
installs, so the frame is decoded by the same ffmpeg a person's computer runs.
"""

from __future__ import annotations

import asyncio
import errno
import importlib
import io
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select

from local_lm import artifacts, media_process, video_probe
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind, JobKind, JobStatus
from local_lm.models import Artifact, ArtifactLibraryEntry, Job

PATTERN = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10"]
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


def _module(name: str) -> Any:
    """A video utility module, imported where a case uses it rather than at collection."""

    return importlib.import_module(f"local_lm.{name}")


def _store(app: FastAPI, content: bytes, name: str = "walk.mp4") -> str:
    with SessionLocal() as session:
        artifact = app.state.services.artifacts.ingest_bytes(
            session, content, kind=ArtifactKind.VIDEO, media_type="video/mp4", original_name=name
        )
        session.commit()
        return str(artifact.id)


async def _finished(client: AsyncClient, job_id: str) -> dict[str, Any]:
    async with asyncio.timeout(60):
        while True:
            job = (await client.get(f"/api/video-utilities/jobs/{job_id}")).json()
            if job["status"] in {"complete", "failed", "cancelled"}:
                return dict(job)
            await asyncio.sleep(0.05)


async def test_a_saved_frame_is_the_one_shown_at_the_time_and_says_which(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "2", *H264))

    accepted = await client.post(
        f"/api/artifacts/{source}/video-frames", json={"requested_seconds": 0.43}
    )

    assert accepted.status_code == 202, accepted.text
    queued = accepted.json()
    assert queued["kind"] == "media_utility"
    job = await _finished(client, queued["id"])
    assert job["status"] == "complete", job
    result = job["result_json"]
    # Frames sit every tenth of a second; at 0.43 s a player shows the frame from 0.4 s.
    assert (result["requested_seconds"], result["actual_seconds"]) == (0.43, 0.4)
    assert (result["source_artifact_id"], result["action"]) == (source, "extract_frame")
    assert (result["width"], result["height"], result["rotation_applied"]) == (64, 48, 0)
    assert result["sample_aspect"] is None
    assert result["tool"]["name"] == "ffmpeg"
    assert result["tool"]["origin"] == "system"

    with SessionLocal() as session:
        picture = session.get(Artifact, result["result_artifact_id"])
        assert picture is not None
        assert (picture.kind, picture.media_type) == (ArtifactKind.IMAGE.value, "image/png")
        assert picture.original_name == "walk frame 0.400s.png"
        entry = session.scalar(
            select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == picture.id)
        )
        assert entry is not None and entry.state == "visible"
        path = app.state.services.artifacts.verified_path(picture)
    with Image.open(path) as image:
        assert (image.format, image.size) == ("PNG", (64, 48))


async def test_a_turned_video_saves_its_frame_turned_and_keeps_its_pixel_shape(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    _make(tmp_path, "wide.mp4", *PATTERN, "-t", "1", "-vf", "setsar=2/1", *H264)
    wide = str(tmp_path / "wide.mp4")
    content = _make(tmp_path, "turned.mp4", "-display_rotation", "90", "-i", wide, "-c", "copy")
    source = _store(app, content)

    accepted = await client.post(
        f"/api/artifacts/{source}/video-frames", json={"requested_seconds": 0}
    )
    job = await _finished(client, accepted.json()["id"])

    result = job["result_json"]
    assert (result["width"], result["height"]) == (48, 64)
    assert result["rotation_applied"] in (90, 270)
    assert result["sample_aspect"] == "2:1"


async def test_a_time_outside_the_video_or_a_video_without_frames_is_refused_before_queueing(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264))
    late = await client.post(f"/api/artifacts/{source}/video-frames", json={"requested_seconds": 5})
    assert (late.status_code, late.json()["code"]) == (422, "video-frame-time-outside")
    length = (await client.get(f"/api/artifacts/{source}/video-probe")).json()["duration_seconds"]
    past = await client.post(
        f"/api/artifacts/{source}/video-frames", json={"requested_seconds": length + 0.001}
    )
    assert (past.status_code, past.json()["code"]) == (422, "video-frame-time-outside")

    stills = _store(app, _make(tmp_path, "stills.mkv", *PATTERN, "-t", "1", "-c:v", "mjpeg"))
    refused = await client.post(
        f"/api/artifacts/{stills}/video-frames", json={"requested_seconds": 0}
    )
    assert (refused.status_code, refused.json()["code"]) == (422, "video-frame-not-offered")

    with SessionLocal() as session:
        kinds = [job.kind for job in session.query(Job).all()]
    assert JobKind.MEDIA_UTILITY.value not in kinds


async def test_a_frame_job_never_waits_for_the_generation_slot(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264))
    scheduler = app.state.services.scheduler

    # Image and video generation share the "primary" slot; it stays taken throughout.
    async with scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/artifacts/{source}/video-frames", json={"requested_seconds": 0}
        )
        job = await _finished(client, accepted.json()["id"])

    assert job["status"] == "complete"


async def test_the_utility_lane_holds_a_frame_job_until_it_is_resumed(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264))
    policy = (await client.get("/api/queue/lanes/utility")).json()
    paused = await client.post(
        "/api/queue/lanes/utility/pause-after-current",
        json={"expected_revision": policy["revision"], "idempotency_key": "pause-utility"},
    )
    assert paused.status_code == 200, paused.text
    assert paused.json()["dispatch_state"] == "paused"

    accepted = await client.post(
        f"/api/artifacts/{source}/video-frames", json={"requested_seconds": 0}
    )
    job_id = accepted.json()["id"]
    async with asyncio.timeout(30):
        while (await client.get(f"/api/video-utilities/jobs/{job_id}")).json()["phase"] != (
            "utility paused"
        ):
            await asyncio.sleep(0.05)
    held = (await client.get(f"/api/video-utilities/jobs/{job_id}")).json()
    assert held["status"] == "queued"
    activity = (await client.get("/api/queue/activity", params={"lane": "utility"})).json()
    assert [(item["owner_id"], item["label"], item["lane"]) for item in activity["items"]] == [
        (job_id, "Video utility", "utility")
    ]
    assert activity["lane_counts"]["utility"] == 1

    resumed = await client.post(
        "/api/queue/lanes/utility/resume",
        json={"expected_revision": paused.json()["revision"], "idempotency_key": "resume-utility"},
    )
    assert resumed.status_code == 200, resumed.text
    assert (await _finished(client, job_id))["status"] == "complete"


async def test_a_cancelled_frame_job_saves_no_picture(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264))
    video_utilities = _module("video_utilities")
    entered = asyncio.Event()
    decode = video_utilities.decode_frame

    async def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await asyncio.sleep(60)
        return await decode(*args, **kwargs)

    monkeypatch.setattr(video_utilities, "decode_frame", held)
    accepted = await client.post(
        f"/api/artifacts/{source}/video-frames", json={"requested_seconds": 0}
    )
    job_id = accepted.json()["id"]
    await asyncio.wait_for(entered.wait(), timeout=30)

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200, cancelled.text
    job = await _finished(client, job_id)
    assert (job["status"], job["result_json"]) == ("cancelled", {})
    with SessionLocal() as session:
        pictures = session.query(Artifact).filter(Artifact.kind == ArtifactKind.IMAGE.value).all()
    assert pictures == []


async def test_a_job_cancelled_while_its_picture_is_saved_ends_complete_after_the_save(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264))
    saving = threading.Event()
    saved: list[float] = []
    video_utilities = _module("video_utilities")
    finish = video_utilities.VideoUtilityManager._finish

    def slow_finish(self: Any, *args: Any) -> None:
        saving.set()
        time.sleep(0.5)
        finish(self, *args)
        saved.append(time.monotonic())

    monkeypatch.setattr(video_utilities.VideoUtilityManager, "_finish", slow_finish)
    accepted = await client.post(
        f"/api/artifacts/{source}/video-frames", json={"requested_seconds": 0}
    )
    job_id = accepted.json()["id"]
    assert await asyncio.to_thread(saving.wait, 30)

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
    answered = time.monotonic()

    # The save had begun, so it finishes before the cancellation answers, and
    # the job it completed is not then marked cancelled.
    assert saved and saved[0] <= answered
    assert (cancelled.status_code, cancelled.json()["code"]) == (409, "job-not-cancellable")
    job = await _finished(client, job_id)
    assert job["status"] == "complete", job
    assert job["result_json"]["result_artifact_id"]


async def test_a_job_with_no_room_for_its_copy_fails_with_its_code(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264))
    utilities = app.state.services.video_utilities
    with SessionLocal() as session:
        artifact = session.get(Artifact, source)
        assert artifact is not None
        job = utilities.stage_frame(session, artifact, 0.0)
        session.commit()
        job_id = job.id
    monkeypatch.setattr(artifacts, "available_bytes", lambda _anchor: 0)
    utilities.start(job_id)

    failed = await _finished(client, job_id)
    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "video-utility-copy-unavailable"},
    )
    assert list(app.state.services.artifacts.root.glob("tool-input-*")) == []


async def test_a_job_that_fails_unexpectedly_ends_failed_rather_than_running(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264))
    utilities = app.state.services.video_utilities
    with SessionLocal() as session:
        artifact = session.get(Artifact, source)
        assert artifact is not None
        job = utilities.stage_frame(session, artifact, 0.0)
        session.commit()
        job_id = job.id

    async def disk_full(*_args: object, **_kwargs: object) -> object:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(_module("video_utilities"), "probe_copy", disk_full)
    utilities.start(job_id)

    failed = await _finished(client, job_id)
    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "video-utility-failed"},
    )
    assert list(app.state.services.artifacts.root.glob("tool-input-*")) == []


async def test_a_job_whose_video_changed_fails_with_its_code_and_can_be_retried(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264))
    utilities = app.state.services.video_utilities
    with SessionLocal() as session:
        artifact = session.get(Artifact, source)
        assert artifact is not None
        job = utilities.stage_frame(session, artifact, 0.0)
        # The request names a digest the stored video does not have.
        job.payload_json = {**job.payload_json, "source_sha256": "0" * 64}
        session.commit()
        job_id = job.id
    utilities.start(job_id)

    failed = await _finished(client, job_id)
    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "video-utility-source-changed"},
    )

    with SessionLocal() as session:
        stored = session.get(Job, job_id)
        assert stored is not None
        stored.payload_json = {**stored.payload_json, "source_sha256": artifact.sha256}
        session.commit()
    retried = await client.post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 200, retried.text
    assert (await _finished(client, job_id))["status"] == "complete"


async def test_a_frame_job_reads_one_private_copy_for_both_of_its_tools(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264))
    store = app.state.services.artifacts
    with SessionLocal() as session:
        artifact = session.get(Artifact, source)
        assert artifact is not None
        stored = store.resolve(artifact)
    run_tool = media_process.run_tool
    read: list[Path] = []

    async def reading(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        read.extend(
            Path(item.removeprefix("file:")) for item in arguments if item.startswith("file:")
        )
        return await run_tool(executable, arguments, **bounds)

    monkeypatch.setattr(video_probe, "run_tool", reading)
    monkeypatch.setattr(_module("video_frames"), "run_tool", reading)
    accepted = await client.post(
        f"/api/artifacts/{source}/video-frames", json={"requested_seconds": 0}
    )
    job = await _finished(client, accepted.json()["id"])

    assert job["status"] == "complete", job
    # The request is checked against its own copy; the job then copies once
    # more and reads that one copy with ffprobe and with both ffmpeg runs.
    [checked, probed, *decoded] = read
    assert decoded == [probed, probed] and checked != probed
    assert stored not in read
    assert [path for path in read if path.exists()] == []
    assert list(store.root.glob("tool-input-*")) == []


async def test_a_frame_job_on_a_video_replaced_after_an_earlier_read_fails_unread(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264)
    source = _store(app, content)
    store = app.state.services.artifacts
    utilities = app.state.services.video_utilities
    with SessionLocal() as session:
        artifact = session.get(Artifact, source)
        assert artifact is not None
        job = utilities.stage_frame(session, artifact, 0.0)
        session.commit()
        job_id = job.id
    # Checked once already, then replaced with bytes of the same size and time.
    path = store.verified_path(artifact)
    status = path.stat()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))

    async def no_tool(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a tool ran on bytes that do not match the record")

    monkeypatch.setattr(video_probe, "run_tool", no_tool)
    monkeypatch.setattr(_module("video_frames"), "run_tool", no_tool)
    utilities.start(job_id)

    failed = await _finished(client, job_id)
    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "artifact-file-unreadable"},
    )
    with SessionLocal() as session:
        pictures = session.query(Artifact).filter(Artifact.kind == ArtifactKind.IMAGE.value).all()
    assert pictures == []
    assert list(store.root.glob("tool-input-*")) == []


async def test_an_interrupted_frame_job_is_started_again_from_its_request(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264))
    utilities = app.state.services.video_utilities
    with SessionLocal() as session:
        artifact = session.get(Artifact, source)
        assert artifact is not None
        job = utilities.stage_frame(session, artifact, 0.2)
        # As a previous run left it: running when the application stopped.
        job.status = JobStatus.INTERRUPTED.value
        session.commit()
        job_id = job.id

    utilities.recover()

    finished = await _finished(client, job_id)
    assert finished["status"] == "complete"
    assert finished["result_json"]["actual_seconds"] == 0.2


async def test_only_video_utility_jobs_are_read_through_the_utility_route(
    client: AsyncClient,
) -> None:
    missing = await client.get("/api/video-utilities/jobs/job_missing")
    assert (missing.status_code, missing.json()["code"]) == (404, "job-not-found")

    with SessionLocal() as session:
        other = Job(kind=JobKind.DOWNLOAD.value, status=JobStatus.QUEUED.value)
        session.add(other)
        session.commit()
        other_id = other.id
    refused = await client.get(f"/api/video-utilities/jobs/{other_id}")
    assert refused.status_code == 404


def test_a_frame_time_is_read_from_its_integer_timestamp() -> None:
    report = _module("video_frames").FRAME_REPORT

    # Older builds print the time to six significant digits; the integer is exact.
    line = b"[Parsed_showinfo_2 @ 0x1] n:   0 pts:1234533333 pts_time:1234.53 duration:33333"
    assert report.findall(line) == [b"1234533333"]
    many = b"n:   0 pts:      0 pts_time:0 x\nn:   1 pts: 100000 pts_time:0.1 x\n"
    assert report.findall(many) == [b"0", b"100000"]
    assert report.findall(b"n: 0 pts: N/A pts_time:N/A") == []


def _pixels(path_or_bytes: Path | bytes) -> bytes:
    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with Image.open(source) as image:
        return image.convert("RGB").tobytes()


def _source_frame(path: Path, index: int) -> bytes:
    """One frame of a video decoded from its very beginning, as the pixels a PNG of it holds."""

    answer = subprocess.run(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return _pixels(answer.stdout)


async def _saved(client: AsyncClient, app: FastAPI, source: str, at: float) -> dict[str, Any]:
    accepted = await client.post(
        f"/api/artifacts/{source}/video-frames", json={"requested_seconds": at}
    )
    assert accepted.status_code == 202, accepted.text
    job = await _finished(client, accepted.json()["id"])
    assert job["status"] == "complete", job
    return dict(job["result_json"])


def _saved_pixels(app: FastAPI, result: dict[str, Any]) -> bytes:
    with SessionLocal() as session:
        picture = session.get(Artifact, result["result_artifact_id"])
        assert picture is not None
        path = app.state.services.artifacts.verified_path(picture)
    return _pixels(Path(path))


@pytest.mark.parametrize(("at", "shown"), [(0.95, 0.9), (1.0, 0.9)])
async def test_the_end_of_a_video_saves_its_last_frame(
    client: AsyncClient, app: FastAPI, tmp_path: Path, at: float, shown: float
) -> None:
    # Inside the last frame's span, and at the very end, where a player that
    # has played to the end reports the video's length.
    source = _store(app, _make(tmp_path, "walk.mp4", *PATTERN, "-t", "1", *H264))
    length = (await client.get(f"/api/artifacts/{source}/video-probe")).json()["duration_seconds"]

    result = await _saved(client, app, source, min(at, length))

    assert result["actual_seconds"] == shown
    assert _saved_pixels(app, result) == _source_frame(tmp_path / "walk.mp4", 9)


async def test_a_picture_shorter_than_its_sound_shows_its_last_frame_to_the_end(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(
        tmp_path,
        "short.mp4",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=10:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=8000:duration=2",
        "-map",
        "0:v",
        "-map",
        "1:a",
        *H264,
        "-c:a",
        "aac",
    )
    source = _store(app, content, "short.mp4")
    length = (await client.get(f"/api/artifacts/{source}/video-probe")).json()["duration_seconds"]
    assert length == pytest.approx(2.0, abs=0.05)

    result = await _saved(client, app, source, 1.5)

    assert result["actual_seconds"] == 0.9
    assert _saved_pixels(app, result) == _source_frame(tmp_path / "short.mp4", 9)


@pytest.mark.parametrize(
    ("name", "encode", "at", "index"),
    [
        # Open keyframes: 1.9 s is shown before, and stored after, the keyframe at 2 s.
        (
            "open.mp4",
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "keyint=10:min-keyint=10:scenecut=0:bframes=3:b-adapt=0:open-gop=1",
            ],
            1.95,
            19,
        ),
        # No edit list: the keyframe at 2 s is stored before the B-frames shown ahead of it.
        (
            "unlisted.mp4",
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "keyint=10:min-keyint=10:scenecut=0:bframes=3:b-adapt=0",
                "-use_editlist",
                "0",
            ],
            1.95,
            19,
        ),
    ],
)
async def test_a_frame_stored_after_a_later_keyframe_is_still_the_one_saved(
    client: AsyncClient,
    app: FastAPI,
    tmp_path: Path,
    name: str,
    encode: list[str],
    at: float,
    index: int,
) -> None:
    source = _store(app, _make(tmp_path, name, *PATTERN, "-t", "4", *encode), name)
    start = (await client.get(f"/api/artifacts/{source}/video-probe")).json()["start_seconds"]

    result = await _saved(client, app, source, round(start + at, 6) - start)

    assert result["actual_seconds"] == pytest.approx(index / 10, abs=0.000001)
    assert _saved_pixels(app, result) == _source_frame(tmp_path / name, index)


async def test_a_frame_far_into_a_video_is_timed_to_the_microsecond(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    # Timestamps from 1234.5 s, where a time printed to six significant digits
    # names another frame.
    _make(
        tmp_path, "plain.mkv", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-t", "1", *H264
    )
    content = _make(
        tmp_path,
        "late.mkv",
        "-i",
        str(tmp_path / "plain.mkv"),
        "-c",
        "copy",
        "-output_ts_offset",
        "1234.5",
    )
    with SessionLocal() as session:
        artifact = app.state.services.artifacts.ingest_bytes(
            session,
            content,
            kind=ArtifactKind.VIDEO,
            media_type="video/x-matroska",
            original_name="late.mkv",
        )
        session.commit()
        source = str(artifact.id)

    result = await _saved(client, app, source, 0.05)

    # Matroska counts in milliseconds: the frame shown at 0.05 s begins at 0.033 s.
    assert result["actual_seconds"] == 0.033
    assert _saved_pixels(app, result) == _source_frame(tmp_path / "late.mkv", 1)
    with SessionLocal() as session:
        picture = session.get(Artifact, result["result_artifact_id"])
        assert picture is not None and picture.original_name == "late frame 0.033s.png"


def test_the_frame_png_is_checked_by_decoding_it() -> None:
    video_frames = _module("video_frames")
    with pytest.raises(video_frames.FrameRefused) as refused:
        video_frames._decoded_size(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    assert refused.value.code == "video-frame-unreadable"

    buffer = io.BytesIO()
    Image.new("RGB", (3, 2)).save(buffer, format="PNG")
    assert video_frames._decoded_size(buffer.getvalue()) == (3, 2)
