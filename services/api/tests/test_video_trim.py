"""Trimming a stored video into a new video, copied from a keyframe without re-encoding.

Fixtures are tiny test patterns made by the real ffmpeg that hosted CI
installs. What a case expects comes from how its fixture was made and from the
case's own ffmpeg and ffprobe runs on the stored result, never from the job's
record of itself.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from pydantic import ValidationError
from sqlalchemy import select

from local_lm import artifacts, video_probe, video_utilities
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind, JobKind, JobStatus
from local_lm.filesystem_links import open_entry
from local_lm.media_tools import MediaTool
from local_lm.models import Artifact, ArtifactLibraryEntry, Job

PATTERN = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10"]
TONE = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000"]
#: Keyframes on every whole second, with B-frames between them.
KEYED = [
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-g",
    "10",
    "-keyint_min",
    "10",
    "-sc_threshold",
    "0",
    "-bf",
    "2",
]


def _trim_module() -> Any:
    """The trim module, imported where a case uses it rather than at collection."""

    return importlib.import_module("local_lm.video_trim")


def _tool(name: str) -> str:
    executable = shutil.which(name)
    assert executable, f"these cases need a real {name} on PATH, as hosted CI provides"
    return executable


def _make(directory: Path, name: str, *arguments: str) -> bytes:
    target = directory / name
    subprocess.run(
        [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *arguments, str(target)],
        check=True,
        timeout=60,
    )
    return target.read_bytes()


def _walk(directory: Path) -> bytes:
    """Four seconds with sound, keyframes at 0, 1, 2 and 3 s."""

    return _make(
        directory, "walk.mp4", *PATTERN, *TONE, "-t", "4", *KEYED, "-c:a", "aac", "-shortest"
    )


def _change(path: Path, change: str, outside: Path) -> bool:
    """Change a tool's output the store holds sealed; False when the system refuses the change.

    Windows refuses every one while the output is held. Elsewhere each goes
    through, and each moves the file's change time, which nothing sets back.
    """

    before = path.stat()
    try:
        if change == "overwritten":
            with path.open("r+b") as written:
                written.write(b"not what was sealed")
        elif change == "replaced":
            # Another file with the very same bytes.
            replacement = outside / "replacement"
            replacement.write_bytes(path.read_bytes())
            try:
                replacement.replace(path)
            finally:
                replacement.unlink(missing_ok=True)
        else:
            # The same file, size and time; only its bytes differ.
            path.write_bytes(b"N" * before.st_size)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            after = path.stat()
            assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
    except PermissionError:
        return False
    return True


def _store(
    app: FastAPI, content: bytes, name: str = "walk.mp4", media_type: str = "video/mp4"
) -> str:
    with SessionLocal() as session:
        artifact = app.state.services.artifacts.ingest_bytes(
            session, content, kind=ArtifactKind.VIDEO, media_type=media_type, original_name=name
        )
        session.commit()
        return str(artifact.id)


def _path(app: FastAPI, artifact_id: str) -> Path:
    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact is not None
        return Path(app.state.services.artifacts.verified_path(artifact))


def _streams(path: Path) -> list[dict[str, Any]]:
    answer = subprocess.run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-count_packets",
            "-show_entries",
            "stream=index,codec_type,codec_name,start_time,duration,nb_read_packets",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return list(json.loads(answer.stdout)["streams"])


def _frames(
    path: Path, at: float | None = None, count: int = 1, *, stored: bool = False
) -> list[str]:
    """Digests of decoded frames in the order shown, from the first or from a time.

    ``stored`` reads an MP4 past its edit list, so frames it hides count too.
    """

    seek = [] if at is None else ["-ss", f"{at:.6f}"]
    past = ["-ignore_editlist", "1"] if stored else []
    answer = subprocess.run(
        [
            _tool("ffmpeg"),
            "-v",
            "error",
            "-noautorotate",
            *past,
            *seek,
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-frames:v",
            str(count),
            "-f",
            "framehash",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    lines = [line for line in answer.stdout.decode().splitlines() if not line.startswith("#")]
    return [line.rsplit(",", 1)[1].strip() for line in lines]


def _picture(path: Path, at: float | None = None) -> str:
    """The digest of a decoded frame: the first one shown, or the first at or after a time."""

    return _frames(path, at)[0]


def _shown_count(path: Path) -> int:
    answer = subprocess.run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return int(answer.stdout.decode().strip())


def _browser_fields(interface: str) -> set[str]:
    """The fields the browser declares for one interface of a trim's result."""

    types = Path(__file__).resolve().parents[3] / "apps" / "web" / "src" / "videoUtilityTypes.ts"
    block = re.search(
        rf"export interface {interface} \{{(.*?)\n\}}", types.read_text(encoding="utf-8"), re.S
    )
    assert block, interface
    return set(re.findall(r"^\s+(\w+)\??:", block.group(1), re.M))


def _leftovers(app: FastAPI) -> list[str]:
    return sorted(
        path.name
        for path in app.state.services.artifacts.root.iterdir()
        if path.name.startswith("tool-")
    )


def _utility_jobs() -> list[Job]:
    with SessionLocal() as session:
        return list(session.scalars(select(Job).where(Job.kind == JobKind.MEDIA_UTILITY.value)))


def _videos() -> list[str]:
    with SessionLocal() as session:
        return sorted(
            str(artifact.id)
            for artifact in session.scalars(
                select(Artifact).where(Artifact.kind == ArtifactKind.VIDEO.value)
            )
        )


async def _preview(client: AsyncClient, source: str, start: float, end: float, keep: bool) -> Any:
    return await client.get(
        f"/api/artifacts/{source}/video-trim-preview",
        params={"start_seconds": start, "end_seconds": end, "keep_audio": str(keep).lower()},
    )


async def _post(
    client: AsyncClient, source: str, start: float, end: float, keep: bool, shown: float
) -> Any:
    return await client.post(
        f"/api/artifacts/{source}/video-trims",
        json={
            "start_seconds": start,
            "end_seconds": end,
            "keep_audio": keep,
            "shown_start_seconds": shown,
        },
    )


async def _finished(client: AsyncClient, job_id: str) -> dict[str, Any]:
    async with asyncio.timeout(180):
        while True:
            job = (await client.get(f"/api/video-utilities/jobs/{job_id}")).json()
            if job["status"] in {"complete", "failed", "cancelled"}:
                return dict(job)
            await asyncio.sleep(0.05)


async def _trimmed(
    client: AsyncClient, source: str, start: float, end: float, keep: bool
) -> dict[str, Any]:
    """Check a cut, trim it from the start the check showed, and wait for the job."""

    checked = await _preview(client, source, start, end, keep)
    assert checked.status_code == 200, checked.text
    accepted = await _post(client, source, start, end, keep, checked.json()["start_seconds"])
    assert accepted.status_code == 202, accepted.text
    return await _finished(client, accepted.json()["id"])


def _stage(app: FastAPI, source: str, start: float, end: float, keep: bool, shown: float) -> str:
    utilities = app.state.services.video_utilities
    with SessionLocal() as session:
        artifact = session.get(Artifact, source)
        assert artifact is not None
        job = utilities.stage_trim(
            session,
            artifact,
            start_seconds=start,
            end_seconds=end,
            keep_audio=keep,
            shown_start_seconds=shown,
        )
        session.commit()
        return str(job.id)


async def test_a_trim_check_names_the_keyframe_before_the_start_and_queues_nothing(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _walk(tmp_path))

    checked = await _preview(client, source, 1.43, 2.7, True)

    assert checked.status_code == 200, checked.text
    preview = checked.json()
    assert (preview["keyframe_seconds"], preview["start_seconds"], preview["from_beginning"]) == (
        1.0,
        1.0,
        False,
    )
    assert (preview["requested_start_seconds"], preview["requested_end_seconds"]) == (1.43, 2.7)
    assert (preview["format"], preview["media_type"]) == ("mp4", "video/mp4")
    assert (preview["audio_streams_kept"], preview["omitted_streams"]) == (1, 0)
    assert preview["keeps_whole_video"] is False
    assert _utility_jobs() == []
    assert _leftovers(app) == []


async def test_a_trim_begins_at_the_shown_keyframe_and_records_what_it_measured(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _walk(tmp_path)
    source = _store(app, content)

    job = await _trimmed(client, source, 1.43, 2.7, True)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    assert job["phase"] == "Video trimmed"
    result = job["result_json"]
    made = _path(app, result["made_artifact_id"])
    # The new video's first shown picture is the source's keyframe at 1 s.
    assert _picture(made) == _picture(tmp_path / "walk.mp4", 1.0)
    assert (result["actual_start_seconds"], result["keyframe_seconds"]) == (1.0, 1.0)
    # The part asked for is all there, and a copy runs on at most a few frames.
    assert 2.7 <= result["actual_end_seconds"] <= 3.3
    [video, sound] = _streams(made)
    # It ends where a player stops showing it: the length the file states for its picture.
    assert result["actual_end_seconds"] == pytest.approx(
        1.0 + float(video["duration"]) - float(video["start_time"]), abs=0.001
    )
    assert (video["codec_type"], video["codec_name"], sound["codec_name"]) == (
        "video",
        "h264",
        "aac",
    )
    assert result["video"]["frames_stored"] == int(video["nb_read_packets"])
    # Nothing before the keyframe is stored, not even hidden behind the MP4's edit list.
    assert _frames(made, stored=True)[0] == _picture(tmp_path / "walk.mp4", 1.0)
    assert int(video["nb_read_packets"]) == _shown_count(made)
    # Every frame of the part asked for is shown, from the keyframe to the requested end.
    assert _frames(made, count=17) == _frames(tmp_path / "walk.mp4", 1.0, 17)
    assert set(result) == _browser_fields("VideoTrimResult")
    assert set(result["video"]) == _browser_fields("TrimmedVideoStream")
    assert {key for entry in result["audio"] for key in entry} == _browser_fields(
        "TrimmedAudioStream"
    )
    assert [entry["codec"] for entry in result["audio"]] == ["aac"]
    assert (result["from_artifact_id"], result["action"], result["format"]) == (
        source,
        "trim",
        "mp4",
    )
    assert (result["tool"]["name"], result["tool"]["origin"]) == ("ffmpeg", "system")
    assert result["measured_with"]["name"] == "ffprobe"
    assert result["in_library"] is True
    with SessionLocal() as session:
        trimmed = session.get(Artifact, result["made_artifact_id"])
        assert trimmed is not None
        assert (trimmed.kind, trimmed.media_type) == (ArtifactKind.VIDEO.value, "video/mp4")
        assert re.fullmatch(r"walk trim 1\.000-\d\.\d{3}s\.mp4", trimmed.original_name or "")
        entry = session.scalar(
            select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == trimmed.id)
        )
        assert entry is not None and entry.state == "visible"
    assert _path(app, source).read_bytes() == content
    assert _leftovers(app) == []


async def test_a_new_video_whose_end_cannot_be_sought_is_measured_from_all_its_packets(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sought far past its end, an MP4 file now and then lists no packet at all.
    source = _store(app, _walk(tmp_path))
    video_trim = _trim_module()
    run_tool = video_trim.run_tool
    reads: list[str] = []

    async def nothing_past_the_end(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        if "packet=pts_time,duration_time" in arguments:
            interval = (
                arguments[arguments.index("-read_intervals") + 1]
                if "-read_intervals" in arguments
                else ""
            )
            reads.append(interval)
            if interval:
                answer = await run_tool(executable, arguments, **bounds)
                return dataclasses.replace(answer, stdout=b"")
        return await run_tool(executable, arguments, **bounds)

    monkeypatch.setattr(video_trim, "run_tool", nothing_past_the_end)

    job = await _trimmed(client, source, 1.43, 2.7, True)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    assert reads == [f"{2 * video_probe.MAX_DURATION_SECONDS}%", ""]
    made = _path(app, job["result_json"]["made_artifact_id"])
    [video, _sound] = _streams(made)
    assert job["result_json"]["actual_end_seconds"] == pytest.approx(
        1.0 + float(video["duration"]) - float(video["start_time"]), abs=0.001
    )


async def test_a_matroska_video_is_measured_to_its_last_packet_when_a_seek_lists_part_of_it(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sought far past its end, a Matroska file can list a few of its last
    # packets and stop there, without an error.
    content = _make(
        tmp_path, "keyed.mkv", *PATTERN, *TONE, "-t", "4", *KEYED, "-c:a", "flac", "-shortest"
    )
    source = _store(app, content, "keyed.mkv", "video/x-matroska")
    video_trim = _trim_module()
    run_tool = video_trim.run_tool

    async def part_past_the_end(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        answer = await run_tool(executable, arguments, **bounds)
        if "packet=pts_time,duration_time" in arguments and "-read_intervals" in arguments:
            first = answer.stdout.splitlines(keepends=True)[:2]
            return dataclasses.replace(answer, stdout=b"".join(first))
        return answer

    monkeypatch.setattr(video_trim, "run_tool", part_past_the_end)

    job = await _trimmed(client, source, 1.43, 2.7, True)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    made = _path(app, job["result_json"]["made_artifact_id"])
    listed = subprocess.run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "0",
            "-show_entries",
            "packet=pts_time,duration_time",
            "-of",
            "csv=p=0",
            str(made),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.split()
    rows = [row.split(",")[:2] for row in listed]
    last = max(float(start) + (0.0 if length == "N/A" else float(length)) for start, length in rows)
    assert job["result_json"]["video"]["end_seconds"] == pytest.approx(last, abs=0.001)


async def test_a_video_whose_timestamps_start_late_is_cut_on_its_own_timeline(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    _make(tmp_path, "plain.mp4", *PATTERN, "-t", "4", *KEYED)
    content = _make(
        tmp_path,
        "late.mkv",
        "-i",
        str(tmp_path / "plain.mp4"),
        "-c",
        "copy",
        "-output_ts_offset",
        "5",
    )
    source = _store(app, content, "late.mkv", "video/x-matroska")

    checked = await _preview(client, source, 2.4, 3.0, False)
    assert checked.json()["keyframe_seconds"] == 2.0, checked.text
    job = await _trimmed(client, source, 2.4, 3.0, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    result = job["result_json"]
    assert result["format"] == "matroska"
    # Two seconds into the video, which is seven seconds into the file's own timestamps,
    # and every frame from there to the requested end.
    assert _frames(_path(app, result["made_artifact_id"]), count=10) == _frames(
        tmp_path / "late.mkv", 2.0, 10
    )


async def test_a_keyframe_far_into_a_files_timestamps_is_matched_to_the_microsecond(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    # From 1234.567 s, a time printed to six significant digits is off by
    # milliseconds, so the keyframe is matched by its integer timestamp.
    _make(tmp_path, "plain.mp4", *PATTERN, "-t", "4", *KEYED)
    content = _make(
        tmp_path,
        "late.mkv",
        "-i",
        str(tmp_path / "plain.mp4"),
        "-c",
        "copy",
        "-output_ts_offset",
        "1234.567",
    )
    source = _store(app, content, "late.mkv", "video/x-matroska")

    job = await _trimmed(client, source, 2.4, 3.0, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    assert job["result_json"]["keyframe_seconds"] == 2.0
    assert _frames(_path(app, job["result_json"]["made_artifact_id"]), count=10) == _frames(
        tmp_path / "late.mkv", 2.0, 10
    )


async def test_a_matroska_video_of_keyframes_only_is_cut_on_the_keyframe_asked_for(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    # No B-frames, so ffmpeg takes no step back, and a keyframe every tenth of a second.
    content = _make(
        tmp_path,
        "every-frame.mkv",
        *PATTERN,
        "-t",
        "3",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "1",
    )
    source = _store(app, content, "every-frame.mkv", "video/x-matroska")

    checked = (await _preview(client, source, 1.43, 2.0, False)).json()
    assert checked["keyframe_seconds"] == 1.4, checked
    job = await _trimmed(client, source, 1.43, 2.0, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    assert _picture(_path(app, job["result_json"]["made_artifact_id"])) == _picture(
        tmp_path / "every-frame.mkv", 1.4
    )


async def test_a_picture_that_begins_after_its_sound_is_copied_from_the_beginning(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(
        tmp_path,
        "sound-first.mkv",
        *TONE,
        "-itsoffset",
        "0.5",
        *PATTERN,
        "-t",
        "4",
        "-map",
        "1:v",
        "-map",
        "0:a",
        *KEYED,
        "-c:a",
        "flac",
        "-shortest",
    )
    source = _store(app, content, "sound-first.mkv", "video/x-matroska")

    early = (await _preview(client, source, 0, 1.2, True)).json()
    assert (early["from_beginning"], early["start_seconds"], early["keyframe_seconds"]) == (
        True,
        0.0,
        0.5,
    )
    job = await _trimmed(client, source, 0, 1.2, True)
    assert job["status"] == "complete", (job["result_json"], job["error"])
    made = _path(app, job["result_json"]["made_artifact_id"])
    [video, sound] = _streams(made)
    # The sound before the first picture is kept.
    assert float(sound["start_time"]) == 0.0 < float(video["start_time"])
    assert job["result_json"]["media_type"] == "video/x-matroska"
    # The picture is recorded from where it begins, at 0.5 s, to its measured end.
    end = job["result_json"]["actual_end_seconds"]
    assert 1.2 <= end <= 1.7
    assert _frames(made, count=7) == _frames(tmp_path / "sound-first.mkv", count=7)
    with SessionLocal() as session:
        trimmed = session.get(Artifact, job["result_json"]["made_artifact_id"])
        assert trimmed is not None
        assert trimmed.original_name == f"sound-first trim 0.000-{end:.3f}s.mkv"

    later = (await _preview(client, source, 2.3, 3.0, True)).json()
    assert (later["from_beginning"], later["keyframe_seconds"]) == (False, 1.5)

    before = await _preview(client, source, 0, 0.4, True)
    assert (before.status_code, before.json()["code"]) == (422, "video-trim-range-before-picture")


async def test_an_mp4_whose_picture_begins_after_its_sound_keeps_that_sound(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    # Made in two steps, the picture copied in half a second late, so the
    # delay is the container's own on every ffmpeg the cases run with.
    _make(tmp_path, "sound.m4a", *TONE, "-t", "4", "-c:a", "aac")
    _make(tmp_path, "picture.mp4", *PATTERN, "-t", "3.5", *KEYED)
    content = _make(
        tmp_path,
        "sound-first.mp4",
        "-i",
        str(tmp_path / "sound.m4a"),
        "-itsoffset",
        "0.5",
        "-i",
        str(tmp_path / "picture.mp4"),
        "-map",
        "1:v",
        "-map",
        "0:a",
        "-c",
        "copy",
    )
    [made_video, made_sound] = _streams(tmp_path / "sound-first.mp4")
    assert float(made_sound["start_time"]) == 0.0 < float(made_video["start_time"])
    source = _store(app, content, "sound-first.mp4")

    job = await _trimmed(client, source, 0, 1.2, True)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    assert job["result_json"]["from_beginning"] is True
    [video, sound] = _streams(_path(app, job["result_json"]["made_artifact_id"]))
    # Copied from the very beginning, so the picture keeps its place after the sound.
    assert float(sound["start_time"]) == 0.0 < float(video["start_time"])


async def test_a_single_keyframe_video_can_only_be_shortened(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(
        tmp_path,
        "one-key.mp4",
        *PATTERN,
        *TONE,
        "-t",
        "3",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "1000",
        "-sc_threshold",
        "0",
        "-bf",
        "2",
        "-c:a",
        "aac",
        "-shortest",
    )
    source = _store(app, content, "one-key.mp4")
    duration = (await client.get(f"/api/artifacts/{source}/video-probe")).json()["duration_seconds"]

    middle = (await _preview(client, source, 1.5, 2.5, True)).json()
    assert (middle["from_beginning"], middle["start_seconds"]) == (True, 0.0)
    whole = (await _preview(client, source, 0, duration, True)).json()
    assert whole["keeps_whole_video"] is True
    refused = await _post(client, source, 0, duration, True, 0.0)
    assert (refused.status_code, refused.json()["code"]) == (422, "video-trim-keeps-whole-video")
    assert refused.json()["detail"] == (
        "That would keep the whole video as it is. Choose an earlier end, or leave out the sound."
    )
    assert _utility_jobs() == []

    silent = await _trimmed(client, source, 0, duration, False)
    assert silent["status"] == "complete", silent
    assert [
        stream["codec_type"]
        for stream in _streams(_path(app, silent["result_json"]["made_artifact_id"]))
    ] == ["video"]


async def test_the_cut_is_bound_to_the_start_that_was_shown(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _walk(tmp_path))

    stale = await _post(client, source, 1.43, 2.7, True, 1.43)
    assert (stale.status_code, stale.json()["code"]) == (409, "video-trim-preview-stale")
    assert _utility_jobs() == []

    job_id = _stage(app, source, 1.43, 2.7, True, 0.9)
    app.state.services.video_utilities.start(job_id)
    moved = await _finished(client, job_id)
    assert (moved["status"], moved["result_json"]) == (
        "failed",
        {"failure_code": "video-trim-start-moved"},
    )
    assert moved["phase"] == "Video not trimmed"
    assert _videos() == [source]

    with SessionLocal() as session:
        stored = session.get(Job, job_id)
        assert stored is not None
        stored.payload_json = {**stored.payload_json, "shown_start_seconds": 1.0}
        session.commit()
    retried = await client.post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 200, retried.text
    assert (await _finished(client, job_id))["status"] == "complete"


async def test_ranges_and_sound_that_cannot_be_trimmed_are_refused_before_queueing(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _walk(tmp_path))
    for start, end in ((1.0, 4.5), (1.0, 1.05), (2.0, 1.0), (0.0, 0.0)):
        refused = await _preview(client, source, start, end, True)
        assert (refused.status_code, refused.json()["code"]) == (422, "video-trim-range-invalid"), (
            start,
            end,
        )

    stills = _store(
        app,
        _make(tmp_path, "stills.mkv", *PATTERN, "-t", "1", "-c:v", "mjpeg"),
        "stills.mkv",
        "video/x-matroska",
    )
    not_offered = await _preview(client, stills, 0.2, 0.8, False)
    assert (not_offered.status_code, not_offered.json()["code"]) == (422, "video-trim-not-offered")

    pcm = _store(
        app,
        _make(tmp_path, "pcm.mkv", *PATTERN, *TONE, "-t", "2", *KEYED, "-c:a", "pcm_s16le"),
        "pcm.mkv",
        "video/x-matroska",
    )
    for refused in (
        await _preview(client, pcm, 1.2, 1.8, True),
        await _post(client, pcm, 1.2, 1.8, True, 1.0),
    ):
        assert (refused.status_code, refused.json()["code"]) == (
            422,
            "video-trim-audio-not-offered",
        )
    silent = await _preview(client, pcm, 1.2, 1.8, False)
    assert silent.status_code == 200, silent.text
    assert silent.json()["format"] == "matroska"
    assert _utility_jobs() == []


async def test_dropping_the_sound_leaves_only_the_picture(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _walk(tmp_path))

    job = await _trimmed(client, source, 1.43, 2.7, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    assert (job["result_json"]["keep_audio"], job["result_json"]["audio"]) == (False, [])
    streams = _streams(_path(app, job["result_json"]["made_artifact_id"]))
    assert [stream["codec_type"] for stream in streams] == ["video"]


async def test_only_the_chosen_streams_are_kept_and_the_picture_comes_first(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(
        tmp_path,
        "two-sounds.mp4",
        *TONE,
        *PATTERN,
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:sample_rate=8000",
        "-map",
        "0:a",
        "-map",
        "1:v",
        "-map",
        "2:a",
        "-t",
        "4",
        *KEYED,
        "-c:a",
        "aac",
    )
    source = _store(app, content, "two-sounds.mp4")

    kept = await _trimmed(client, source, 1.43, 2.7, True)
    dropped = await _trimmed(client, source, 1.43, 2.7, False)

    assert (kept["status"], dropped["status"]) == ("complete", "complete")
    both = _streams(_path(app, kept["result_json"]["made_artifact_id"]))
    assert [(stream["codec_type"], stream["codec_name"]) for stream in both] == [
        ("video", "h264"),
        ("audio", "aac"),
        ("audio", "aac"),
    ]
    alone = _streams(_path(app, dropped["result_json"]["made_artifact_id"]))
    assert [stream["codec_type"] for stream in alone] == ["video"]


async def test_a_turned_video_stays_turned(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    _make(tmp_path, "wide.mp4", *PATTERN, "-t", "2", "-vf", "setsar=2/1", *KEYED)
    wide = str(tmp_path / "wide.mp4")
    content = _make(tmp_path, "turned.mp4", "-display_rotation", "90", "-i", wide, "-c", "copy")
    source = _store(app, content, "turned.mp4")
    before = (await client.get(f"/api/artifacts/{source}/video-probe")).json()

    job = await _trimmed(client, source, 1.2, 1.9, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    after = (
        await client.get(f"/api/artifacts/{job['result_json']['made_artifact_id']}/video-probe")
    ).json()
    assert after["video"]["rotation"] == before["video"]["rotation"] != 0
    assert after["video"]["sample_aspect"] == before["video"]["sample_aspect"] == "2:1"
    # Shown as the source is: its pixel shape applied, then its turn.
    assert (
        (after["video"]["display_width"], after["video"]["display_height"])
        == (
            before["video"]["display_width"],
            before["video"]["display_height"],
        )
        == (48, 128)
    )


async def test_an_output_that_differs_from_the_check_is_not_kept(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    video_trim = _trim_module()
    run_tool = video_trim.run_tool

    async def without_sound(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        if "-fs" in arguments:
            # The cut, with only its first stream chosen: the sound the check promised is left out.
            second = arguments.index("-map", arguments.index("-map") + 1)
            arguments = arguments[:second] + arguments[second + 2 :]
        return await run_tool(executable, arguments, **bounds)

    monkeypatch.setattr(video_trim, "run_tool", without_sound)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    app.state.services.video_utilities.start(job_id)

    failed = await _finished(client, job_id)
    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "video-trim-output-mismatch"},
    )
    assert _videos() == [source]
    assert _leftovers(app) == []


@pytest.mark.parametrize("change", ["overwritten", "replaced", "rewritten as it was"])
async def test_a_new_video_cannot_change_between_its_checks_and_being_kept(
    client: AsyncClient,
    app: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    source = _store(app, _walk(tmp_path))
    store = app.state.services.artifacts
    measure = _trim_module().check_trim
    changed: list[bool] = []

    async def then_changed(path: Path, *arguments: Any) -> Any:
        # The checks pass on the video that was made; then it is changed.
        measured = await measure(path, *arguments)
        changed.append(_change(path, change, tmp_path))
        return measured

    monkeypatch.setattr(video_utilities, "check_trim", then_changed)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    app.state.services.video_utilities.start(job_id)

    job = await _finished(client, job_id)
    assert changed == [os.name != "nt"]
    if os.name == "nt":
        # Windows refused the change while the new video was held, so it is
        # kept as it was checked.
        assert job["status"] == "complete", (job["result_json"], job["error"])
    else:
        assert (job["status"], job["result_json"]) == (
            "failed",
            {"failure_code": "video-trim-output-mismatch"},
        )
        assert _videos() == [source]
    assert _leftovers(app) == []
    assert [path.name for path in store.root.iterdir() if path.name.startswith("ingest-")] == []


async def test_a_new_video_changed_while_it_is_checked_and_changed_back_is_not_kept(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    measure = _trim_module().check_trim
    swapped: list[bool] = []

    async def while_checked(path: Path, *arguments: Any) -> Any:
        # A shorter real video is measured in its place; then the very same
        # bytes, size and time are put back before it is kept.
        original = path.read_bytes()
        before = path.stat()
        shorter = _make(
            tmp_path,
            "shorter.mp4",
            "-i",
            str(path),
            "-t",
            "0.8",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-write_tmcd",
            "0",
        )
        try:
            path.write_bytes(shorter)
        except PermissionError:
            swapped.append(False)
            return await measure(path, *arguments)
        swapped.append(True)
        try:
            return await measure(path, *arguments)
        finally:
            path.write_bytes(original)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    monkeypatch.setattr(video_utilities, "check_trim", while_checked)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    app.state.services.video_utilities.start(job_id)

    job = await _finished(client, job_id)
    assert swapped == [os.name != "nt"]
    if os.name == "nt":
        assert job["status"] == "complete", (job["result_json"], job["error"])
        result = job["result_json"]
        kept = _streams(_path(app, result["made_artifact_id"]))
        assert int(kept[0]["nb_read_packets"]) == result["video"]["frames_stored"]
    else:
        assert (job["status"], job["result_json"]) == (
            "failed",
            {"failure_code": "video-trim-output-mismatch"},
        )
        assert _videos() == [source]
    assert _leftovers(app) == []


async def test_a_copy_that_begins_on_another_keyframe_is_not_kept(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    video_trim = _trim_module()
    run_tool = video_trim.run_tool

    async def a_second_later(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        if "-fs" in arguments:
            at = arguments.index("-ss") + 1
            arguments = [*arguments[:at], f"{float(arguments[at]) + 1.0:.6f}", *arguments[at + 1 :]]
        return await run_tool(executable, arguments, **bounds)

    monkeypatch.setattr(video_trim, "run_tool", a_second_later)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    app.state.services.video_utilities.start(job_id)

    failed = await _finished(client, job_id)
    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "video-trim-start-unverified"},
    )
    assert _videos() == [source]
    assert _leftovers(app) == []


async def test_a_cut_that_writes_nothing_is_not_kept(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    video_trim = _trim_module()
    run_tool = video_trim.run_tool

    async def nothing_written(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        if "-fs" in arguments:
            # The cut answers as ffmpeg does, but writes nothing into its file.
            arguments = ["-hide_banner", "-version"]
        return await run_tool(executable, arguments, **bounds)

    monkeypatch.setattr(video_trim, "run_tool", nothing_written)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    app.state.services.video_utilities.start(job_id)

    failed = await _finished(client, job_id)
    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "video-trim-not-written"},
    )
    assert _videos() == [source]
    assert _leftovers(app) == []


async def test_streams_a_trim_leaves_out_make_a_whole_length_cut_a_real_change(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    words = tmp_path / "words.srt"
    words.write_text("1\n00:00:00,000 --> 00:00:01,000\nOne\n\n", encoding="utf-8")
    content = _make(
        tmp_path,
        "subtitled.mkv",
        *PATTERN,
        "-i",
        str(words),
        "-t",
        "3",
        "-map",
        "0:v",
        "-map",
        "1:s",
        *KEYED,
        "-c:s",
        "srt",
    )
    source = _store(app, content, "subtitled.mkv", "video/x-matroska")
    duration = (await client.get(f"/api/artifacts/{source}/video-probe")).json()["duration_seconds"]

    preview = (await _preview(client, source, 0, duration, False)).json()
    assert (preview["omitted_streams"], preview["keeps_whole_video"]) == (1, False)
    job = await _trimmed(client, source, 0, duration, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    made = _path(app, job["result_json"]["made_artifact_id"])
    assert [stream["codec_type"] for stream in _streams(made)] == ["video"]


async def test_a_trim_whose_video_is_already_in_recently_deleted_says_so(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _walk(tmp_path))
    first = await _trimmed(client, source, 1.43, 2.7, True)
    with SessionLocal() as session:
        entry = session.scalar(
            select(ArtifactLibraryEntry).where(
                ArtifactLibraryEntry.artifact_id == first["result_json"]["made_artifact_id"]
            )
        )
        assert entry is not None
        entry.state = "trashed"
        entry.deleted_at = datetime.now(UTC)
        entry.recovery_id = "recovery_" + "f" * 32
        entry.version += 1
        session.commit()

    second = await _trimmed(client, source, 1.43, 2.7, True)

    assert second["status"] == "complete", second
    # The same bytes are the same video, which is in Recently Deleted.
    assert second["result_json"]["made_artifact_id"] == first["result_json"]["made_artifact_id"]
    assert second["result_json"]["in_library"] is False


async def test_answers_that_differ_from_the_plan_are_refused(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    video_trim = _trim_module()
    run_tool = video_trim.run_tool
    fault = {"kind": ""}

    async def faulty(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        answer = await run_tool(executable, arguments, **bounds)
        asked = (
            arguments[arguments.index("-read_intervals") + 1]
            if "-read_intervals" in arguments
            else ""
        )
        if fault["kind"] == "later-keyframe" and asked.startswith("1.4305"):
            later = {"frames": [{"key_frame": 1, "pts_time": "2.000000"}]}
            return dataclasses.replace(answer, stdout=json.dumps(later).encode())
        if (
            fault["kind"] == "keyframe-time"
            and "-skip_frame" in arguments
            and "framehash" in arguments
        ):
            moved = re.sub(rb"\bpts:\s*-?\d+", b"pts:1100000", answer.stderr, count=1)
            return dataclasses.replace(answer, stderr=moved)
        if fault["kind"] == "narrower" and "-count_packets" in arguments:
            payload = json.loads(answer.stdout)
            payload["streams"][0]["width"] = 32
            return dataclasses.replace(answer, stdout=json.dumps(payload).encode())
        return answer

    monkeypatch.setattr(video_trim, "run_tool", faulty)

    # A keyframe after the start that is not the first one is not trusted.
    fault["kind"] = "later-keyframe"
    refused = await _preview(client, source, 1.43, 2.7, True)
    assert (refused.status_code, refused.json()["code"]) == (422, "video-trim-keyframe-unknown")
    for kind, code in (
        ("keyframe-time", "video-trim-keyframe-unknown"),
        ("narrower", "video-trim-output-mismatch"),
    ):
        fault["kind"] = kind
        job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
        app.state.services.video_utilities.start(job_id)
        failed = await _finished(client, job_id)
        assert (failed["status"], failed["result_json"]) == ("failed", {"failure_code": code}), kind
    assert _videos() == [source]
    assert _leftovers(app) == []


async def test_a_timecode_track_does_not_follow_the_part_into_the_new_video(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(
        tmp_path,
        "timecoded.mp4",
        *PATTERN,
        *TONE,
        "-t",
        "4",
        *KEYED,
        "-c:a",
        "aac",
        "-timecode",
        "01:00:00:00",
        "-shortest",
    )
    source = _store(app, content, "timecoded.mp4")
    assert [stream["codec_type"] for stream in _streams(tmp_path / "timecoded.mp4")] == [
        "video",
        "audio",
        "data",
    ]

    job = await _trimmed(client, source, 1.43, 2.7, True)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    made = _path(app, job["result_json"]["made_artifact_id"])
    assert [stream["codec_type"] for stream in _streams(made)] == ["video", "audio"]


async def test_a_matroska_video_that_starts_late_is_measured_from_its_start(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    _make(tmp_path, "plain.mp4", *PATTERN, "-t", "4", *KEYED)
    content = _make(
        tmp_path,
        "late.mkv",
        "-i",
        str(tmp_path / "plain.mp4"),
        "-c",
        "copy",
        "-output_ts_offset",
        "5",
    )
    source = _store(app, content, "late.mkv", "video/x-matroska")

    probe = (await client.get(f"/api/artifacts/{source}/video-probe")).json()
    assert probe["start_seconds"] == 5.0
    assert probe["duration_seconds"] == pytest.approx(4.0, abs=0.15)
    # A part past the last picture is refused, and the whole video is the whole video.
    past = await _preview(client, source, 4.5, 8.0, False)
    assert (past.status_code, past.json()["code"]) == (422, "video-trim-range-invalid")
    whole = (await _preview(client, source, 0, probe["duration_seconds"], False)).json()
    assert whole["keeps_whole_video"] is True


async def test_a_part_that_starts_on_an_open_keyframe_is_not_kept(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(
        tmp_path,
        "open.mkv",
        *PATTERN,
        "-t",
        "4",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-x264-params",
        "keyint=10:min-keyint=10:scenecut=0:bframes=3:open-gop=1",
    )
    source = _store(app, content, "open.mkv", "video/x-matroska")

    job = await _trimmed(client, source, 2.43, 3.5, False)

    assert (job["status"], job["result_json"]) == (
        "failed",
        {"failure_code": "video-trim-start-incomplete"},
    )
    assert _videos() == [source]
    assert _leftovers(app) == []


async def test_a_webm_stays_a_webm(client: AsyncClient, app: FastAPI, tmp_path: Path) -> None:
    content = _make(
        tmp_path,
        "clip.webm",
        *PATTERN,
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000",
        "-t",
        "3",
        "-c:v",
        "libvpx-vp9",
        "-g",
        "10",
        "-deadline",
        "realtime",
        "-cpu-used",
        "8",
        "-c:a",
        "libopus",
        "-shortest",
    )
    source = _store(app, content, "clip.webm", "video/webm")

    job = await _trimmed(client, source, 1.2, 2.5, True)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    assert (job["result_json"]["format"], job["result_json"]["media_type"]) == (
        "webm",
        "video/webm",
    )
    made = _path(app, job["result_json"]["made_artifact_id"])
    assert b"webm" in made.read_bytes()[:64]
    with SessionLocal() as session:
        trimmed = session.get(Artifact, job["result_json"]["made_artifact_id"])
        assert trimmed is not None and (trimmed.original_name or "").endswith(".webm")


async def test_a_transparent_webm_is_measured_to_its_last_picture(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    # Each picture's transparency travels beside it as side data.
    content = _make(
        tmp_path,
        "clear.webm",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=10,format=yuva420p",
        "-t",
        "3",
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-g",
        "10",
        "-deadline",
        "realtime",
        "-cpu-used",
        "8",
    )
    source = _store(app, content, "clear.webm", "video/webm")

    job = await _trimmed(client, source, 1.2, 2.5, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    result = job["result_json"]
    listed = subprocess.run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "0",
            "-show_entries",
            "packet=pts_time,duration_time",
            "-of",
            "csv=p=0",
            str(_path(app, result["made_artifact_id"])),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    rows = [line.strip().split(",") for line in listed.splitlines() if line.strip()]
    assert rows and all(row[2:] == [""] for row in rows), rows
    last = max(float(row[0]) + float(row[1]) for row in rows)
    assert result["video"]["end_seconds"] == pytest.approx(last, abs=0.000001)
    assert result["actual_end_seconds"] == pytest.approx(1.0 + last, abs=0.000001)


async def test_a_cancelled_trim_keeps_no_video_and_no_working_files(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    entered = asyncio.Event()
    cut = _trim_module().cut_video

    async def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await asyncio.sleep(60)
        return await cut(*args, **kwargs)

    monkeypatch.setattr(video_utilities, "cut_video", held)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    app.state.services.video_utilities.start(job_id)
    await asyncio.wait_for(entered.wait(), timeout=120)
    assert len(_leftovers(app)) == 2

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200, cancelled.text
    job = await _finished(client, job_id)
    assert (job["status"], job["result_json"]) == ("cancelled", {})
    assert _videos() == [source]
    assert _leftovers(app) == []


async def test_a_trim_cancelled_while_its_new_video_is_sealed_stops_and_keeps_nothing(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    sealing = threading.Event()
    stopped: list[bool] = []
    seal = artifacts.ArtifactStore._seal_tool_output

    def held(self: Any, output: Any, stop: threading.Event) -> Any:
        sealing.set()
        stopped.append(stop.wait(120))
        try:
            return seal(self, output, stop)
        except BaseException as exc:
            stopped.append(isinstance(exc, artifacts._CopyAbandoned))
            raise

    monkeypatch.setattr(artifacts.ArtifactStore, "_seal_tool_output", held)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    app.state.services.video_utilities.start(job_id)
    assert await asyncio.to_thread(sealing.wait, 120)

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200, cancelled.text
    job = await _finished(client, job_id)
    assert (job["status"], job["result_json"]) == ("cancelled", {})
    assert stopped == [True, True]
    assert _videos() == [source]
    assert _leftovers(app) == []


async def test_a_trim_cancelled_while_its_video_is_saved_ends_complete_after_the_save(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    saving = threading.Event()
    saved: list[float] = []
    finish = video_utilities.VideoUtilityManager._finish

    def slow_finish(self: Any, *args: Any) -> None:
        saving.set()
        time.sleep(0.5)
        finish(self, *args)
        saved.append(time.monotonic())

    monkeypatch.setattr(video_utilities.VideoUtilityManager, "_finish", slow_finish)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    app.state.services.video_utilities.start(job_id)
    assert await asyncio.to_thread(saving.wait, 120)

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
    answered = time.monotonic()

    assert saved and saved[0] <= answered
    assert (cancelled.status_code, cancelled.json()["code"]) == (409, "job-not-cancellable")
    job = await _finished(client, job_id)
    assert job["status"] == "complete", (job["result_json"], job["error"])
    assert _leftovers(app) == []


async def test_an_interrupted_trim_is_started_again_from_its_request(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _walk(tmp_path))
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    with SessionLocal() as session:
        stored = session.get(Job, job_id)
        assert stored is not None
        # As a previous run left it: running when the application stopped.
        stored.status = JobStatus.INTERRUPTED.value
        session.commit()

    app.state.services.video_utilities.recover()

    finished = await _finished(client, job_id)
    assert finished["status"] == "complete", finished
    assert finished["result_json"]["keyframe_seconds"] == 1.0


async def test_the_utility_lane_holds_a_trim_and_the_generation_slot_never_does(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _walk(tmp_path))
    policy = (await client.get("/api/queue/lanes/utility")).json()
    paused = await client.post(
        "/api/queue/lanes/utility/pause-after-current",
        json={"expected_revision": policy["revision"], "idempotency_key": "pause-trims"},
    )
    assert paused.status_code == 200, paused.text

    accepted = await _post(client, source, 1.43, 2.7, True, 1.0)
    job_id = accepted.json()["id"]
    async with asyncio.timeout(30):
        while (await client.get(f"/api/video-utilities/jobs/{job_id}")).json()["phase"] != (
            "utility paused"
        ):
            await asyncio.sleep(0.05)
    assert (await client.get(f"/api/video-utilities/jobs/{job_id}")).json()["status"] == "queued"

    # Image and video generation share the "primary" slot; it stays taken throughout.
    async with app.state.services.scheduler.lease("primary"):
        resumed = await client.post(
            "/api/queue/lanes/utility/resume",
            json={
                "expected_revision": paused.json()["revision"],
                "idempotency_key": "resume-trims",
            },
        )
        assert resumed.status_code == 200, resumed.text
        assert (await _finished(client, job_id))["status"] == "complete"


async def test_a_trim_reads_one_private_copy_and_writes_one_file_the_store_made(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    stored = _path(app, source)
    video_trim = _trim_module()
    calls: list[list[str]] = []
    run_tool = video_trim.run_tool

    async def recording(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        calls.append(list(arguments))
        return await run_tool(executable, arguments, **bounds)

    monkeypatch.setattr(video_probe, "run_tool", recording)
    monkeypatch.setattr(video_trim, "run_tool", recording)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    app.state.services.video_utilities.start(job_id)

    assert (await _finished(client, job_id))["status"] == "complete"
    files = [
        [item.removeprefix("file:") for item in call if item.startswith("file:")] for call in calls
    ]
    copy, output = Path(files[0][0]), Path(files[-1][0])
    assert re.fullmatch(r"tool-input-[0-9a-f]{32}\.tmp", copy.name)
    assert re.fullmatch(r"tool-output-[0-9a-f]{32}\.tmp", output.name)
    # Its end is read from past the end, and from the start only when that read
    # lists nothing, which ffprobe does now and then for the same file.
    ends = [call for call in calls if "packet=pts_time,duration_time" in call]
    assert 1 <= len(ends) <= 2 and "-read_intervals" in ends[0]
    assert all("-read_intervals" not in call for call in ends[1:])
    # The probe, both keyframe lookups and the keyframe's decode read the copy;
    # the cut reads the copy and writes the output; then the output is read for
    # its streams, its end, its first packets, its first shown frame and its first
    # stored frame.
    assert [[Path(item) for item in names] for names in files] == [
        [copy],
        [copy],
        [copy],
        [copy],
        [copy, output],
        *[[output]] * (4 + len(ends)),
    ]
    assert all(Path(item) != stored for names in files for item in names)
    for call in calls:
        first = next(index for index, item in enumerate(call) if item.startswith("file:"))
        assert call[call.index("-protocol_whitelist") + 1] == "file"
        assert call[call.index("-format_whitelist") + 1] == video_probe.DEMUXERS
        assert call.index("-protocol_whitelist") < first and call.index("-format_whitelist") < first
    cut = calls[4]
    for flag, value in (("-c", "copy"), ("-map_chapters", "-1"), ("-f", "mp4")):
        assert cut[cut.index(flag) + 1] == value
    assert [cut[index + 1] for index, item in enumerate(cut) if item == "-map"] == ["0:0", "0:1"]
    assert {"-nostdin", "-fs", "-y"} <= set(cut)
    assert not copy.exists() and not output.exists()
    assert _leftovers(app) == []


async def test_a_trim_with_no_room_for_its_working_files_fails_with_its_code(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    monkeypatch.setattr(artifacts, "available_bytes", lambda _anchor: 0)
    app.state.services.video_utilities.start(job_id)

    failed = await _finished(client, job_id)

    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "video-utility-copy-unavailable"},
    )
    assert "room" in failed["error"]
    assert _leftovers(app) == []


async def test_a_utility_request_with_an_unknown_action_fails_as_changed(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _walk(tmp_path))
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    with SessionLocal() as session:
        stored = session.get(Job, job_id)
        assert stored is not None
        stored.payload_json = {**stored.payload_json, "action": "rotate"}
        session.commit()
    app.state.services.video_utilities.start(job_id)

    failed = await _finished(client, job_id)

    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "video-utility-source-changed"},
    )
    assert failed["phase"] == "Not finished"


async def test_a_file_made_for_a_tool_keeps_only_what_the_store_allows_and_then_goes(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = app.state.services.artifacts

    async with store.tool_output(maximum_bytes=8) as output:
        assert output.path.parent == store.root
        assert re.fullmatch(r"tool-output-[0-9a-f]{32}\.tmp", output.path.name)
        assert output.path.read_bytes() == b""
        output.path.write_bytes(b"123456789")
        with pytest.raises(artifacts.ArtifactCopyUnavailable):
            async with store.sealed_tool_output(output):
                pass
        output.path.write_bytes(b"12345678")
        assert store.tool_output_size(output) == 8
        async with store.sealed_tool_output(output) as seal:
            with SessionLocal() as session:
                kept = store.ingest_tool_output(
                    session,
                    output,
                    seal=seal,
                    kind=ArtifactKind.VIDEO,
                    media_type="video/mp4",
                    original_name="a.mp4",
                    metadata={},
                )
                session.commit()
                kept_path = Path(store.verified_path(kept))
    assert kept_path.read_bytes() == b"12345678"
    assert not output.path.exists()

    with pytest.raises(RuntimeError):
        async with store.tool_output(maximum_bytes=8) as abandoned:
            raise RuntimeError("the tool failed")
    assert not abandoned.path.exists()

    with pytest.raises(ValueError):
        store.tool_output_size(artifacts.ToolOutput(store.root / "made-elsewhere.tmp", 8))

    monkeypatch.setattr(
        artifacts, "available_bytes", lambda _anchor: 2 * 8 + artifacts._COPY_HEADROOM - 1
    )
    with pytest.raises(artifacts.ArtifactCopyUnavailable):
        async with store.tool_output(maximum_bytes=8):
            pass
    assert [
        path.name for path in store.root.iterdir() if path.name.startswith("tool-output-")
    ] == []


async def test_a_sealed_tool_output_is_held_unchanged_and_kept_only_as_it_was_sealed(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store: artifacts.ArtifactStore = app.state.services.artifacts
    sealed = b"the sealed bytes"

    def keep(output: artifacts.ToolOutput, seal: artifacts.ToolOutputSeal) -> Artifact:
        with SessionLocal() as session:
            kept = store.ingest_tool_output(
                session,
                output,
                seal=seal,
                kind=ArtifactKind.VIDEO,
                media_type="video/mp4",
                original_name="a.mp4",
                metadata={},
            )
            session.commit()
            return kept

    async with store.tool_output(maximum_bytes=64) as output:
        output.path.write_bytes(sealed)
        async with store.sealed_tool_output(output) as seal:
            assert (seal.sha256, seal.size) == (hashlib.sha256(sealed).hexdigest(), len(sealed))
            assert keep(output, seal).sha256 == seal.sha256
        # The block's end lets the file go.
        output.path.write_bytes(b"let go")

    # Each change while it is sealed: Windows refuses it, and elsewhere keeping
    # refuses the changed file.
    for change in ("overwritten", "replaced", "rewritten as it was"):
        async with store.tool_output(maximum_bytes=64) as output:
            output.path.write_bytes(sealed)
            async with store.sealed_tool_output(output) as seal:
                assert _change(output.path, change, tmp_path) is (os.name != "nt"), change
                if os.name == "nt":
                    assert keep(output, seal).sha256 == seal.sha256
                else:
                    with pytest.raises(artifacts.ToolOutputChanged, match="after it was sealed"):
                        keep(output, seal)

    # A file that grows while it is read: Windows refuses the growth, and
    # elsewhere it is not sealed.
    async with store.tool_output(maximum_bytes=64) as output:
        output.path.write_bytes(sealed)
        read = artifacts._BoundedRead.readinto
        grown: list[bool] = []

        def growing(self: Any, buffer: Any) -> int:
            if not grown:
                try:
                    with output.path.open("ab") as more:
                        more.write(b"!")
                except PermissionError:
                    grown.append(False)
                else:
                    grown.append(True)
            return read(self, buffer)

        with monkeypatch.context() as patch:
            patch.setattr(artifacts._BoundedRead, "readinto", growing)
            if os.name == "nt":
                async with store.sealed_tool_output(output) as seal:
                    assert seal.size == len(sealed)
            else:
                with pytest.raises(artifacts.ToolOutputChanged, match="while it was sealed"):
                    async with store.sealed_tool_output(output):
                        pass
        assert grown == [os.name != "nt"]

    # One still open for writing elsewhere cannot be held unchanged on Windows;
    # other systems have no sharing modes, and its change time binds it there.
    async with store.tool_output(maximum_bytes=64) as output:
        output.path.write_bytes(sealed)
        with output.path.open("r+b"):
            if os.name == "nt":
                with pytest.raises(artifacts.ToolOutputChanged, match="held unchanged"):
                    async with store.sealed_tool_output(output):
                        pass
            else:
                async with store.sealed_tool_output(output) as seal:
                    assert seal.size == len(sealed)

    # Bytes without the sealed digest are refused at the end of the read,
    # before anything is published under them.
    other = b"other sealed bytes"
    async with store.tool_output(maximum_bytes=64) as output:
        output.path.write_bytes(other)
        async with store.sealed_tool_output(output) as seal:
            wrong = dataclasses.replace(seal, sha256=hashlib.sha256(sealed).hexdigest())
            with pytest.raises(artifacts.ToolOutputChanged, match="bytes that were sealed"):
                keep(output, wrong)
    digest = hashlib.sha256(other).hexdigest()
    with SessionLocal() as session:
        assert session.scalar(select(Artifact).where(Artifact.sha256 == digest)) is None
    assert not (store.root / digest[:2] / digest[2:4] / digest).exists()
    assert [
        path.name for path in store.root.iterdir() if path.name.startswith(("ingest-", "tool-"))
    ] == []


@pytest.mark.parametrize("mode_change", ["ignored", "refused"])
async def test_a_file_system_without_its_own_change_time_keeps_no_new_video(
    client: AsyncClient,
    app: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode_change: str,
) -> None:
    source = _store(app, _walk(tmp_path))
    native = os

    class FatStatus:
        """A file's state as Linux reports it on FAT: one time, in two-second steps."""

        def __init__(self, status: os.stat_result) -> None:
            self._status = status
            self.st_mtime_ns = status.st_mtime_ns // 2_000_000_000 * 2_000_000_000
            self.st_ctime_ns = self.st_mtime_ns

        def __getattr__(self, name: str) -> Any:
            return getattr(self._status, name)

    class PosixOnFat:
        """No sharing modes, and a mode change leaves the change time where it was."""

        name = "posix"

        def fstat(self, descriptor: int) -> Any:
            return FatStatus(native.fstat(descriptor))

        def fchmod(self, descriptor: int, mode: int) -> None:
            if mode_change == "refused":
                raise PermissionError("this file system keeps no file modes")

        def __getattr__(self, name: str) -> Any:
            return getattr(native, name)

    monkeypatch.setattr(artifacts, "os", PosixOnFat())
    monkeypatch.setattr(artifacts, "open_entry_unshared", open_entry)
    monkeypatch.setattr(artifacts, "_CHANGE_TIME_PROOF_SECONDS", 0.2)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.0)
    app.state.services.video_utilities.start(job_id)

    job = await _finished(client, job_id)
    assert (job["status"], job["result_json"]) == (
        "failed",
        {"failure_code": "video-trim-output-mismatch"},
    )
    assert _videos() == [source]
    assert _leftovers(app) == []


@pytest.mark.skipif(os.name == "nt", reason="Windows holds the file by its sharing modes")
async def test_a_seal_rests_on_a_change_time_it_has_seen_move(app: FastAPI) -> None:
    store: artifacts.ArtifactStore = app.state.services.artifacts
    async with store.tool_output(maximum_bytes=64) as output:
        output.path.write_bytes(b"the sealed bytes")
        before = output.path.stat()
        async with store.sealed_tool_output(output) as seal:
            after = output.path.stat()
            assert seal.identity[-1] == after.st_ctime_ns > before.st_ctime_ns
            assert time.time_ns() > after.st_ctime_ns
            # A change to nothing but the change time is still a change.
            os.chmod(output.path, after.st_mode & 0o777)
            with (
                SessionLocal() as session,
                pytest.raises(artifacts.ToolOutputChanged, match="after it was sealed"),
            ):
                store.ingest_tool_output(
                    session,
                    output,
                    seal=seal,
                    kind=ArtifactKind.VIDEO,
                    media_type="video/mp4",
                    original_name="a.mp4",
                    metadata={},
                )


async def test_a_caller_cancelled_while_its_output_file_is_made_leaves_none(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = app.state.services.artifacts
    entered, release, made = threading.Event(), threading.Event(), threading.Event()
    create = store._create_tool_output

    def slowly(maximum_bytes: int) -> str:
        entered.set()
        release.wait(10)
        try:
            return str(create(maximum_bytes))
        finally:
            made.set()

    monkeypatch.setattr(store, "_create_tool_output", slowly)

    async def use() -> None:
        async with store.tool_output(maximum_bytes=8):
            pass

    task = asyncio.create_task(use())
    assert await asyncio.to_thread(entered.wait, 10)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(made.wait, 10)

    assert _leftovers(app) == []


def test_a_tool_output_is_read_no_further_than_its_bound(tmp_path: Path) -> None:
    written = tmp_path / "grown.bin"
    written.write_bytes(b"12345")

    with io.FileIO(written, "r") as source, pytest.raises(artifacts.ArtifactCopyUnavailable):
        reader = io.BufferedReader(artifacts._BoundedRead(source, 4))
        reader.read()


def test_keyframe_answers_that_cannot_be_trusted_are_refused() -> None:
    video_trim = _trim_module()

    def answer(*frames: dict[str, object]) -> bytes:
        return json.dumps({"frames": list(frames)}).encode()

    assert video_trim.keyframe_time(answer({"key_frame": 1, "pts_time": "1.000000"})) == 1.0
    assert (
        video_trim.keyframe_time(
            answer({"key_frame": 1, "pts_time": "N/A", "best_effort_timestamp_time": "2.5"})
        )
        == 2.5
    )
    for refused in (
        b"not json",
        answer(),
        answer({"key_frame": 0, "pts_time": "1.0"}),
        answer({"key_frame": 1, "pts_time": "nan"}),
        answer({"key_frame": 1, "pts_time": "1" * 33}),
        answer({"key_frame": 1, "pts_time": "99999999"}),
    ):
        with pytest.raises(video_trim.TrimRefused) as caught:
            video_trim.keyframe_time(refused)
        assert caught.value.code == "video-trim-keyframe-unknown"


def test_b_frames_and_frame_report_times_are_read_strictly() -> None:
    video_trim = _trim_module()

    def answer(*streams: dict[str, object]) -> bytes:
        return json.dumps({"streams": list(streams)}).encode()

    assert video_trim.b_frames_stated(answer({"has_b_frames": 2}, {})) is True
    assert video_trim.b_frames_stated(answer({"has_b_frames": 0}, {"has_b_frames": 0})) is False
    assert video_trim.b_frames_stated(answer({})) is False
    for refused in (
        b"not json",
        b"{}",
        answer({"has_b_frames": "2"}),
        answer({"has_b_frames": True}),
    ):
        with pytest.raises(video_trim.TrimRefused):
            video_trim.b_frames_stated(refused)

    # Frame times are read in whole microseconds, so two readings of one time agree to that.
    assert video_trim._same_time(1234.567001, 1234.567)
    assert video_trim._same_time(1.0, 1.0000015)
    assert not video_trim._same_time(1.0001, 1.0)
    assert not video_trim._same_time(1234.57, 1234.567)


def test_a_start_is_whole_only_when_no_early_picture_precedes_its_keyframe() -> None:
    video_trim = _trim_module()

    video_trim.leading_pictures_complete(b"0.000000,K__\n0.300000,___\n0.100000,___\n")
    # Packets with side data, such as a transparent picture's, list one more field, empty.
    video_trim.leading_pictures_complete(b"0.000000,K__,\r\n0.100000,___,\r\n")
    for refused in (b"0.000000,K__\nN/A,___\n", b"0.500000,K__\n0.400000,___\n", b"0.1,___\n"):
        with pytest.raises(video_trim.TrimRefused) as caught:
            video_trim.leading_pictures_complete(refused)
        assert caught.value.code == "video-trim-start-incomplete"
    for unreadable in (b"", b"0.000000,K__,x\n", b"0.000000\n"):
        with pytest.raises(video_trim.TrimRefused) as caught:
            video_trim.leading_pictures_complete(unreadable)
        assert caught.value.code == "video-trim-unmeasured"


def test_a_frame_digest_and_a_last_packet_end_are_read_strictly() -> None:
    video_trim = _trim_module()
    digest = "ab" * 32

    assert (
        video_trim.frame_digest(
            f"#format: frame checksums\n0, 10, 10, 1, 4608, {digest}\n".encode()
        )
        == digest
    )
    assert (
        video_trim.frame_digest(f"0, 1, 1, 1, 9, {digest}\n0, 2, 2, 1, 9, {digest}\n".encode())
        is None
    )
    assert video_trim.frame_digest(b"0, 1, 1, 1, 9, not-a-digest\n") is None

    assert (
        video_trim.last_packet_end(b"1.000000,0.100000\n1.200000,N/A\n0.900000,0.100000\n") == 1.2
    )
    assert video_trim.last_packet_end(b"N/A,0.100000\n0.500000,0.100000\n") == pytest.approx(0.6)
    assert video_trim.last_packet_end(
        b"1.000000,0.100000,\n1.100000,0.100000,,\n"
    ) == pytest.approx(1.2)
    for refused in (b"", b"1.0\n", b"one,0.1\n", b"1.0,-0.1\n", b"1.0,0.1,x\n", b"1.0,\n"):
        with pytest.raises(video_trim.TrimRefused) as caught:
            video_trim.last_packet_end(refused)
        assert caught.value.code == "video-trim-unmeasured"


def test_sound_that_an_mp4_cannot_take_unchanged_is_not_offered() -> None:
    tool = MediaTool(name="ffprobe", executable=Path("ffprobe"), version="test", sha256="0" * 64)

    def probe(format_name: str, sound: str) -> Any:
        payload = {
            "format": {"format_name": format_name, "duration": "4.0", "start_time": "0.0"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 64,
                    "height": 48,
                },
                {"index": 1, "codec_type": "audio", "codec_name": sound},
            ],
        }
        artifact = Artifact(id="sha256:" + "1" * 64, sha256="1" * 64)
        return video_probe.describe(payload, artifact, tool)

    assert probe("mov,mp4,m4a,3gp,3g2,mj2", "vorbis").can_keep_audio is False
    assert probe("matroska,webm", "vorbis").can_keep_audio is True
    assert probe("mov,mp4,m4a,3gp,3g2,mj2", "aac").can_keep_audio is True


def test_a_trim_request_is_bounded_and_kept_apart_from_a_frame_request() -> None:
    fields = {
        "source_artifact_id": "sha256:" + "1" * 64,
        "source_sha256": "1" * 64,
        "requested_start_seconds": 1.0,
        "requested_end_seconds": 2.0,
        "keep_audio": True,
        "shown_start_seconds": 1.0,
    }

    assert video_utilities.TrimRequest(**fields).action == "trim"
    for change in (
        {"requested_start_seconds": -1.0},
        {"requested_end_seconds": float("inf")},
        {"shown_start_seconds": 10**9},
        {"action": "extract_frame"},
    ):
        with pytest.raises(ValidationError):
            video_utilities.TrimRequest(**{**fields, **change})
