"""Cutting a stored video on exact frames by re-encoding the chosen part.

Fixtures are tiny test patterns made by the real ffmpeg that hosted CI
installs. What a case expects comes from how its fixture was made and from the
case's own ffmpeg and ffprobe runs on the stored result, never from the job's
record of itself.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import os
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from pydantic import ValidationError
from sqlalchemy import select
from test_video_trim import (
    KEYED,
    PATTERN,
    TONE,
    _browser_fields,
    _change,
    _finished,
    _frames,
    _leftovers,
    _make,
    _path,
    _store,
    _streams,
    _tool,
    _utility_jobs,
    _videos,
    _walk,
)

from local_lm import video_probe, video_utilities
from local_lm.db import SessionLocal
from local_lm.domain import JobStatus
from local_lm.models import Artifact, ArtifactLibraryEntry, Job


def _exact() -> Any:
    """The exact cut's module, imported where a case uses it rather than at collection."""

    return importlib.import_module("local_lm.video_trim_exact")


def _trim() -> Any:
    return importlib.import_module("local_lm.video_trim")


async def _preview(client: AsyncClient, source: str, start: float, end: float, keep: bool) -> Any:
    return await client.get(
        f"/api/artifacts/{source}/video-trim-preview",
        params={
            "start_seconds": start,
            "end_seconds": end,
            "keep_audio": str(keep).lower(),
            "mode": "exact",
        },
    )


async def _post(
    client: AsyncClient,
    source: str,
    start: float,
    end: float,
    keep: bool,
    shown: float,
    count: int | None,
) -> Any:
    return await client.post(
        f"/api/artifacts/{source}/video-trims",
        json={
            "start_seconds": start,
            "end_seconds": end,
            "keep_audio": keep,
            "shown_start_seconds": shown,
            "mode": "exact",
            "shown_frame_count": count,
        },
    )


async def _cut(
    client: AsyncClient, source: str, start: float, end: float, keep: bool
) -> dict[str, Any]:
    """Check an exact cut, make it from the frames the check showed, and wait for the job."""

    checked = await _preview(client, source, start, end, keep)
    assert checked.status_code == 200, checked.text
    shown = checked.json()
    accepted = await _post(
        client, source, start, end, keep, shown["start_seconds"], shown["frame_count"]
    )
    assert accepted.status_code == 202, accepted.text
    return await _finished(client, accepted.json()["id"])


def _stage(
    app: FastAPI, source: str, start: float, end: float, keep: bool, shown: float, count: int
) -> str:
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
            mode="exact",
            shown_frame_count=count,
        )
        session.commit()
        return str(job.id)


def _lossless(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Re-encode losslessly, so the frames kept can be compared bit for bit with the source's."""

    module = _exact()
    run_tool = module.run_tool
    fired: list[bool] = []

    async def lossless(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        if "-crf" in arguments:
            at = arguments.index("-crf")
            arguments = [*arguments[:at], "-qp", "0", *arguments[at + 2 :]]
            fired.append(True)
        return await run_tool(executable, arguments, **bounds)

    monkeypatch.setattr(module, "run_tool", lossless)
    return fired


def _packets(path: Path) -> list[tuple[Fraction, str]]:
    """The picture stream's packets in the order shown: each time as a fraction, and its flags."""

    answer = subprocess.run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=time_base:packet=pts,flags",
            "-of",
            "compact=p=0:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    lines = answer.stdout.decode().split()
    base = next(line for line in lines if "/" in line)
    numerator, denominator = (int(part) for part in base.split("/"))
    rows = [line.split("|") for line in lines if "|" in line]
    return sorted((Fraction(int(pts) * numerator, denominator), flags) for pts, flags in rows)


def _psnr(made: Path, index: int, source: Path, at: float) -> float:
    """How like one frame of the new video is to the source's frame at a time."""

    answer = subprocess.run(
        [
            _tool("ffmpeg"),
            "-hide_banner",
            "-nostdin",
            "-i",
            str(made),
            "-ss",
            f"{at:.6f}",
            "-i",
            str(source),
            "-filter_complex",
            f"[0:v]select=eq(n\\,{index}),setpts=PTS-STARTPTS[a];"
            "[1:v]trim=end_frame=1,setpts=PTS-STARTPTS[b];[a][b]psnr",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    found = re.search(rb"average:(inf|[0-9.]+)", answer.stderr)
    assert found, answer.stderr[-500:]
    return float("inf") if found.group(1) == b"inf" else float(found.group(1))


async def test_an_exact_check_names_the_frames_and_queues_nothing(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _walk(tmp_path))

    checked = await _preview(client, source, 1.43, 2.7, True)

    assert checked.status_code == 200, checked.text
    preview = checked.json()
    # The frame on screen at 1.43 s began at 1.4 s; the last one before 2.7 s is 2.6 s.
    assert (preview["mode"], preview["start_seconds"], preview["last_frame_seconds"]) == (
        "exact",
        1.4,
        2.6,
    )
    assert (preview["end_seconds"], preview["frame_count"], preview["keyframe_seconds"]) == (
        2.7,
        13,
        1.0,
    )
    assert (preview["format"], preview["media_type"], preview["audio_streams_kept"]) == (
        "mp4",
        "video/mp4",
        1,
    )
    assert (preview["keeps_whole_video"], preview["began_at_first_picture"]) == (False, False)
    assert _utility_jobs() == []
    assert _leftovers(app) == []


async def test_an_exact_cut_holds_exactly_the_frames_shown_and_records_how_close_they_are(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _walk(tmp_path)
    source = _store(app, content)

    job = await _cut(client, source, 1.43, 2.7, True)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    assert job["phase"] == "Video trimmed"
    result = job["result_json"]
    made = _path(app, result["result_artifact_id"])
    [video, sound] = _streams(made)
    assert (video["codec_name"], sound["codec_name"]) == ("h264", "aac")
    # Thirteen frames, a tenth of a second apart from zero, as the source had them.
    packets = _packets(made)
    assert [time for time, _flags in packets] == [Fraction(index, 10) for index in range(13)]
    assert packets[0][1].startswith("K")
    original = tmp_path / "walk.mp4"
    # The first frame is the source's frame at 1.4 s, not its neighbours, and the last is 2.6 s.
    first = {at: _psnr(made, 0, original, at) for at in (1.3, 1.4, 1.5)}
    assert max(first, key=lambda at: first[at]) == 1.4, first
    last = {at: _psnr(made, 12, original, at) for at in (2.5, 2.6, 2.7)}
    assert max(last, key=lambda at: last[at]) == 2.6, last
    assert (result["mode"], result["actual_start_seconds"], result["keyframe_seconds"]) == (
        "exact",
        1.4,
        1.0,
    )
    assert result["actual_end_seconds"] == pytest.approx(2.7, abs=0.001)
    assert result["frames"]["count"] == 13
    assert (result["frames"]["first_seconds"], result["frames"]["last_seconds"]) == (1.4, 2.6)
    # Encoded from the keyframe at 0 s and checked from the part's own, at 1 s.
    assert (result["frames"]["decoded_from_seconds"], result["frames"]["checked_from_seconds"]) == (
        0.0,
        1.0,
    )
    assert result["quality"]["ssim_mean"] >= 0.95
    assert result["quality"]["psnr_lowest_db"] >= 35
    assert result["encoding"] == {
        "video_encoder": "libx264",
        "preset": "medium",
        "crf": 18,
        "pixel_format": "yuv420p",
        "audio_encoder": "aac",
        "audio_bitrates": [128000],
    }
    assert float(sound["duration"]) == pytest.approx(1.3, abs=0.03)
    assert set(result) == _browser_fields("VideoTrimResult")
    assert set(result["frames"]) == _browser_fields("ExactTrimFrames")
    assert set(result["quality"]) == _browser_fields("ExactTrimQuality")
    assert set(result["encoding"]) == _browser_fields("ExactTrimEncoding")
    assert result["in_library"] is True
    with SessionLocal() as session:
        trimmed = session.get(Artifact, result["result_artifact_id"])
        assert trimmed is not None
        assert (trimmed.media_type, trimmed.original_name) == (
            "video/mp4",
            "walk trim 1.400-2.700s.mp4",
        )
        entry = session.scalar(
            select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == trimmed.id)
        )
        assert entry is not None and entry.state == "visible"
    assert _path(app, source).read_bytes() == content
    assert _leftovers(app) == []


async def test_a_lossless_exact_cut_holds_the_source_frames_bit_for_bit(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    fired = _lossless(monkeypatch)

    job = await _cut(client, source, 1.43, 2.7, False)

    assert fired, "the encode was not made lossless"
    assert job["status"] == "complete", (job["result_json"], job["error"])
    made = _path(app, job["result_json"]["result_artifact_id"])
    assert _frames(made, count=14) == _frames(tmp_path / "walk.mp4", 1.4, 13)
    # Identical frames measure as identical, which PSNR states as infinite.
    assert job["result_json"]["quality"]["psnr_lowest_db"] is None
    assert job["result_json"]["quality"]["ssim_mean"] == 1.0


async def test_the_sound_is_cut_at_the_same_moments_as_the_picture(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    # A short burst of tone at the start of every second, and silence between.
    content = _make(
        tmp_path,
        "bursts.mp4",
        *PATTERN,
        "-f",
        "lavfi",
        "-i",
        "aevalsrc='if(lt(mod(t,1),0.02),sin(2*PI*1000*t),0)':s=48000",
        "-t",
        "4",
        *KEYED,
        "-c:a",
        "aac",
        "-shortest",
    )
    source = _store(app, content, "bursts.mp4")

    job = await _cut(client, source, 1.43, 2.7, True)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    made = _path(app, job["result_json"]["result_artifact_id"])
    answer = subprocess.run(
        [
            _tool("ffmpeg"),
            "-hide_banner",
            "-nostdin",
            "-i",
            str(made),
            "-af",
            "silencedetect=n=-40dB:d=0.1",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    # The burst at 2 s in the source is 0.6 s into a part that starts at 1.4 s.
    ends = [float(found) for found in re.findall(rb"silence_end: ([0-9.]+)", answer.stderr)]
    # Within an AAC frame of it; a frame of picture early or late would be 0.1 s off.
    assert ends and ends[0] == pytest.approx(0.6, abs=0.03), answer.stderr[-800:]


async def test_a_late_matroska_video_with_b_frames_is_cut_on_its_own_frames(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    _lossless(monkeypatch)

    checked = (await _preview(client, source, 2.43, 3.0, False)).json()
    assert (checked["start_seconds"], checked["frame_count"], checked["format"]) == (
        2.4,
        6,
        "mp4",
    )
    job = await _cut(client, source, 2.43, 3.0, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    frames = job["result_json"]["frames"]
    assert (frames["decoded_from_seconds"], frames["checked_from_seconds"]) == (1.0, 2.0)
    made = _path(app, job["result_json"]["result_artifact_id"])
    assert _frames(made, count=7) == _frames(tmp_path / "late.mkv", 2.4, 6)


async def test_a_part_that_starts_on_an_open_keyframe_is_cut_whole(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A copy refuses this start; decoding from the keyframe before it leaves nothing missing.
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
    _lossless(monkeypatch)

    job = await _cut(client, source, 2.43, 3.5, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    made = _path(app, job["result_json"]["result_artifact_id"])
    assert _frames(made, count=12) == _frames(tmp_path / "open.mkv", 2.4, 11)


@pytest.mark.parametrize(
    ("name", "encode"),
    [
        (
            "open.mp4",
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "keyint=10:min-keyint=10:scenecut=0:bframes=3:open-gop=1",
            ],
        ),
        # libx265's own defaults open every group of pictures.
        ("hevc.mp4", ["-c:v", "libx265", "-pix_fmt", "yuv420p", "-g", "10"]),
    ],
)
@pytest.mark.parametrize(
    ("start", "first", "decoded", "checked"), [(2.43, 2.4, 1.0, 2.0), (1.95, 1.9, 0.0, 1.0)]
)
async def test_an_mp4_of_open_keyframes_is_cut_on_its_own_frames(
    client: AsyncClient,
    app: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    encode: list[str],
    start: float,
    first: float,
    decoded: float,
    checked: float,
) -> None:
    # An MP4 is sought by decode time, and an open keyframe is stored before
    # pictures shown ahead of it: 1.9 s is shown before, and stored after, 2 s.
    content = _make(tmp_path, name, *PATTERN, "-t", "4", *encode)
    source = _store(app, content, name)
    _lossless(monkeypatch)

    job = await _cut(client, source, start, 3.0, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    frames = job["result_json"]["frames"]
    assert (
        frames["first_seconds"],
        frames["decoded_from_seconds"],
        frames["checked_from_seconds"],
    ) == (
        first,
        decoded,
        checked,
    )
    count = round((3.0 - first) * 10)
    made = _path(app, job["result_json"]["result_artifact_id"])
    # The source decoded from its very beginning: a seek into it would meet the same keyframes.
    shown = _frames(tmp_path / name, count=40)
    assert _frames(made, count=count + 1) == shown[round(first * 10) : round(first * 10) + count]


async def test_a_stream_copy_that_began_between_keyframes_is_cut_from_its_first_picture(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Copied from 1.5 s without re-encoding: its edit list hides the keyframe
    # its first pictures are decoded from. A fixed pattern of three B-frames
    # stores those pictures where a seek to the first keyframe shown passes them.
    _make(
        tmp_path,
        "fixed.mp4",
        *PATTERN,
        "-t",
        "4",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-x264-params",
        "keyint=10:min-keyint=10:scenecut=0:bframes=3:b-adapt=0",
    )
    content = _make(
        tmp_path,
        "copied.mp4",
        "-ss",
        "1.5",
        "-i",
        str(tmp_path / "fixed.mp4"),
        "-map",
        "0:v",
        "-c",
        "copy",
    )
    source = _store(app, content, "copied.mp4")
    _lossless(monkeypatch)

    checked = (await _preview(client, source, 0.0, 1.0, False)).json()
    assert (checked.get("start_seconds"), checked.get("frame_count")) == (0.0, 10), checked
    job = await _cut(client, source, 0.0, 1.0, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    made = _path(app, job["result_json"]["result_artifact_id"])
    assert _frames(made, count=11) == _frames(tmp_path / "copied.mp4", count=10)


async def test_a_variable_frame_rate_mp4_is_checked_from_its_own_keyframe(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Frames 0.2 s apart, then 0.08 s apart; keyframes at 0, 1.04 and 1.84 s.
    content = _make(
        tmp_path,
        "uneven.mp4",
        *PATTERN,
        "-t",
        "4",
        "-vf",
        "settb=1/16000,setpts='if(lt(N,2),N*0.2,0.4+(N-2)*0.08)/TB'",
        "-fps_mode",
        "passthrough",
        "-enc_time_base",
        "1:16000",
        *KEYED,
    )
    source = _store(app, content, "uneven.mp4")
    _lossless(monkeypatch)

    job = await _cut(client, source, 1.43, 1.8, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    frames = job["result_json"]["frames"]
    assert frames["checked_from_seconds"] == pytest.approx(1.04, abs=0.001), frames
    assert frames["decoded_from_seconds"] < frames["checked_from_seconds"]
    made = _path(app, job["result_json"]["result_artifact_id"])
    assert _frames(made, count=frames["count"] + 1) == _frames(
        tmp_path / "uneven.mp4", frames["first_seconds"], frames["count"]
    )


@pytest.mark.parametrize(
    ("start", "end", "first", "count"),
    [
        # On a keyframe exactly, within the first group, and to the very end.
        (2.0, 2.5, 2.0, 5),
        (0.25, 0.8, 0.2, 6),
        (3.05, 4.0, 3.0, 10),
    ],
)
async def test_exact_cuts_begin_and_end_on_the_frames_at_their_times(
    client: AsyncClient,
    app: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start: float,
    end: float,
    first: float,
    count: int,
) -> None:
    source = _store(app, _walk(tmp_path))
    _lossless(monkeypatch)

    job = await _cut(client, source, start, end, False)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    assert (
        job["result_json"]["frames"]["first_seconds"],
        job["result_json"]["frames"]["count"],
    ) == (
        first,
        count,
    )
    made = _path(app, job["result_json"]["result_artifact_id"])
    assert _frames(made, count=count + 1) == _frames(tmp_path / "walk.mp4", first, count)


@pytest.mark.parametrize(
    ("name", "media_type", "codec"),
    [("late-picture.mp4", "video/mp4", "aac"), ("late-picture.mkv", "video/x-matroska", "flac")],
)
async def test_a_start_before_the_first_picture_starts_on_it_and_leaves_earlier_sound_out(
    client: AsyncClient, app: FastAPI, tmp_path: Path, name: str, media_type: str, codec: str
) -> None:
    content = _make(
        tmp_path,
        name,
        *TONE,
        "-itsoffset",
        "0.5",
        *PATTERN,
        "-t",
        "3",
        "-map",
        "0:a",
        "-map",
        "1:v",
        *KEYED,
        "-c:a",
        codec,
    )
    # The case needs a picture that starts half a second after the sound. Some
    # ffmpeg builds (6.1 among them) write such an MP4 with both starting at
    # once; Matroska keeps the late start with every build, and its FLAC sound
    # has no encoder delay to start it early.
    starts = [float(stream["start_time"]) for stream in _streams(tmp_path / name)]
    if media_type == "video/mp4" and starts[1] - starts[0] < 0.25:
        pytest.skip("this ffmpeg writes the MP4 with its picture starting with its sound")
    assert starts[1] - starts[0] == pytest.approx(0.5, abs=0.05), starts
    source = _store(app, content, name, media_type)
    probe = (await client.get(f"/api/artifacts/{source}/video-probe")).json()
    assert probe["can_trim_exact"], probe

    checked = (await _preview(client, source, 0.0, 1.5, True)).json()

    assert checked["began_at_first_picture"] is True, (
        starts,
        probe["start_seconds"],
        {key: checked[key] for key in ("start_seconds", "keyframe_seconds", "frame_count")},
    )
    assert checked["start_seconds"] == pytest.approx(0.5, abs=0.001)
    job = await _cut(client, source, 0.0, 1.5, True)
    assert job["status"] == "complete", (job["result_json"], job["error"])
    assert job["result_json"]["frames"]["began_at_first_picture"] is True
    [video, sound] = _streams(_path(app, job["result_json"]["result_artifact_id"]))
    assert float(sound["start_time"]) >= float(video["start_time"]) - 0.001


@pytest.mark.parametrize(
    ("kind", "limit"),
    [
        ("full-colour", "exact-trim-picture-unsupported"),
        ("full-range", "exact-trim-picture-unsupported"),
        ("turned", "exact-trim-turned-unsupported"),
    ],
)
async def test_pictures_an_exact_cut_would_change_are_refused_before_queueing(
    client: AsyncClient, app: FastAPI, tmp_path: Path, kind: str, limit: str
) -> None:
    encode = ["-c:v", "libx264", "-g", "10"]
    if kind == "turned":
        _make(tmp_path, "upright.mp4", *PATTERN, "-t", "2", *encode, "-pix_fmt", "yuv420p")
        upright = str(tmp_path / "upright.mp4")
        content = _make(
            tmp_path, "clip.mp4", "-display_rotation", "90", "-i", upright, "-c", "copy"
        )
    else:
        picture = (
            ["-pix_fmt", "yuv444p"]
            if kind == "full-colour"
            else ["-pix_fmt", "yuv420p", "-color_range", "pc"]
        )
        content = _make(tmp_path, "clip.mp4", *PATTERN, "-t", "2", *encode, *picture)
    source = _store(app, content, "clip.mp4")

    probe = (await client.get(f"/api/artifacts/{source}/video-probe")).json()
    refused = await _preview(client, source, 0.5, 1.5, False)

    assert (probe["can_trim"], probe["can_trim_exact"]) == (True, False)
    assert limit in probe["exact_trim_limits"]
    assert (refused.status_code, refused.json()["code"]) == (422, "video-trim-exact-not-offered")
    assert _utility_jobs() == []
    assert _leftovers(app) == []


def test_pictures_an_exact_cut_would_change_are_named_from_their_stated_facts() -> None:
    facts = video_probe.VideoStreamFacts(
        index=0,
        codec="h264",
        width=65,
        height=49,
        display_width=65,
        display_height=49,
        sample_aspect=None,
        rotation=0,
        frame_rate="10/1",
        frame_rate_form="constant",
        time_base="1/10240",
        pixel_format="yuv420p",
        color_range=None,
        color_space=None,
        color_transfer=None,
        color_primaries=None,
        field_order="progressive",
    )
    assert video_probe.exact_trim_limits(facts, []) == ["exact-trim-frame-size-unsupported"]
    large = facts.model_copy(update={"width": 4098, "height": 1080})
    assert video_probe.exact_trim_limits(large, []) == ["exact-trim-frame-size-unsupported"]
    upright = facts.model_copy(update={"width": 64, "height": 48})
    for picture in (
        {"field_order": "unreadable"},
        {"field_order": "tt"},
        {"color_transfer": "smpte2084"},
        {"color_transfer": "arib-std-b67"},
        {"color_primaries": "bt2020"},
        {"color_range": "pc"},
        {"pixel_format": "yuv420p10le"},
        {"pixel_format": None},
    ):
        assert video_probe.exact_trim_limits(upright.model_copy(update=picture), []) == [
            "exact-trim-picture-unsupported"
        ], picture
    tagged = upright.model_copy(
        update={"color_range": "tv", "color_space": "bt709", "color_transfer": "bt709"}
    )
    assert video_probe.exact_trim_limits(tagged, []) == []
    # Two transfers ffmpeg's encoder options name differently from how streams state them.
    colours = importlib.import_module("local_lm.video_trim_exact")._colours
    pal = tagged.model_copy(update={"color_transfer": "bt470bg", "color_primaries": "bt470bg"})
    assert colours(pal) == [
        "-color_primaries",
        "bt470bg",
        "-color_trc",
        "gamma28",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
    ]
    assert "gamma22" in colours(tagged.model_copy(update={"color_transfer": "bt470m"}))
    stereo = video_probe.AudioStreamFacts(index=1, codec="pcm_s16le", channels=2, sample_rate=48000)
    surround = video_probe.AudioStreamFacts(index=2, codec="aac", channels=6, sample_rate=48000)
    assert video_probe.exact_trim_limits(upright, [stereo]) == []
    assert video_probe.exact_trim_limits(upright, [stereo, surround]) == [
        "exact-trim-audio-unsupported"
    ]


async def test_sound_an_exact_cut_cannot_re_encode_can_only_be_left_out(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(
        tmp_path,
        "surround.mp4",
        *PATTERN,
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000",
        "-t",
        "3",
        *KEYED,
        "-c:a",
        "aac",
        "-ac",
        "6",
        "-shortest",
    )
    source = _store(app, content, "surround.mp4")

    probe = (await client.get(f"/api/artifacts/{source}/video-probe")).json()
    kept = await _preview(client, source, 1.0, 2.0, True)
    silent = await _preview(client, source, 1.0, 2.0, False)

    assert (probe["can_trim_exact"], probe["can_keep_audio_exact"]) == (True, False)
    assert probe["exact_trim_limits"] == ["exact-trim-audio-unsupported"]
    assert (kept.status_code, kept.json()["code"]) == (422, "video-trim-exact-audio-not-offered")
    assert silent.status_code == 200, silent.text


async def test_pcm_sound_a_copy_cannot_keep_is_re_encoded_by_an_exact_cut(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    content = _make(
        tmp_path,
        "pcm.mkv",
        *PATTERN,
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000",
        "-t",
        "3",
        *KEYED,
        "-c:a",
        "pcm_s16le",
        "-shortest",
    )
    source = _store(app, content, "pcm.mkv", "video/x-matroska")

    job = await _cut(client, source, 1.0, 2.0, True)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    [_video, sound] = _streams(_path(app, job["result_json"]["result_artifact_id"]))
    assert sound["codec_name"] == "aac"


async def test_ranges_an_exact_cut_cannot_take_are_refused_before_queueing(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))

    probe = (await client.get(f"/api/artifacts/{source}/video-probe")).json()
    length = probe["duration_seconds"]
    whole = (await _preview(client, source, 0.0, length, False)).json()
    assert (whole["keeps_whole_video"], whole["frame_count"]) == (True, 40), whole
    refused = await _post(client, source, 0.0, length, False, 0.0, whole["frame_count"])
    assert (refused.status_code, refused.json()["code"]) == (422, "video-trim-keeps-whole-video")
    assert "keep the original quality instead" in refused.json()["detail"]
    silent = _store(app, _make(tmp_path, "plain.mp4", *PATTERN, "-t", "4", *KEYED), "plain.mp4")
    quiet = (await client.get(f"/api/artifacts/{silent}/video-probe")).json()["duration_seconds"]
    still = (await _preview(client, silent, 0.0, quiet, False)).json()
    refused_silent = await _post(client, silent, 0.0, quiet, False, 0.0, still["frame_count"])
    assert (refused_silent.status_code, refused_silent.json()["code"]) == (
        422,
        "video-trim-keeps-whole-video",
    )
    assert "sound" not in refused_silent.json()["detail"]
    short = await _preview(client, source, 1.0, 1.05, False)
    assert (short.status_code, short.json()["code"]) == (422, "video-trim-range-invalid")
    monkeypatch.setattr(_exact(), "MAX_EXACT_FRAMES", 5)
    long = await _preview(client, source, 1.0, 2.0, False)
    assert (long.status_code, long.json()["code"]) == (422, "video-trim-exact-too-long")
    monkeypatch.setattr(_exact(), "MAX_EXACT_FRAMES", 7200)
    monkeypatch.setattr(_exact(), "MAX_LEAD_SECONDS", 0.5)
    late = await _preview(client, source, 1.83, 2.0, False)
    assert (late.status_code, late.json()["code"]) == (422, "video-trim-exact-too-long")
    assert _utility_jobs() == []
    assert _leftovers(app) == []


async def test_an_ffmpeg_that_cannot_encode_h264_cannot_cut_on_exact_frames(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    module = _exact()
    run_tool = module.run_tool
    asked: list[str] = []

    async def without_h264(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        if "-h" in arguments:
            asked.append(arguments[-1])
            answer = await run_tool(executable, arguments, **bounds)
            return type(answer)(
                returncode=0, stdout=b"Codec 'libx264' is not recognized by FFmpeg.\n", stderr=b""
            )
        return await run_tool(executable, arguments, **bounds)

    monkeypatch.setattr(module, "run_tool", without_h264)

    refused = await _preview(client, source, 1.0, 2.0, False)
    assert (refused.status_code, refused.json()["code"]) == (503, "video-trim-encoder-missing")
    assert asked == ["encoder=libx264"]
    job_id = _stage(app, source, 1.0, 2.0, False, 1.0, 10)
    app.state.services.video_utilities.start(job_id)
    failed = await _finished(client, job_id)
    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "video-trim-encoder-missing"},
    )
    assert _leftovers(app) == []


async def test_an_exact_cut_is_bound_to_the_start_and_the_frames_that_were_shown(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _walk(tmp_path))

    moved = await _post(client, source, 1.43, 2.7, False, 1.43, 13)
    assert (moved.status_code, moved.json()["code"]) == (409, "video-trim-preview-stale")
    fewer = await _post(client, source, 1.43, 2.7, False, 1.4, 12)
    assert (fewer.status_code, fewer.json()["code"]) == (409, "video-trim-preview-stale")
    unsaid = await _post(client, source, 1.43, 2.7, False, 1.4, None)
    assert unsaid.status_code == 422, unsaid.text
    assert _utility_jobs() == []

    job_id = _stage(app, source, 1.43, 2.7, False, 1.4, 12)
    app.state.services.video_utilities.start(job_id)
    failed = await _finished(client, job_id)
    assert (failed["status"], failed["result_json"]) == (
        "failed",
        {"failure_code": "video-trim-frames-moved"},
    )
    with SessionLocal() as session:
        stored = session.get(Job, job_id)
        assert stored is not None
        stored.payload_json = {**stored.payload_json, "shown_frame_count": 13}
        session.commit()
    retried = await client.post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 200, retried.text
    assert (await _finished(client, job_id))["status"] == "complete"


@pytest.mark.parametrize(
    ("fault", "code"),
    [
        ("later-frames", "video-trim-frames-unverified"),
        ("one-poor-frame", "video-trim-reencode-unlike"),
        ("low-ssim", "video-trim-reencode-unlike"),
        ("same-start", "video-trim-start-unchecked"),
        ("every-other-frame", "video-trim-frames-unverified"),
        ("poor-quality", "video-trim-reencode-unlike"),
        ("no-sound", "video-trim-output-mismatch"),
        ("wider-colour", "video-trim-output-mismatch"),
        ("no-room-to-finish", "video-trim-not-encoded"),
        ("other-reference", "video-trim-start-incomplete"),
        ("no-ssim", "video-trim-unmeasured"),
    ],
)
async def test_a_re_encode_that_is_not_what_was_planned_is_not_kept(
    client: AsyncClient,
    app: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    code: str,
) -> None:
    source = _store(app, _walk(tmp_path))
    module = _exact()
    run_tool = module.run_tool
    fired: list[str] = []

    def replaced(arguments: list[str], flag: str, value: str) -> list[str]:
        at = arguments.index(flag) + 1
        return [*arguments[:at], value, *arguments[at + 1 :]]

    async def faulty(executable: Path, arguments: list[str], **bounds: Any) -> Any:
        encoding = "-crf" in arguments
        comparing = "-loglevel" in arguments and "info" in arguments
        graph = arguments[arguments.index("-filter_complex") + 1] if encoding else ""
        if encoding and fault == "later-frames":
            # One frame later at both ends: as many frames, as spaced, but not the planned ones.
            changed = graph.replace(
                "trim=start_pts=14336:end_pts=26625", "trim=start_pts=15360:end_pts=27649"
            )
            assert changed != graph
            arguments = replaced(arguments, "-filter_complex", changed)
            fired.append(fault)
        elif encoding and fault == "one-poor-frame":
            # One frame spoiled before it is encoded; the digest of what was decoded is not.
            changed = graph.replace(
                "[e]setpts=PTS-STARTPTS[v]",
                "[e]setpts=PTS-STARTPTS,"
                "drawbox=x=0:y=0:w=24:h=16:color=black:t=fill:enable='eq(n\\,6)'[v]",
            )
            assert changed != graph
            arguments = replaced(arguments, "-filter_complex", changed)
            fired.append(fault)
        elif encoding and fault == "every-other-frame":
            changed = graph.replace(
                "[e]setpts=PTS-STARTPTS[v]", "[e]setpts=PTS-STARTPTS,select=not(mod(n\\,2))[v]"
            )
            assert changed != graph
            arguments = replaced(arguments, "-filter_complex", changed)
            fired.append(fault)
        elif encoding and fault == "poor-quality":
            arguments = replaced(arguments, "-crf", "51")
            fired.append(fault)
        elif encoding and fault == "no-sound":
            # The sound the check promised is neither cut nor kept.
            at = arguments.index("[a0]") - 1
            arguments = replaced(
                arguments[:at] + arguments[at + 2 :], "-filter_complex", graph.split(";[0:1]")[0]
            )
            fired.append(fault)
        elif encoding and fault == "wider-colour":
            arguments = replaced(arguments, "-pix_fmt", "yuv444p")
            fired.append(fault)
        elif encoding and fault == "no-room-to-finish":
            # ffmpeg stops at the size bound and still exits cleanly.
            assert arguments[arguments.index("-fs") + 1] == "1500"
            fired.append(fault)
        elif comparing and fault == "other-reference":
            graph = arguments[arguments.index("-filter_complex") + 1]
            arguments = replaced(
                arguments,
                "-filter_complex",
                graph.replace(",split[h][r]", ",negate,split[h][r]", 1),
            )
            fired.append(fault)
        elif comparing and fault == "low-ssim":
            answer = await run_tool(executable, arguments, **bounds)
            fired.append(fault)
            text = re.sub(rb"All:\d\.\d+ ", b"All:0.900000 ", answer.stderr)
            assert text != answer.stderr
            return type(answer)(returncode=answer.returncode, stdout=answer.stdout, stderr=text)
        elif comparing and fault == "same-start":
            # The check decoded from where the encode's decode began.
            arguments = replaced(arguments, "-ss", "0.000000")
            fired.append(fault)
        elif comparing and fault == "no-ssim":
            answer = await run_tool(executable, arguments, **bounds)
            fired.append(fault)
            text = re.sub(rb"\[ssim@aligned[^\n]*\n", b"", answer.stderr)
            return type(answer)(returncode=answer.returncode, stdout=answer.stdout, stderr=text)
        return await run_tool(executable, arguments, **bounds)

    monkeypatch.setattr(module, "run_tool", faulty)
    if fault == "no-room-to-finish":
        monkeypatch.setattr(module, "MAX_INPUT_BYTES", 1500)
        monkeypatch.setattr(module, "CUT_ROOM", 0)
    keep = fault == "no-sound"
    job_id = _stage(app, source, 1.43, 2.7, keep, 1.4, 13)
    app.state.services.video_utilities.start(job_id)

    failed = await _finished(client, job_id)
    assert fired == [fault]
    assert (failed["status"], failed["result_json"]) == ("failed", {"failure_code": code})
    assert _videos() == [source]
    assert _leftovers(app) == []


async def test_an_exact_cut_reads_one_private_copy_and_writes_one_file_the_store_made(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    stored = _path(app, source)
    calls: list[list[str]] = []
    modules = [video_probe, _trim(), _exact()]
    originals = [module.run_tool for module in modules]

    for module, run_tool in zip(modules, originals, strict=True):

        async def recording(
            executable: Path, arguments: list[str], _run: Any = run_tool, **bounds: Any
        ) -> Any:
            calls.append(list(arguments))
            return await _run(executable, arguments, **bounds)

        monkeypatch.setattr(module, "run_tool", recording)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.4, 13)
    app.state.services.video_utilities.start(job_id)

    assert (await _finished(client, job_id))["status"] == "complete"
    files = {
        Path(item.removeprefix("file:")).name for call in calls for item in call if "file:" in item
    }
    assert len(files) == 2
    assert all(re.fullmatch(r"tool-(input|output)-[0-9a-f]{32}\.tmp", name) for name in files), (
        files
    )
    assert all(Path(item.removeprefix("file:")) != stored for call in calls for item in call)
    for call in calls:
        inputs = [index for index, item in enumerate(call) if item == "-i"]
        for number, index in enumerate(inputs):
            since = call[(inputs[number - 1] if number else 0) : index]
            assert since[since.index("-protocol_whitelist") + 1] == "file"
            assert since[since.index("-format_whitelist") + 1] == video_probe.DEMUXERS
    [encode] = [call for call in calls if "-crf" in call]
    for flag, value in (
        ("-c:v", "libx264"),
        ("-preset", "medium"),
        ("-crf", "18"),
        ("-pix_fmt", "yuv420p"),
        ("-c:a:0", "aac"),
        ("-fps_mode:v", "passthrough"),
        ("-enc_time_base:v", "1:10240"),
        ("-map_metadata", "-1"),
        ("-f", "mp4"),
    ):
        assert encode[encode.index(flag) + 1] == value, flag
    assert {"-nostdin", "-noautorotate", "-copyts", "-fs", "-y"} <= set(encode)
    assert "-t" not in encode
    assert encode[-1] == "pipe:1"
    assert [encode[index + 1] for index, item in enumerate(encode) if item == "-map"][-2:] == [
        "[h]",
        "[k]",
    ]
    assert _leftovers(app) == []


async def test_a_cancelled_exact_cut_keeps_no_video_and_no_working_files(
    client: AsyncClient, app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _store(app, _walk(tmp_path))
    entered = asyncio.Event()
    encode = _exact().encode_part

    async def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await asyncio.sleep(60)
        return await encode(*args, **kwargs)

    monkeypatch.setattr(video_utilities, "encode_part", held)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.4, 13)
    app.state.services.video_utilities.start(job_id)
    await asyncio.wait_for(entered.wait(), timeout=120)
    assert len(_leftovers(app)) == 2

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200, cancelled.text
    job = await _finished(client, job_id)
    assert (job["status"], job["result_json"]) == ("cancelled", {})
    assert _videos() == [source]
    assert _leftovers(app) == []


@pytest.mark.parametrize("change", ["overwritten", "replaced", "rewritten as it was"])
async def test_an_exact_cut_cannot_change_between_its_checks_and_being_kept(
    client: AsyncClient,
    app: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    source = _store(app, _walk(tmp_path))
    compare = _exact().compare_with_source
    changed: list[bool] = []

    async def then_changed(copy: Path, made: Path, *arguments: Any) -> Any:
        # The checks and the comparison pass on the video that was made; then it is changed.
        compared = await compare(copy, made, *arguments)
        changed.append(_change(made, change, tmp_path))
        return compared

    monkeypatch.setattr(video_utilities, "compare_with_source", then_changed)
    job_id = _stage(app, source, 1.43, 2.7, True, 1.4, 13)
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


async def test_an_interrupted_exact_cut_is_started_again_from_its_request(
    client: AsyncClient, app: FastAPI, tmp_path: Path
) -> None:
    source = _store(app, _walk(tmp_path))
    job_id = _stage(app, source, 1.43, 2.7, False, 1.4, 13)
    with SessionLocal() as session:
        stored = session.get(Job, job_id)
        assert stored is not None
        stored.status = JobStatus.INTERRUPTED.value
        session.commit()

    app.state.services.video_utilities.recover()

    finished = await _finished(client, job_id)
    assert finished["status"] == "complete", finished
    assert (finished["result_json"]["mode"], finished["result_json"]["frames"]["count"]) == (
        "exact",
        13,
    )


def test_a_trim_request_names_its_mode_and_only_an_exact_one_its_frames() -> None:
    request = video_utilities.TrimRequest
    common = {
        "source_artifact_id": "sha256:" + "a" * 64,
        "source_sha256": "a" * 64,
        "requested_start_seconds": 1.0,
        "requested_end_seconds": 2.0,
        "keep_audio": False,
        "shown_start_seconds": 1.0,
    }
    # A request stored before cuts had a mode is a copy.
    assert request.model_validate(common).mode == "copy"
    assert request.model_validate({**common, "mode": "exact", "shown_frame_count": 10}).mode == (
        "exact"
    )
    for refused in (
        {**common, "mode": "smart"},
        {**common, "mode": "exact"},
        {**common, "mode": "copy", "shown_frame_count": 10},
        {**common, "mode": "exact", "shown_frame_count": 0},
    ):
        with pytest.raises(ValidationError):
            request.model_validate(refused)


def test_frame_listings_are_read_strictly_and_choose_the_frames_on_screen() -> None:
    module = _exact()
    refused_code = "video-trim-frames-unknown"
    rows = module.frame_rows(b"10240,1024,K__\n13312,1024,___\n11264,1024,___\n12288,1024,___\n")
    assert [row.ticks for row in rows] == [10240, 11264, 12288, 13312]
    # Discarded packets are not shown; a corrupt one, a duplicate or an unreadable row refuses.
    assert [row.ticks for row in module.frame_rows(b"9216,1024,_D_\n10240,1024,K__\n")] == [10240]
    # Packets with side data, such as a transparent picture's, list one more field, empty.
    with_side_data = module.frame_rows(b"10240,1024,K__,\r\n11264,1024,___,\r\n")
    assert [row.ticks for row in with_side_data] == [10240, 11264]
    for unreadable in (
        b"",
        b"N/A,1024,K__\n",
        b"10240,1024,K_C\n",
        b"10240,1024,K__\n10240,1024,___\n",
        b"10240;1024;K__\n",
        b"10240,1024,K__,x\n",
    ):
        with pytest.raises(module.TrimRefused) as caught:
            module.frame_rows(unreadable)
        assert caught.value.code == refused_code
    base = Fraction(1, 10240)
    chosen = module.choose_frames(rows, base, 1.1003, 1.25)
    assert (chosen.frames, chosen.next_ticks, chosen.end_ticks) == ((11264, 12288), 13312, 13312)
    # Half a millisecond either side of a frame's time counts as at it.
    assert module.choose_frames(rows, base, 1.0996, 1.3004).frames == (11264, 12288)
    assert module.choose_frames(rows, base, 0.5, 1.15).began_at_first_picture is True
    end = module.choose_frames(rows, base, 1.25, 2.0)
    assert (end.frames, end.next_ticks, end.end_ticks) == ((12288, 13312), None, 14336)
    with pytest.raises(module.TrimRefused) as before:
        module.choose_frames(rows, base, 0.2, 0.9)
    assert before.value.code == "video-trim-range-before-picture"
    close = module.frame_rows(b"10240,1024,K__\n10245,1024,___\n")
    with pytest.raises(module.TrimRefused):
        module.choose_frames(close, base, 1.0, 2.0)
    open_ended = module.frame_rows(b"10240,N/A,K__\n")
    with pytest.raises(module.TrimRefused):
        module.choose_frames(open_ended, base, 1.0, 2.0)


def test_frame_digests_and_comparison_reports_are_read_strictly() -> None:
    module = _exact()
    digest = "ab" * 32
    header = b"#format: frame checksums\n#tb 0: 1/10240\n"
    listed = header + f"0,      10240,      10240,     1024,     4608, {digest}\n".encode()
    assert module.frame_digests(listed, Fraction(1, 10240)) == module.Decoded(
        [module.FedFrame(10240, digest)], None
    )
    both = listed.replace(b"#tb 0: 1/10240\n", b"#tb 0: 1/10240\n#tb 1: 1/10240\n") + (
        f"1,          0,          0,     1024,     4608, {digest}\n".encode()
    )
    assert module.frame_digests(both, Fraction(1, 10240)).began == 0
    for unreadable, base in (
        (listed, Fraction(1, 1000)),
        (listed.replace(b"#tb 0: 1/10240\n", b""), Fraction(1, 10240)),
        (header + b"0, 1, 1, 1, 9, not-a-digest\n", Fraction(1, 10240)),
        (both.replace(b"#tb 1: 1/10240", b"#tb 1: 1/1000"), Fraction(1, 10240)),
        (both + f"1, 1024, 1024, 1024, 4608, {digest}\n".encode(), Fraction(1, 10240)),
    ):
        with pytest.raises(module.TrimRefused) as caught:
            module.frame_digests(unreadable, base)
        assert caught.value.code == "video-trim-frames-unverified"
    psnr = (
        b"[psnr@aligned @ 0x55d5c2a3b4c0] PSNR y:48.279814 u:47.883921 v:46.973739 "
        b"average:47.968226 min:46.790483 max:49.034890\n"
    )
    ssim = (
        b"[ssim@aligned @ 0000022b1e42100] SSIM Y:0.998709 (28.891677) U:0.998811 (29.248596) "
        b"V:0.998756 (29.052092) All:0.998734 (28.975843)\n"
    )
    stated = module.closeness_stated(psnr + ssim)
    assert (stated.ssim_mean, stated.psnr_mean_db, stated.psnr_lowest_db) == (
        0.998734,
        47.968226,
        46.790483,
    )
    identical = module.closeness_stated(
        b"[psnr@aligned @ 0x1] PSNR y:inf u:inf v:inf average:inf min:inf max:inf\n"
        b"[ssim@aligned @ 0x2] SSIM Y:1.000000 (inf) U:1.000000 (inf) V:1.000000 (inf) "
        b"All:1.000000 (inf)\n"
    )
    assert (identical.psnr_mean_db, identical.psnr_lowest_db, identical.ssim_mean) == (
        None,
        None,
        1.0,
    )
    for unreadable in (psnr, ssim, psnr + psnr + ssim, b""):
        with pytest.raises(module.TrimRefused) as caught:
            module.closeness_stated(unreadable)
        assert caught.value.code == "video-trim-unmeasured"
    assert module.encoder_offered(b"Encoder libx264 [libx264 H.264 / AVC]:\n", "libx264")
    assert not module.encoder_offered(b"Codec 'libx264' is not recognized by FFmpeg.\n", "libx264")
    assert not module.encoder_offered(b"Encoder aac [AAC]:\n", "libx264")
    assert not module.encoder_offered(b"", "libx264")


@pytest.mark.parametrize("length", [None, 0])
async def test_a_new_video_whose_last_frame_states_no_length_ends_where_the_next_frame_began(
    client: AsyncClient,
    app: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    length: int | None,
) -> None:
    # As some ffmpeg builds write the re-encoded file: no stored length for its last frame.
    module = _exact()
    listed = module._output_frames

    async def unstated(output: Path, ffprobe: Any) -> Any:
        rows = await listed(output, ffprobe)
        return [*rows[:-1], dataclasses.replace(rows[-1], duration=length)]

    monkeypatch.setattr(module, "_output_frames", unstated)
    source = _store(app, _walk(tmp_path))

    job = await _cut(client, source, 1.43, 2.7, True)

    assert job["status"] == "complete", (job["result_json"], job["error"])
    result = job["result_json"]
    assert (result["actual_start_seconds"], result["frames"]["count"]) == (1.4, 13)
    assert result["actual_end_seconds"] == pytest.approx(2.7, abs=0.001)
