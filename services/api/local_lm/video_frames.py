"""One frame of a stored video, decoded within fixed bounds, with the time it really has.

A request names a time, but a video has frames only at its own timestamps. The
frame saved is the one shown at that time, as a paused player shows it: the
last frame that begins at or before it, so a time in a frame's span, at the
very end, or after the picture ends while the sound runs on, saves the frame on
screen, and a time before the first picture saves that picture. Its own
timestamp is recorded as the actual time beside the requested one; the two
differ whenever the request falls between frames, and the record says so
rather than calling the frame exact.

FINDING IT. ffmpeg first decodes, without saving anything, from where a seek to
the time lands up to the time itself, and reports every frame it passes. An
MP4 is sought by decode time, so a seek can land on a keyframe stored before
the pictures shown ahead of it; when nothing it passes is at or before the
time, the seek starts earlier, twice as far back each time, within a fixed
reach. The frame chosen is then decoded on its own from the same seek point.
Frames are retimed to whole microseconds before they are reported, so the time
recorded is the frame's own integer timestamp, never a printed time some
builds round to six significant digits.

The frame is kept as a lossless PNG with the stream's rotation applied, as a
player shows it. A non-square pixel shape is recorded, not applied, so the PNG
holds the pixels the stream holds. As with the probe, ffmpeg reads a private
copy of exactly the verified bytes, through the same demuxer and protocol
limits, and both of its outputs and its time are bounded.
"""

from __future__ import annotations

import asyncio
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from PIL import Image, UnidentifiedImageError

from .media_process import ToolOutputTooLarge, ToolResult, run_tool
from .media_tools import MediaTool, MediaToolUnavailable, ToolUnavailableCode
from .models import Artifact
from .video_probe import DEMUXERS, MAX_DIMENSION, VideoProbe

DECODE_SECONDS: Final = 60.0
#: A lossless frame at the largest size the probe offers fits well within this.
MAX_FRAME_BYTES: Final = 256 * 1024 * 1024
#: ffmpeg's frame report and its description of the input; never shown to anyone.
MAX_REPORT_BYTES: Final = 1024 * 1024
#: The report of every frame passed on the way to the time: a few hundred bytes each.
MAX_PASSED_BYTES: Final = 16 * 1024 * 1024
#: How far before the time the first seek starts, and the furthest any seek reaches back.
FIRST_REACH_SECONDS: Final = 1.0
MAX_REACH_SECONDS: Final = 64.0
#: A frame that begins within half a millisecond after the time counts as shown at it.
SHOWN_WITHIN: Final = 0.0005
#: A frame's report from the showinfo filter after ``settb=1/1000000``: its
#: timestamp in whole microseconds.
FRAME_REPORT: Final = re.compile(rb"\bn:\s*\d{1,9}\s+pts:\s*(-?\d{1,18})\s+pts_time:")

FrameRefusalCode = Literal[
    "video-frame-not-offered",
    "video-frame-time-outside",
    "video-frame-unreadable",
    "video-frame-timed-out",
]

_MESSAGES: Final[dict[FrameRefusalCode, str]] = {
    "video-frame-not-offered": "A frame cannot be saved from this video.",
    "video-frame-time-outside": "That time is not within this video.",
    "video-frame-unreadable": "No frame could be decoded at that time.",
    "video-frame-timed-out": "Decoding that frame took longer than the video utilities allow.",
}


class FrameRefused(Exception):
    """A frame was not decoded, for one fixed reason."""

    def __init__(self, code: FrameRefusalCode) -> None:
        super().__init__(_MESSAGES[code])
        self.code: FrameRefusalCode = code


@dataclass(frozen=True)
class DecodedFrame:
    png: bytes
    width: int
    height: int
    requested_seconds: float
    #: The decoded frame's own time, counted from the start of the video.
    actual_seconds: float
    #: The turn applied to the pixels, as the stream's display matrix states it.
    rotation_applied: Literal[0, 90, 180, 270]
    #: A non-square pixel shape the PNG keeps rather than stretches.
    sample_aspect: str | None


async def decode_frame(
    copy: Path,
    artifact: Artifact,
    probe: VideoProbe,
    ffmpeg: MediaTool,
    requested_seconds: float,
) -> DecodedFrame:
    """Decode the frame shown at the requested time, or refuse with a fixed reason.

    ``copy`` is the file ``ArtifactStore.verified_copy`` made of the stored video,
    never the stored file itself.
    """

    if probe.artifact_sha256 != artifact.sha256 or not probe.can_save_frame:
        raise FrameRefused("video-frame-not-offered")
    duration = probe.duration_seconds
    if duration is None or not 0 <= requested_seconds <= duration:
        raise FrameRefused("video-frame-time-outside")
    until = probe.start_seconds + requested_seconds + SHOWN_WITHIN
    reach = 0.0
    while True:
        seek = round(requested_seconds - reach, 6)
        start = None if seek <= 0 else seek
        passed = await _frames_passed(copy, probe, ffmpeg, start, until)
        if passed:
            chosen: int | None = max(passed)
            break
        if start is None:
            # Read from the very beginning and nothing starts by then: the time
            # comes before the first picture, which is the one shown.
            chosen = None
            break
        if reach >= MAX_REACH_SECONDS:
            raise FrameRefused("video-frame-unreadable")
        reach = FIRST_REACH_SECONDS if reach == 0 else reach * 2

    png, shown = await _decode_one(copy, probe, ffmpeg, start, chosen)
    width, height = await asyncio.to_thread(_decoded_size, png)
    return DecodedFrame(
        png=png,
        width=width,
        height=height,
        requested_seconds=requested_seconds,
        actual_seconds=round(shown / 1_000_000 - probe.start_seconds, 6),
        rotation_applied=probe.video.rotation,
        sample_aspect=probe.video.sample_aspect,
    )


async def _frames_passed(
    copy: Path, probe: VideoProbe, ffmpeg: MediaTool, start: float | None, until: float
) -> list[int]:
    """The timestamps, in microseconds, of every frame from where ``start`` lands to ``until``."""

    answer = await _ffmpeg(
        ffmpeg,
        [
            *_source(copy, probe, start),
            "-vf",
            f"settb=1/1000000,trim=end={until:.6f},showinfo",
            "-f",
            "null",
            "-",
        ],
        stdout_limit=MAX_REPORT_BYTES,
        stderr_limit=MAX_PASSED_BYTES,
    )
    if answer.returncode != 0:
        raise FrameRefused("video-frame-unreadable")
    return [int(found) for found in FRAME_REPORT.findall(answer.stderr)]


async def _decode_one(
    copy: Path, probe: VideoProbe, ffmpeg: MediaTool, start: float | None, chosen: int | None
) -> tuple[bytes, int]:
    """The chosen frame as a PNG and its timestamp in microseconds; with none chosen, the first."""

    keep = "" if chosen is None else f",trim=start_pts={chosen}:end_pts={chosen + 1}"
    answer = await _ffmpeg(
        ffmpeg,
        [
            *_source(copy, probe, start),
            "-frames:v",
            "1",
            "-vf",
            f"settb=1/1000000{keep},showinfo",
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "pipe:1",
        ],
        stdout_limit=MAX_FRAME_BYTES,
        stderr_limit=MAX_REPORT_BYTES,
    )
    reported = [int(found) for found in FRAME_REPORT.findall(answer.stderr)]
    if answer.returncode != 0 or not answer.stdout or len(reported) != 1:
        raise FrameRefused("video-frame-unreadable")
    if chosen is not None and reported[0] != chosen:
        raise FrameRefused("video-frame-unreadable")
    return answer.stdout, reported[0]


def _source(copy: Path, probe: VideoProbe, start: float | None) -> list[str]:
    """The stored video's picture as ffmpeg reads it, from the keyframe ``start`` is sought to."""

    seek = [] if start is None else ["-noaccurate_seek", "-ss", f"{start:.6f}"]
    return [
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "info",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        DEMUXERS,
        *seek,
        "-copyts",
        "-i",
        f"file:{copy}",
        "-map",
        f"0:{probe.video.index}",
    ]


async def _ffmpeg(
    ffmpeg: MediaTool, arguments: list[str], *, stdout_limit: int, stderr_limit: int
) -> ToolResult:
    """Run ffmpeg within the decode's time and output bounds, its failures turned into refusals."""

    try:
        return await run_tool(
            ffmpeg.executable,
            arguments,
            seconds=DECODE_SECONDS,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
        )
    except TimeoutError:
        raise FrameRefused("video-frame-timed-out") from None
    except ToolOutputTooLarge:
        raise FrameRefused("video-frame-unreadable") from None
    except OSError as exc:
        # Found a moment ago, but it could not be started now: removed or
        # replaced meanwhile. The stored video is not at fault.
        code: ToolUnavailableCode = (
            "media-tool-missing" if isinstance(exc, FileNotFoundError) else "media-tool-unreadable"
        )
        raise MediaToolUnavailable(code) from exc


def _decoded_size(content: bytes) -> tuple[int, int]:
    """The size of a PNG that fully decodes within the probe's frame bounds."""

    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "PNG":
                raise FrameRefused("video-frame-unreadable")
            width, height = image.size
            if not (0 < width <= MAX_DIMENSION and 0 < height <= MAX_DIMENSION):
                raise FrameRefused("video-frame-unreadable")
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise FrameRefused("video-frame-unreadable") from exc
    return width, height
