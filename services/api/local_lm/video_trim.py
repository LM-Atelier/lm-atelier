"""Part of a stored video copied into a new video without re-encoding, from a keyframe.

A copy that does not re-encode can only begin on a keyframe, a frame the video
stores whole. The plan finds the keyframe at or before the requested start and
says where the new video will begin before anything is queued; the job copies
from that keyframe and checks what it wrote before keeping it. The end is never
predicted: it is measured once the new video exists.

ONE TIMELINE. A request, a plan, and a record's requested and actual start and
end count seconds from where the probe says the video's timestamps begin, as a
frame's time does; a record's measured stream entries are on the new file's own
timeline. ffprobe's read intervals take the file's own timestamps, so the plan
adds that start back for them; ffmpeg's ``-ss`` counts from the start by itself.

WHERE A COPY BEGINS. ffmpeg starts a stream copy at the keyframe its seek lands
on, and the two kinds of file differ in what they do with frames before the
seek point: an MP4 keeps them but hides them behind its edit list, Matroska
shows them. So an MP4 is cut from the keyframe's own time, which lands on it
and stores nothing before it, and a Matroska file from just past the keyframe.
Before seeking a Matroska file, ffmpeg steps back 3/23 s when any of its
streams has B-frames; the plan reads that from the file and seeks that much
further, so the seek still lands on the keyframe. A part that starts at the
first keyframe is copied from the very beginning, which keeps any sound that
starts before the picture.

WHAT IS CHECKED BEFORE KEEPING. The new file must hold one picture stream with
the source's codec, size, pixel shape and turn, then exactly the kept sound
streams, and nothing else. Its first shown frame is decoded and compared with
the source's keyframe decoded the same way, so a copy that begins anywhere else
is not kept; an MP4's stored frames, read past its edit list, must begin with
that keyframe too, so nothing cut away is kept hidden in the file. No picture
stored near the start may come before the keyframe or lack a time, which is
how pictures that lean on the part cut away are stored. Its end is read from
its last group of pictures.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict

from .artifacts import ToolOutput
from .media_process import ToolOutputTooLarge, run_tool
from .media_tools import MediaTool, MediaToolUnavailable, ToolUnavailableCode
from .models import Artifact
from .video_frames import DECODE_SECONDS, FRAME_REPORT, MAX_REPORT_BYTES
from .video_probe import (
    COPY_VIDEO_CODECS,
    DEMUXERS,
    MAX_DURATION_SECONDS,
    MAX_ERROR_BYTES,
    MAX_PROBE_BYTES,
    AudioStreamFacts,
    Container,
    VideoProbe,
    VideoProbeRefused,
    VideoStreamFacts,
    container_named,
    stream_facts,
)

#: How far past or before a keyframe a seek aims, so rounding never lands on its neighbour.
SEEK_PAST: Final = 0.0005
#: The step back ffmpeg takes before seeking a Matroska file with B-frames: 3/23 s, rounded up.
MATROSKA_STEP_BACK: Final = 0.130435
MIN_TRIM_SECONDS: Final = 0.1
#: Room a cut may take beyond the source's own size; it never needs more than the source.
CUT_ROOM: Final = 64 * 1024 * 1024
KEYFRAME_SECONDS: Final = 30.0
MAX_KEYFRAME_BYTES: Final = 64 * 1024
CUT_SECONDS: Final = 600.0
MEASURE_SECONDS: Final = 300.0
END_SECONDS: Final = 120.0
MAX_END_BYTES: Final = 16 * 1024 * 1024
MAX_HASH_BYTES: Final = 64 * 1024
#: How many of the new video's first packets are read to see that its start is whole.
_LEADING_PACKETS: Final = 32
#: Two times read from a file agree within this, or the start is not the one planned.
_SAME_TIME: Final = 2e-6

OutputFormat = Literal["mp4", "webm", "matroska"]
OutputMediaType = Literal["video/mp4", "video/webm", "video/x-matroska"]
TrimRefusalCode = Literal[
    "video-trim-not-offered",
    "video-trim-not-copyable",
    "video-trim-audio-not-offered",
    "video-trim-range-invalid",
    "video-trim-range-before-picture",
    "video-trim-keyframe-unknown",
    "video-trim-timed-out",
    "video-trim-start-moved",
    "video-trim-not-written",
    "video-trim-output-mismatch",
    "video-trim-start-unverified",
    "video-trim-unmeasured",
    "video-trim-start-incomplete",
    "video-trim-exact-not-offered",
    "video-trim-encoder-missing",
    "video-trim-exact-audio-not-offered",
    "video-trim-frames-unknown",
    "video-trim-exact-too-long",
    "video-trim-frames-moved",
    "video-trim-not-encoded",
    "video-trim-frames-unverified",
    "video-trim-reencode-unlike",
    "video-trim-start-unchecked",
]
TrimMode = Literal["copy", "exact"]

_MESSAGES: Final[dict[TrimRefusalCode, str]] = {
    "video-trim-not-offered": "This video cannot be trimmed.",
    "video-trim-not-copyable": (
        "This video's picture cannot be copied into a new file of the same kind."
    ),
    "video-trim-audio-not-offered": (
        "This video's sound cannot be copied unchanged, so it can only be left out."
    ),
    "video-trim-range-invalid": "Choose a part at least 0.1 s long, within the video.",
    "video-trim-range-before-picture": "That part ends before the video's first picture.",
    "video-trim-keyframe-unknown": "No keyframe could be found to start the copy from.",
    "video-trim-timed-out": (
        "This video took longer to read, copy or re-encode than the video utilities allow."
    ),
    "video-trim-start-moved": (
        "The video would no longer start where the check showed. Check the cut again."
    ),
    "video-trim-not-written": "The chosen part could not be copied into a new video.",
    "video-trim-output-mismatch": (
        "The new video did not come out as the check showed, so it was not kept."
    ),
    "video-trim-start-unverified": (
        "The new video did not begin at the keyframe the check showed, so it was not kept."
    ),
    "video-trim-unmeasured": "The new video could not be measured, so it was not kept.",
    "video-trim-start-incomplete": (
        "The keyframe this part starts on needs pictures from before it, so the new video "
        "was not kept."
    ),
    "video-trim-exact-not-offered": (
        "This video cannot be cut on exact frames. It can still be trimmed from a keyframe."
    ),
    "video-trim-encoder-missing": (
        "The FFmpeg on this computer cannot make H.264 video and AAC sound, so it cannot cut "
        "on exact frames."
    ),
    "video-trim-exact-audio-not-offered": (
        "This video's sound cannot be re-encoded for an exact cut, so it can only be left out."
    ),
    "video-trim-frames-unknown": (
        "The frames of this part could not be read, each with a time of its own."
    ),
    "video-trim-exact-too-long": (
        "That part is too long, or starts too far after a keyframe, to re-encode. Choose a "
        "shorter part, or trim it from a keyframe."
    ),
    "video-trim-frames-moved": (
        "The part would no longer hold the frames the check showed. Check the cut again."
    ),
    "video-trim-not-encoded": "The chosen part could not be re-encoded into a new video.",
    "video-trim-frames-unverified": (
        "The new video did not hold exactly the frames the check showed, so it was not kept."
    ),
    "video-trim-reencode-unlike": (
        "The re-encoded part did not come out close enough to the original, so it was not kept."
    ),
    "video-trim-start-unchecked": (
        "Where the part starts could not be checked against a second decode of the original, so "
        "the new video was not kept."
    ),
}

_OUTPUTS: Final[dict[OutputFormat, tuple[OutputMediaType, str]]] = {
    "mp4": ("video/mp4", ".mp4"),
    "webm": ("video/webm", ".webm"),
    "matroska": ("video/x-matroska", ".mkv"),
}
_WEBM_VIDEO: Final = frozenset({"vp8", "vp9", "av1"})
_WEBM_AUDIO: Final = frozenset({"opus", "vorbis"})
_HASH_LINE: Final = re.compile(r"[^,]*(?:,[^,]*){4},\s*([0-9a-f]{64})")


class TrimRefused(Exception):
    """A trim was not planned or not kept, for one fixed reason."""

    def __init__(self, code: TrimRefusalCode) -> None:
        super().__init__(_MESSAGES[code])
        self.code: TrimRefusalCode = code


class VideoTrimPreview(BaseModel):
    """Where a trim of a stored video would begin, worked out before anything is queued.

    A copy says where it begins and leaves its end to be measured. A cut on
    exact frames says both, and how many frames it holds.
    """

    model_config = ConfigDict(frozen=True)

    version: Literal[1] = 1
    mode: TrimMode
    artifact_id: str
    artifact_sha256: str
    requested_start_seconds: float
    requested_end_seconds: float
    keep_audio: bool
    #: Where the new video begins on the original's timeline: 0 when copied from
    #: the beginning, and for an exact cut the time of its first frame.
    start_seconds: float
    #: The keyframe a copy's picture begins with, or that an exact cut's check decodes from.
    keyframe_seconds: float
    from_beginning: bool
    #: For an exact cut: its last frame's time, where the part ends, and how many frames it holds.
    last_frame_seconds: float | None
    end_seconds: float | None
    frame_count: int | None
    #: For an exact cut whose start comes before the video's first picture,
    #: which it then starts on.
    began_at_first_picture: bool
    #: The part chosen is the whole video, kept as it is; such a trim is refused.
    keeps_whole_video: bool
    audio_streams_kept: int
    #: Subtitle, cover picture and other streams the new video leaves out.
    omitted_streams: int
    format: OutputFormat
    media_type: OutputMediaType


@dataclass(frozen=True)
class TrimPlan:
    preview: VideoTrimPreview
    #: The probe's start of the video's timestamps, which read intervals add back.
    timeline_start: float
    video: VideoStreamFacts
    audio: tuple[AudioStreamFacts, ...]
    suffix: str
    #: ffmpeg's ``-ss`` for the cut; None copies from the very beginning.
    seek_seconds: float | None
    #: How much the cut reads, from where it seeks to the requested end.
    window_seconds: float


@dataclass(frozen=True)
class DecodedPicture:
    #: The frame's own timestamp in the file it was decoded from.
    pts_seconds: float
    sha256: str


@dataclass(frozen=True)
class TrimMeasurement:
    """What the new video holds, read from the new file itself."""

    #: Where the file says its picture stream starts, when it says.
    video_start_seconds: float | None
    #: The first frame a player shows, on the new file's own timeline.
    first_shown_seconds: float
    #: Where its picture ends: its last frame's end, or that frame's start with no length
    #: stated, and never past the length the file states for its picture.
    video_end_seconds: float
    #: Frames as the file stores them, which can include frames its edit list hides.
    frames_stored: int
    audio: tuple[tuple[str, float | None, int], ...]


async def plan_trim(
    copy: Path,
    artifact: Artifact,
    probe: VideoProbe,
    ffprobe: MediaTool,
    requested_start: float,
    requested_end: float,
    keep_audio: bool,
) -> TrimPlan:
    """Where a copy of the requested part would begin, or why it cannot be made.

    ``copy`` is the file ``ArtifactStore.verified_copy`` made of the stored video.
    """

    if probe.artifact_sha256 != artifact.sha256 or not probe.can_trim:
        raise TrimRefused("video-trim-not-offered")
    duration = probe.duration_seconds
    if duration is None:
        raise TrimRefused("video-trim-not-offered")
    if keep_audio and probe.audio and not probe.can_keep_audio:
        raise TrimRefused("video-trim-audio-not-offered")
    kept = tuple(probe.audio) if keep_audio else ()
    output = _output_format(artifact, probe, kept)
    start = round(requested_start, 6)
    end = round(requested_end, 6)
    if not (start >= 0 and end <= round(duration, 6) and end - start >= MIN_TRIM_SECONDS - 1e-9):
        raise TrimRefused("video-trim-range-invalid")

    first = await keyframe_at(copy, probe, ffprobe, max(probe.start_seconds, 0.0))
    target = max(probe.start_seconds + start + SEEK_PAST, 0.0)
    found = await keyframe_at(copy, probe, ffprobe, target)
    from_beginning = abs(found - first) <= _SAME_TIME
    if found > target and not from_beginning:
        # A keyframe past the requested start that is not the first one: the
        # answer is not the one asked for, so nothing is guessed from it.
        raise TrimRefused("video-trim-keyframe-unknown")
    keyframe = round(found - probe.start_seconds, 6)
    if end <= keyframe + 0.001:
        raise TrimRefused("video-trim-range-before-picture")

    if from_beginning:
        seek: float | None = None
        window = end
    elif probe.container == "mp4":
        seek = keyframe
        window = end - seek
    else:
        seek = keyframe + SEEK_PAST
        if await has_b_frames(copy, ffprobe):
            seek += MATROSKA_STEP_BACK
        window = max(end - seek, 0.001)

    media_type, suffix = _OUTPUTS[output]
    preview = VideoTrimPreview(
        mode="copy",
        artifact_id=artifact.id,
        artifact_sha256=artifact.sha256,
        requested_start_seconds=start,
        requested_end_seconds=end,
        keep_audio=keep_audio,
        start_seconds=0.0 if from_beginning else keyframe,
        keyframe_seconds=keyframe,
        from_beginning=from_beginning,
        last_frame_seconds=None,
        end_seconds=None,
        frame_count=None,
        began_at_first_picture=False,
        keeps_whole_video=(
            from_beginning
            and end >= duration - SEEK_PAST
            and (keep_audio or not probe.audio)
            and probe.omitted_streams == 0
        ),
        audio_streams_kept=len(kept),
        omitted_streams=probe.omitted_streams,
        format=output,
        media_type=media_type,
    )
    return TrimPlan(
        preview=preview,
        timeline_start=probe.start_seconds,
        video=probe.video,
        audio=kept,
        suffix=suffix,
        seek_seconds=None if seek is None else round(seek, 6),
        window_seconds=round(window, 6),
    )


async def keyframe_picture(copy: Path, plan: TrimPlan, ffmpeg: MediaTool) -> DecodedPicture:
    """The source's planned keyframe, decoded, with the time it really has."""

    picture = await _picture(
        copy,
        plan.video.index,
        ffmpeg,
        keyframe=plan.preview.keyframe_seconds,
        refusal="video-trim-keyframe-unknown",
    )
    expected = plan.timeline_start + plan.preview.keyframe_seconds
    if not _same_time(picture.pts_seconds, expected):
        raise TrimRefused("video-trim-keyframe-unknown")
    return picture


async def cut_video(copy: Path, plan: TrimPlan, ffmpeg: MediaTool, output: ToolOutput) -> None:
    """Copy the planned part, unchanged, into the file the store made for it."""

    arguments = [
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        DEMUXERS,
    ]
    if plan.seek_seconds is not None:
        arguments += ["-ss", f"{plan.seek_seconds:.6f}"]
    arguments += ["-t", f"{plan.window_seconds:.6f}", "-i", f"file:{copy}"]
    arguments += ["-map", f"0:{plan.video.index}"]
    for audio in plan.audio:
        arguments += ["-map", f"0:{audio.index}"]
    arguments += ["-c", "copy", "-map_chapters", "-1", "-fs", str(output.maximum_bytes)]
    if plan.preview.format == "mp4":
        # No timecode track: the new video is one picture stream and its kept sound.
        arguments += ["-movflags", "+faststart", "-write_tmcd", "0"]
    # The kind is named, so the private file's name never picks it; the file
    # exists already, made by the store, so it is overwritten.
    arguments += ["-f", plan.preview.format, "-y", f"file:{output.path}"]
    try:
        answer = await run_tool(
            ffmpeg.executable,
            arguments,
            seconds=CUT_SECONDS,
            stdout_limit=MAX_HASH_BYTES,
            stderr_limit=MAX_REPORT_BYTES,
        )
    except TimeoutError:
        raise TrimRefused("video-trim-timed-out") from None
    except ToolOutputTooLarge:
        raise TrimRefused("video-trim-not-written") from None
    except OSError as exc:
        raise tool_unavailable(exc) from exc
    if answer.returncode != 0:
        raise TrimRefused("video-trim-not-written")


async def check_trim(
    output: Path,
    plan: TrimPlan,
    ffprobe: MediaTool,
    ffmpeg: MediaTool,
    source: DecodedPicture,
) -> TrimMeasurement:
    """Measure the new video, and refuse it unless it is what the plan said it would be."""

    facts = await measure_output(output, ffprobe)
    container = container_named(facts.get("format", {}).get("format_name"))
    streams = facts.get("streams")
    if not isinstance(streams, list) or not all(isinstance(item, dict) for item in streams):
        raise TrimRefused("video-trim-unmeasured")
    expected_container: Container = "mp4" if plan.preview.format == "mp4" else "matroska"
    if container != expected_container or len(streams) != 1 + len(plan.audio):
        raise TrimRefused("video-trim-output-mismatch")
    try:
        video = stream_facts(streams[0])
        kept = [stream_facts(stream) for stream in streams[1:]]
    except VideoProbeRefused:
        raise TrimRefused("video-trim-output-mismatch") from None
    if not isinstance(video, VideoStreamFacts) or (
        video.codec,
        video.width,
        video.height,
        video.sample_aspect,
        video.rotation,
    ) != (
        plan.video.codec,
        plan.video.width,
        plan.video.height,
        plan.video.sample_aspect,
        plan.video.rotation,
    ):
        raise TrimRefused("video-trim-output-mismatch")
    stored = packet_count(streams[0])
    if stored < 1:
        raise TrimRefused("video-trim-output-mismatch")
    audio: list[tuple[str, float | None, int]] = []
    for planned, stream, facts_of in zip(plan.audio, streams[1:], kept, strict=True):
        if not isinstance(facts_of, AudioStreamFacts) or facts_of.codec != planned.codec:
            raise TrimRefused("video-trim-output-mismatch")
        audio.append((facts_of.codec, stated_time(stream.get("start_time")), packet_count(stream)))

    end = await _video_end(output, ffprobe, expected_container)
    await _require_whole_start(output, ffprobe)
    shown = await _picture(output, 0, ffmpeg, keyframe=None)
    if shown.sha256 != source.sha256:
        raise TrimRefused("video-trim-start-unverified")
    if plan.preview.format == "mp4" and not plan.preview.from_beginning:
        # Read past the edit list: what the file stores must begin with the
        # keyframe too, so no part the person cut away is kept hidden in it.
        stored_first = await _picture(output, 0, ffmpeg, keyframe=None, past_edit_list=True)
        if stored_first.sha256 != source.sha256:
            raise TrimRefused("video-trim-start-unverified")
    stated = stated_time(streams[0].get("duration"))
    if stated is not None and stated > 0:
        # A copy can store a frame past the requested end without the frames
        # before it; the length the file states is where a player stops.
        end = min(end, shown.pts_seconds + stated)
    return TrimMeasurement(
        video_start_seconds=stated_time(streams[0].get("start_time")),
        first_shown_seconds=shown.pts_seconds,
        video_end_seconds=end,
        frames_stored=stored,
        audio=tuple(audio),
    )


def trim_record(
    source: Artifact,
    plan: TrimPlan,
    measured: TrimMeasurement,
    tool: dict[str, str],
    measured_with: dict[str, str],
) -> dict[str, Any]:
    """What was asked for and what the new video measured.

    The requested and actual start and end are on the original's timeline;
    the ``video`` and ``audio`` entries are measured on the new file's own.
    Only ``source_artifact_id`` names another artifact, so the record keeps the
    source and nothing else.
    """

    preview = plan.preview
    end = preview.keyframe_seconds + measured.video_end_seconds - measured.first_shown_seconds
    return {
        "action": "trim",
        "mode": "copy",
        "source_artifact_id": source.id,
        "source_sha256": source.sha256,
        "requested_start_seconds": preview.requested_start_seconds,
        "requested_end_seconds": preview.requested_end_seconds,
        "actual_start_seconds": preview.start_seconds,
        "keyframe_seconds": preview.keyframe_seconds,
        "actual_end_seconds": round(end, 6),
        "from_beginning": preview.from_beginning,
        "keep_audio": preview.keep_audio,
        "format": preview.format,
        "media_type": preview.media_type,
        "video": {
            "codec": plan.video.codec,
            "start_seconds": _rounded(measured.video_start_seconds),
            "first_shown_seconds": round(measured.first_shown_seconds, 6),
            "end_seconds": round(measured.video_end_seconds, 6),
            "frames_stored": measured.frames_stored,
            "display_width": plan.video.display_width,
            "display_height": plan.video.display_height,
            "rotation": plan.video.rotation,
            "sample_aspect": plan.video.sample_aspect,
        },
        "audio": [
            {"codec": codec, "start_seconds": _rounded(start), "packets": packets}
            for codec, start, packets in measured.audio
        ],
        "omitted_streams": preview.omitted_streams,
        # What only a cut on exact frames records.
        "frames": None,
        "quality": None,
        "encoding": None,
        "tool": tool,
        "measured_with": measured_with,
    }


def _output_format(
    artifact: Artifact, probe: VideoProbe, kept: tuple[AudioStreamFacts, ...]
) -> OutputFormat:
    """The kind of new file, from a fixed table and never from a file name."""

    if probe.video.codec not in COPY_VIDEO_CODECS[probe.container]:
        raise TrimRefused("video-trim-not-copyable")
    if probe.container == "mp4":
        return "mp4"
    if (
        (artifact.media_type or "").casefold() == "video/webm"
        and probe.video.codec in _WEBM_VIDEO
        and all(stream.codec in _WEBM_AUDIO for stream in kept)
    ):
        return "webm"
    return "matroska"


async def keyframe_at(copy: Path, probe: VideoProbe, ffprobe: MediaTool, absolute: float) -> float:
    """The keyframe at or before a time in the file's own timestamps."""

    try:
        answer = await run_tool(
            ffprobe.executable,
            [
                "-v",
                "error",
                "-hide_banner",
                "-protocol_whitelist",
                "file",
                "-format_whitelist",
                DEMUXERS,
                # The stream's own index: a cover picture counts among the
                # video streams, so "v:0" could name it.
                "-select_streams",
                str(probe.video.index),
                "-skip_frame",
                "nokey",
                "-show_entries",
                "frame=key_frame,pts_time,best_effort_timestamp_time",
                "-read_intervals",
                f"{absolute:.6f}%+#1",
                "-of",
                "json",
                f"file:{copy}",
            ],
            seconds=KEYFRAME_SECONDS,
            stdout_limit=MAX_KEYFRAME_BYTES,
            stderr_limit=MAX_ERROR_BYTES,
        )
    except TimeoutError:
        raise TrimRefused("video-trim-timed-out") from None
    except ToolOutputTooLarge:
        raise TrimRefused("video-trim-keyframe-unknown") from None
    except OSError as exc:
        raise tool_unavailable(exc) from exc
    if answer.returncode != 0:
        raise TrimRefused("video-trim-keyframe-unknown")
    return keyframe_time(answer.stdout)


async def has_b_frames(copy: Path, ffprobe: MediaTool) -> bool:
    """Whether any stream of the file has B-frames, as ffmpeg reads the file before seeking it."""

    try:
        answer = await run_tool(
            ffprobe.executable,
            [
                "-v",
                "error",
                "-hide_banner",
                "-protocol_whitelist",
                "file",
                "-format_whitelist",
                DEMUXERS,
                "-show_entries",
                "stream=has_b_frames",
                "-of",
                "json",
                f"file:{copy}",
            ],
            seconds=KEYFRAME_SECONDS,
            stdout_limit=MAX_KEYFRAME_BYTES,
            stderr_limit=MAX_ERROR_BYTES,
        )
    except TimeoutError:
        raise TrimRefused("video-trim-timed-out") from None
    except ToolOutputTooLarge:
        raise TrimRefused("video-trim-keyframe-unknown") from None
    except OSError as exc:
        raise tool_unavailable(exc) from exc
    if answer.returncode != 0:
        raise TrimRefused("video-trim-keyframe-unknown")
    return b_frames_stated(answer.stdout)


def b_frames_stated(content: bytes) -> bool:
    """Whether ffprobe's answer states B-frames for any stream; an unreadable answer refuses."""

    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TrimRefused("video-trim-keyframe-unknown") from None
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not all(isinstance(stream, dict) for stream in streams):
        raise TrimRefused("video-trim-keyframe-unknown")
    stated = [stream["has_b_frames"] for stream in streams if "has_b_frames" in stream]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in stated):
        raise TrimRefused("video-trim-keyframe-unknown")
    return any(value > 0 for value in stated)


def keyframe_time(content: bytes) -> float:
    """The first keyframe's time in ffprobe's answer, or a refusal when there is none to trust."""

    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TrimRefused("video-trim-keyframe-unknown") from None
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list):
        raise TrimRefused("video-trim-keyframe-unknown")
    for frame in frames:
        if isinstance(frame, dict) and frame.get("key_frame") == 1:
            stated = frame.get("pts_time")
            if stated is None or stated == "N/A":
                stated = frame.get("best_effort_timestamp_time")
            seconds = stated_time(stated)
            if seconds is None:
                break
            return seconds
    raise TrimRefused("video-trim-keyframe-unknown")


async def _picture(
    path: Path,
    index: int,
    ffmpeg: MediaTool,
    *,
    keyframe: float | None,
    past_edit_list: bool = False,
    refusal: TrimRefusalCode = "video-trim-start-unverified",
) -> DecodedPicture:
    """One decoded frame's own timestamp and digest, or ``refusal`` when there is none.

    With ``keyframe``, the keyframe at that time; without, the first frame a
    player shows, after anything an edit list hides, or with
    ``past_edit_list`` the first frame an MP4 stores. The turn is not
    applied, so the stored pictures are what is compared.
    """

    arguments = [
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "info",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        DEMUXERS,
        "-noautorotate",
    ]
    if keyframe is not None:
        arguments += ["-skip_frame", "nokey", "-ss", f"{max(keyframe - SEEK_PAST, 0.0):.6f}"]
    if past_edit_list:
        arguments += ["-ignore_editlist", "1"]
    arguments += [
        "-copyts",
        "-i",
        f"file:{path}",
        "-map",
        f"0:{index}",
        "-frames:v",
        "1",
        "-vf",
        "settb=1/1000000,showinfo",
        "-f",
        "framehash",
        "-hash",
        "sha256",
        "pipe:1",
    ]
    try:
        answer = await run_tool(
            ffmpeg.executable,
            arguments,
            seconds=DECODE_SECONDS,
            stdout_limit=MAX_HASH_BYTES,
            stderr_limit=MAX_REPORT_BYTES,
        )
    except TimeoutError:
        raise TrimRefused("video-trim-timed-out") from None
    except ToolOutputTooLarge:
        raise TrimRefused(refusal) from None
    except OSError as exc:
        raise tool_unavailable(exc) from exc
    # The filter may report a frame or two more before ffmpeg stops; the first is the one hashed.
    reports = FRAME_REPORT.findall(answer.stderr)
    digest = frame_digest(answer.stdout)
    if answer.returncode != 0 or not reports or digest is None:
        raise TrimRefused(refusal)
    return DecodedPicture(pts_seconds=int(reports[0]) / 1_000_000, sha256=digest)


def frame_digest(content: bytes) -> str | None:
    """The one frame's sha256 in ffmpeg's framehash output, when there is exactly one."""

    try:
        text = content.decode("ascii")
    except UnicodeDecodeError:
        return None
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if len(lines) != 1:
        return None
    match = _HASH_LINE.fullmatch(lines[0].strip())
    return None if match is None else match.group(1)


async def measure_output(output: Path, ffprobe: MediaTool) -> dict[str, Any]:
    """ffprobe's account of a new video's container and streams, or a refusal."""

    try:
        answer = await run_tool(
            ffprobe.executable,
            [
                "-v",
                "error",
                "-hide_banner",
                "-protocol_whitelist",
                "file",
                "-format_whitelist",
                DEMUXERS,
                "-count_packets",
                "-show_entries",
                "format=format_name,start_time,duration"
                ":stream=index,codec_type,codec_name,width,height,sample_aspect_ratio,"
                "start_time,duration,nb_read_packets,time_base,channels,sample_rate,"
                "pix_fmt,color_range,color_space,color_transfer,color_primaries,field_order"
                ":stream_disposition=attached_pic"
                ":stream_side_data=rotation"
                ":stream_tags=rotate",
                "-of",
                "json",
                f"file:{output}",
            ],
            seconds=MEASURE_SECONDS,
            stdout_limit=MAX_PROBE_BYTES,
            stderr_limit=MAX_ERROR_BYTES,
        )
    except (TimeoutError, ToolOutputTooLarge):
        raise TrimRefused("video-trim-unmeasured") from None
    except OSError as exc:
        raise tool_unavailable(exc) from exc
    if answer.returncode != 0:
        raise TrimRefused("video-trim-unmeasured")
    try:
        payload = json.loads(answer.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TrimRefused("video-trim-unmeasured") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("format"), dict):
        raise TrimRefused("video-trim-unmeasured")
    return payload


async def _video_end(output: Path, ffprobe: MediaTool, container: Container) -> float:
    """Where the new video's picture ends: the latest end among its picture packets.

    An MP4 file is read from past any end it can have, so the read begins at
    its last keyframe and runs to the end; when that lists no packet, as it now
    and then does, every packet of the picture stream is read instead. A
    Matroska file sought that far can list a few of its last packets and stop
    there, without an error, so every packet of its picture is read, within the
    same bounds.
    """

    if container == "mp4":
        listed = await _end_packets(output, ffprobe, f"{2 * MAX_DURATION_SECONDS}%")
        if listed.strip():
            return last_packet_end(listed)
    return last_packet_end(await _end_packets(output, ffprobe, None))


async def _end_packets(output: Path, ffprobe: MediaTool, interval: str | None) -> bytes:
    """The picture stream's packet times and lengths, from ``interval`` or from the start."""

    arguments = [
        "-v",
        "error",
        "-hide_banner",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        DEMUXERS,
        "-select_streams",
        "0",
    ]
    if interval is not None:
        arguments += ["-read_intervals", interval]
    arguments += [
        "-show_entries",
        "packet=pts_time,duration_time",
        "-of",
        "csv=p=0",
        f"file:{output}",
    ]
    try:
        answer = await run_tool(
            ffprobe.executable,
            arguments,
            seconds=END_SECONDS,
            stdout_limit=MAX_END_BYTES,
            stderr_limit=MAX_ERROR_BYTES,
        )
    except (TimeoutError, ToolOutputTooLarge):
        raise TrimRefused("video-trim-unmeasured") from None
    except OSError as exc:
        raise tool_unavailable(exc) from exc
    if answer.returncode != 0:
        raise TrimRefused("video-trim-unmeasured")
    return answer.stdout


async def _require_whole_start(output: Path, ffprobe: MediaTool) -> None:
    """Refuse a new video whose first pictures need ones from before its first keyframe.

    A keyframe that opens a group of pictures may be followed by pictures that
    lean on the group before it, which the copy left out; they are stored
    without a time of their own or before the keyframe's.
    """

    try:
        answer = await run_tool(
            ffprobe.executable,
            [
                "-v",
                "error",
                "-hide_banner",
                "-protocol_whitelist",
                "file",
                "-format_whitelist",
                DEMUXERS,
                "-select_streams",
                "0",
                "-read_intervals",
                f"%+#{_LEADING_PACKETS}",
                "-show_entries",
                "packet=pts_time,flags",
                "-of",
                "csv=p=0",
                f"file:{output}",
            ],
            seconds=END_SECONDS,
            stdout_limit=MAX_KEYFRAME_BYTES,
            stderr_limit=MAX_ERROR_BYTES,
        )
    except (TimeoutError, ToolOutputTooLarge):
        raise TrimRefused("video-trim-unmeasured") from None
    except OSError as exc:
        raise tool_unavailable(exc) from exc
    if answer.returncode != 0:
        raise TrimRefused("video-trim-unmeasured")
    leading_pictures_complete(answer.stdout)


def leading_pictures_complete(content: bytes) -> None:
    """Refuse when an early packet comes before the first keyframe's time or has no time."""

    try:
        text = content.decode("ascii")
    except UnicodeDecodeError:
        raise TrimRefused("video-trim-unmeasured") from None
    rows = [listed_fields(line, 2) for line in text.splitlines() if line.strip()]
    if not rows or any(len(row) != 2 for row in rows):
        raise TrimRefused("video-trim-unmeasured")
    first = stated_time(rows[0][0])
    if first is None or "K" not in rows[0][1]:
        raise TrimRefused("video-trim-start-incomplete")
    for stated, _flags in rows[1:]:
        seconds = stated_time(stated)
        if seconds is None or seconds < first:
            raise TrimRefused("video-trim-start-incomplete")


def last_packet_end(content: bytes) -> float:
    """The latest end among listed packets; a packet with no stated length ends where it starts."""

    try:
        text = content.decode("ascii")
    except UnicodeDecodeError:
        raise TrimRefused("video-trim-unmeasured") from None
    ends: list[float] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = listed_fields(line, 2)
        if len(fields) != 2:
            raise TrimRefused("video-trim-unmeasured")
        pts = stated_time(fields[0])
        if pts is None:
            if fields[0] == "N/A":
                continue
            raise TrimRefused("video-trim-unmeasured")
        length = 0.0 if fields[1] == "N/A" else stated_time(fields[1])
        if length is None or length < 0:
            raise TrimRefused("video-trim-unmeasured")
        ends.append(pts + length)
    if not ends:
        raise TrimRefused("video-trim-unmeasured")
    return max(ends)


def listed_fields(line: str, count: int) -> list[str]:
    """One row of ffprobe's csv listing, less the empty fields a packet's side data adds.

    A packet that carries side data, such as the transparency of a VP9 picture,
    is listed with one more field per side data entry, empty when none of its
    entries were asked for. Anything else past ``count`` fields stays, so the
    row does not have the shape asked for.
    """

    fields = line.strip().split(",")
    while len(fields) > count and fields[-1] == "":
        fields.pop()
    return fields


def stated_time(value: object) -> float | None:
    """A time as ffprobe states it: a short decimal within the bounds a video may have."""

    if not isinstance(value, str) or not 0 < len(value) <= 32:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if seconds != seconds or abs(seconds) > 2 * MAX_DURATION_SECONDS:
        return None
    return seconds


def _rounded(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds, 6)


def packet_count(stream: dict[str, Any]) -> int:
    """How many packets ffprobe counted in one stream of a new video, or a refusal."""

    value = stream.get("nb_read_packets")
    if not isinstance(value, str) or not value.isdigit() or len(value) > 12:
        raise TrimRefused("video-trim-unmeasured")
    return int(value)


def _same_time(found: float, expected: float) -> bool:
    """Whether a frame's time, read in whole microseconds, is the expected one."""

    return abs(found - expected) <= _SAME_TIME


def tool_unavailable(exc: OSError) -> MediaToolUnavailable:
    # Found a moment ago, but it could not be started now: removed or replaced
    # meanwhile. The stored video is not at fault.
    code: ToolUnavailableCode = (
        "media-tool-missing" if isinstance(exc, FileNotFoundError) else "media-tool-unreadable"
    )
    return MediaToolUnavailable(code)
