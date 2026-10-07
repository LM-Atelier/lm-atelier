"""What a stored video is, read by ffprobe within fixed bounds before any utility uses it.

The probe answers three things about one stored video: what it is (container,
streams, duration, frame size and how frames are shown), what a utility may do
with it, and, for each thing it may not do, why.

BOUNDS. ffprobe reads a private copy of the stored video, made from the very
bytes that verify against its digest, so what it reads is what the record
names even if the stored file changes meanwhile. It reads only through the file
protocol, and only with the MP4-family and Matroska demuxers. A playlist, a
concatenation list or any other container that names further files is therefore
refused before it opens anything. Input size and stream count are
capped, the probe has a time limit, and both of ffprobe's outputs are bounded.

WHAT IT DOES NOT CLAIM. These are the facts the container states, not a decode
of every frame. A frame rate is called constant only when the stream's nominal
and average rates agree, which is how ffprobe describes the stream.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict

from .artifacts import ArtifactStore
from .media_process import ToolOutputTooLarge, run_tool
from .media_tools import MediaTool, MediaToolUnavailable, ToolUnavailableCode
from .models import Artifact

Container = Literal["mp4", "matroska"]
FrameRateForm = Literal["constant", "variable", "unknown"]
ProbeRefusalCode = Literal[
    "video-probe-not-a-video",
    "video-probe-too-large",
    "video-probe-unreadable",
    "video-probe-timed-out",
    "video-probe-no-video-stream",
    "video-probe-too-many-streams",
]
LimitCode = Literal[
    "video-codec-unsupported",
    "audio-codec-unsupported",
    "video-duration-unknown",
    "video-duration-too-long",
    "video-frame-too-large",
    "video-rotation-unsupported",
]
#: Why a video cannot be cut on exact frames; the last only leaves its sound out.
ExactTrimLimitCode = Literal[
    "exact-trim-picture-unsupported",
    "exact-trim-frame-size-unsupported",
    "exact-trim-turned-unsupported",
    "exact-trim-audio-unsupported",
]

PROBE_VERSION: Final = 1

MAX_INPUT_BYTES: Final = 4 * 1024 * 1024 * 1024
MAX_DURATION_SECONDS: Final = 4 * 60 * 60
MAX_STREAMS: Final = 16
MAX_DIMENSION: Final = 8192
PROBE_SECONDS: Final = 30.0
MAX_PROBE_BYTES: Final = 256 * 1024
MAX_ERROR_BYTES: Final = 64 * 1024

#: The only demuxers allowed to open a stored video, by ffmpeg's own names.
DEMUXERS: Final = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm"
_CONTAINERS: Final[dict[str, Container]] = {
    "mov,mp4,m4a,3gp,3g2,mj2": "mp4",
    "matroska,webm": "matroska",
}
#: Video codecs a frame can be decoded from and a trim can copy.
VIDEO_CODECS: Final = frozenset({"h264", "hevc", "vp8", "vp9", "av1", "mpeg4"})
#: Audio codecs a trim can copy unchanged into the same container.
AUDIO_CODECS: Final = frozenset({"aac", "mp3", "opus", "vorbis", "flac", "alac", "ac3", "eac3"})
#: What a trim copies into a new file of each kind without re-encoding. Matroska
#: takes every codec above; an MP4 takes fewer, and never Vorbis.
COPY_VIDEO_CODECS: Final[dict[Container, frozenset[str]]] = {
    "mp4": frozenset({"h264", "hevc", "av1", "vp9", "mpeg4"}),
    "matroska": VIDEO_CODECS,
}
COPY_AUDIO_CODECS: Final[dict[Container, frozenset[str]]] = {
    "mp4": frozenset({"aac", "mp3", "opus", "alac", "ac3", "eac3"}),
    "matroska": AUDIO_CODECS,
}

#: What a cut on exact frames re-encodes without changing how the picture
#: looks: 8-bit 4:2:0 in limited range, standard-range colours, whole frames.
EXACT_PIXEL_FORMAT: Final = "yuv420p"
_EXACT_RANGES: Final = frozenset({"tv"})
_EXACT_TRANSFERS: Final = frozenset({"bt709", "smpte170m", "bt470bg", "bt470m"})
_EXACT_PRIMARIES: Final = frozenset({"bt709", "smpte170m", "bt470bg"})
_EXACT_SPACES: Final = frozenset({"bt709", "smpte170m", "bt470bg"})
_EXACT_FIELD_ORDERS: Final = frozenset({"progressive"})
#: The largest frame an exact cut re-encodes: no side over 4096, no more pixels than 2160p.
EXACT_MAX_SIDE: Final = 4096
EXACT_MAX_PIXELS: Final = 3840 * 2160
#: Sound an exact cut re-encodes as AAC: one or two channels at a rate AAC takes.
EXACT_AUDIO_CODECS: Final = AUDIO_CODECS | frozenset({"pcm_s16le", "pcm_s24le", "pcm_f32le"})
EXACT_SAMPLE_RATES: Final = frozenset(
    {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000}
)

_ENTRIES: Final = (
    "format=format_name,duration,start_time,nb_streams"
    ":stream=index,codec_type,codec_name,width,height,sample_aspect_ratio,"
    "r_frame_rate,avg_frame_rate,time_base,duration,channels,sample_rate,"
    "pix_fmt,color_range,color_space,color_transfer,color_primaries,field_order"
    ":stream_disposition=attached_pic"
    ":stream_side_data=rotation"
    ":stream_tags=rotate"
)
_CODEC_NAME: Final = re.compile(r"[a-z0-9_]{1,32}")
_PROPERTY_NAME: Final = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_RATIO: Final = re.compile(r"(\d{1,10})[/:](\d{1,10})")

_REFUSAL_MESSAGES: Final[dict[ProbeRefusalCode, str]] = {
    "video-probe-not-a-video": "This file is not a video.",
    "video-probe-too-large": "This video is larger than the video utilities read.",
    "video-probe-unreadable": "This video could not be read as an MP4 or Matroska file.",
    "video-probe-timed-out": "Reading this video took longer than the video utilities allow.",
    "video-probe-no-video-stream": "This file has no video stream.",
    "video-probe-too-many-streams": "This video has more streams than the video utilities read.",
}


class VideoProbeRefused(Exception):
    """The stored file cannot be described as a video the utilities read."""

    def __init__(self, code: ProbeRefusalCode) -> None:
        super().__init__(_REFUSAL_MESSAGES[code])
        self.code: ProbeRefusalCode = code


class VideoStreamFacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    codec: str
    width: int
    height: int
    #: The frame as it is shown: pixel shape and rotation applied.
    display_width: int
    display_height: int
    #: One pixel's width to height, when the stream states one other than square.
    sample_aspect: str | None
    #: Counterclockwise degrees, as the stream's display matrix states them.
    rotation: Literal[0, 90, 180, 270]
    frame_rate: str | None
    frame_rate_form: FrameRateForm
    time_base: str | None
    #: How the stream states its pictures are stored, by ffmpeg's names; None
    #: when it does not say, and "unreadable" when what it says is not a name.
    pixel_format: str | None
    color_range: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    field_order: str | None


class AudioStreamFacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    codec: str
    channels: int | None
    sample_rate: int | None


class VideoProbe(BaseModel):
    """One stored video as the utilities see it, and what they may do with it."""

    model_config = ConfigDict(frozen=True)

    version: Literal[1] = PROBE_VERSION
    artifact_id: str
    artifact_sha256: str
    container: Container
    duration_seconds: float | None
    #: Where the video's own timestamps begin. Times a utility records are counted from here.
    start_seconds: float
    video: VideoStreamFacts
    audio: list[AudioStreamFacts]
    #: Further video, subtitle, data and attachment streams, which no utility keeps.
    omitted_streams: int
    can_save_frame: bool
    can_trim: bool
    #: Whether a trim can copy every audio stream unchanged into a file of the
    #: same kind; dropping audio is always possible.
    can_keep_audio: bool
    limits: list[LimitCode]
    #: Whether a part can be re-encoded to begin and end on exact frames.
    can_trim_exact: bool
    #: Whether such a cut can keep every audio stream, re-encoded as AAC.
    can_keep_audio_exact: bool
    exact_trim_limits: list[ExactTrimLimitCode]
    #: The ffprobe found on the system path, by its reported version and file digest.
    tool: dict[str, str]


async def probe_video(store: ArtifactStore, artifact: Artifact, ffprobe: MediaTool) -> VideoProbe:
    """Describe one stored video, or refuse it with a fixed reason.

    Its type and size are refused before anything is read; every other refusal
    follows a verified copy of the video and ffprobe's reading of that copy.
    """

    require_probe_input(artifact)
    async with store.verified_copy(artifact, maximum_bytes=MAX_INPUT_BYTES) as copy:
        return await probe_copy(copy, artifact, ffprobe)


async def probe_copy(copy: Path, artifact: Artifact, ffprobe: MediaTool) -> VideoProbe:
    """Describe a stored video from the copy ``ArtifactStore.verified_copy`` made of it.

    For a caller that runs several tools over one video and copies it once.
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
                "-show_entries",
                _ENTRIES,
                "-of",
                "json",
                f"file:{copy}",
            ],
            seconds=PROBE_SECONDS,
            stdout_limit=MAX_PROBE_BYTES,
            stderr_limit=MAX_ERROR_BYTES,
        )
    except TimeoutError:
        raise VideoProbeRefused("video-probe-timed-out") from None
    except ToolOutputTooLarge:
        raise VideoProbeRefused("video-probe-unreadable") from None
    except OSError as exc:
        # Found a moment ago, but it could not be started now: removed or
        # replaced meanwhile. The stored video is not at fault.
        code: ToolUnavailableCode = (
            "media-tool-missing" if isinstance(exc, FileNotFoundError) else "media-tool-unreadable"
        )
        raise MediaToolUnavailable(code) from exc
    if answer.returncode != 0:
        raise VideoProbeRefused("video-probe-unreadable")
    try:
        payload = json.loads(answer.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise VideoProbeRefused("video-probe-unreadable") from None
    return describe(payload, artifact, ffprobe)


def require_probe_input(artifact: Artifact) -> None:
    """Refuse, before anything is read or run, a stored file the probe would never read."""

    if not (artifact.media_type or "").casefold().startswith("video/"):
        raise VideoProbeRefused("video-probe-not-a-video")
    if artifact.size_bytes > MAX_INPUT_BYTES:
        raise VideoProbeRefused("video-probe-too-large")


def describe(payload: object, artifact: Artifact, ffprobe: MediaTool) -> VideoProbe:
    """Turn ffprobe's answer into the probe record, refusing anything malformed."""

    if not isinstance(payload, dict):
        raise VideoProbeRefused("video-probe-unreadable")
    container_format = payload.get("format")
    streams = payload.get("streams")
    if not isinstance(container_format, dict) or not isinstance(streams, list):
        raise VideoProbeRefused("video-probe-unreadable")
    container = _CONTAINERS.get(str(container_format.get("format_name")))
    if container is None:
        raise VideoProbeRefused("video-probe-unreadable")
    if len(streams) > MAX_STREAMS:
        raise VideoProbeRefused("video-probe-too-many-streams")
    entries = [stream for stream in streams if isinstance(stream, dict)]
    if len(entries) != len(streams):
        raise VideoProbeRefused("video-probe-unreadable")

    pictures = [
        stream
        for stream in entries
        if stream.get("codec_type") == "video" and not _attached_picture(stream)
    ]
    if not pictures:
        raise VideoProbeRefused("video-probe-no-video-stream")
    video = _video_facts(pictures[0])
    audio = [_audio_facts(stream) for stream in entries if stream.get("codec_type") == "audio"]
    omitted = len(entries) - 1 - len(audio)

    duration = _seconds(container_format.get("duration")) or _seconds(pictures[0].get("duration"))
    start = _offset(container_format.get("start_time"))
    if container == "matroska" and duration is not None and start > 0:
        # Matroska states where the video ends rather than how long it runs;
        # every time a utility takes is counted from the start.
        duration = duration - start if duration > start else None
    limits: list[LimitCode] = []
    if video.codec not in VIDEO_CODECS:
        limits.append("video-codec-unsupported")
    if duration is None:
        limits.append("video-duration-unknown")
    elif duration > MAX_DURATION_SECONDS:
        limits.append("video-duration-too-long")
    if max(video.width, video.height, video.display_width, video.display_height) > MAX_DIMENSION:
        limits.append("video-frame-too-large")
    if _rotation(pictures[0]) is None:
        limits.append("video-rotation-unsupported")
    can_save_frame = not limits
    can_keep_audio = all(stream.codec in COPY_AUDIO_CODECS[container] for stream in audio)
    if not can_keep_audio:
        limits.append("audio-codec-unsupported")
    exact_limits = exact_trim_limits(video, audio)

    return VideoProbe(
        artifact_id=artifact.id,
        artifact_sha256=artifact.sha256,
        container=container,
        duration_seconds=duration,
        start_seconds=start,
        video=video,
        audio=audio,
        omitted_streams=omitted,
        can_save_frame=can_save_frame,
        can_trim=can_save_frame,
        can_keep_audio=can_keep_audio,
        limits=limits,
        can_trim_exact=can_save_frame and not set(exact_limits) - {"exact-trim-audio-unsupported"},
        can_keep_audio_exact="exact-trim-audio-unsupported" not in exact_limits,
        exact_trim_limits=exact_limits,
        tool=ffprobe.record(),
    )


def exact_trim_limits(
    video: VideoStreamFacts, audio: list[AudioStreamFacts]
) -> list[ExactTrimLimitCode]:
    """Why a part of this video cannot be re-encoded to begin and end on exact frames.

    Only pictures an H.264 re-encode keeps looking the same are offered: 8-bit
    4:2:0 in limited range with standard-range colours, whole frames, an even
    frame size within bounds, and no turn, which the re-encode would not keep.
    """

    limits: list[ExactTrimLimitCode] = []
    if (
        video.pixel_format != EXACT_PIXEL_FORMAT
        or not _unset_or_in(video.color_range, _EXACT_RANGES)
        or not _unset_or_in(video.color_transfer, _EXACT_TRANSFERS)
        or not _unset_or_in(video.color_primaries, _EXACT_PRIMARIES)
        or not _unset_or_in(video.color_space, _EXACT_SPACES)
        or not _unset_or_in(video.field_order, _EXACT_FIELD_ORDERS)
    ):
        limits.append("exact-trim-picture-unsupported")
    if (
        video.width % 2
        or video.height % 2
        or max(video.width, video.height) > EXACT_MAX_SIDE
        or video.width * video.height > EXACT_MAX_PIXELS
    ):
        limits.append("exact-trim-frame-size-unsupported")
    if video.rotation != 0:
        limits.append("exact-trim-turned-unsupported")
    if not all(
        stream.codec in EXACT_AUDIO_CODECS
        and stream.channels in (1, 2)
        and stream.sample_rate in EXACT_SAMPLE_RATES
        for stream in audio
    ):
        limits.append("exact-trim-audio-unsupported")
    return limits


def _unset_or_in(value: str | None, allowed: frozenset[str]) -> bool:
    return value is None or value in allowed


def container_named(format_name: object) -> Container | None:
    """The kind of file ffprobe's format name describes, when it is one the utilities read."""

    return _CONTAINERS.get(str(format_name))


def stream_facts(stream: dict[str, Any]) -> VideoStreamFacts | AudioStreamFacts | None:
    """One stream of ffprobe's answer as the probe reads it, or None when no utility keeps it."""

    if stream.get("codec_type") == "video" and not _attached_picture(stream):
        return _video_facts(stream)
    if stream.get("codec_type") == "audio":
        return _audio_facts(stream)
    return None


def _video_facts(stream: dict[str, Any]) -> VideoStreamFacts:
    width = _count(stream.get("width"))
    height = _count(stream.get("height"))
    if width is None or height is None:
        raise VideoProbeRefused("video-probe-unreadable")
    aspect = _ratio(stream.get("sample_aspect_ratio"))
    if aspect is not None and aspect != 1:
        shown_width = max(1, round(width * aspect))
        sample_aspect: str | None = f"{aspect.numerator}:{aspect.denominator}"
    else:
        shown_width = width
        sample_aspect = None
    stated = _rotation(stream)
    # A turn that is not a right angle is named among the limits; it is shown unturned.
    rotation: Literal[0, 90, 180, 270] = 0 if stated is None else stated
    display = (height, shown_width) if rotation in (90, 270) else (shown_width, height)
    nominal = _ratio(stream.get("r_frame_rate"))
    average = _ratio(stream.get("avg_frame_rate"))
    rate = average or nominal
    if nominal is None or average is None:
        form: FrameRateForm = "unknown"
    else:
        form = "constant" if nominal == average else "variable"
    time_base = _ratio(stream.get("time_base"))
    return VideoStreamFacts(
        index=_count(stream.get("index"), minimum=0) or 0,
        codec=_codec(stream.get("codec_name")),
        width=width,
        height=height,
        display_width=display[0],
        display_height=display[1],
        sample_aspect=sample_aspect,
        rotation=rotation,
        frame_rate=None if rate is None else f"{rate.numerator}/{rate.denominator}",
        frame_rate_form=form,
        time_base=None if time_base is None else f"{time_base.numerator}/{time_base.denominator}",
        pixel_format=_property(stream.get("pix_fmt")),
        color_range=_property(stream.get("color_range")),
        color_space=_property(stream.get("color_space")),
        color_transfer=_property(stream.get("color_transfer")),
        color_primaries=_property(stream.get("color_primaries")),
        field_order=_property(stream.get("field_order")),
    )


def _audio_facts(stream: dict[str, Any]) -> AudioStreamFacts:
    return AudioStreamFacts(
        index=_count(stream.get("index"), minimum=0) or 0,
        codec=_codec(stream.get("codec_name")),
        channels=_count(stream.get("channels")),
        sample_rate=_count(stream.get("sample_rate")),
    )


def _attached_picture(stream: dict[str, Any]) -> bool:
    disposition = stream.get("disposition")
    return isinstance(disposition, dict) and disposition.get("attached_pic") == 1


def _rotation(stream: dict[str, Any]) -> Literal[0, 90, 180, 270] | None:
    """The stream's rotation in counterclockwise degrees, or None when not a right angle."""

    stated: object = None
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for entry in side_data:
            if isinstance(entry, dict) and "rotation" in entry:
                stated = entry["rotation"]
                break
    if stated is None:
        tags = stream.get("tags")
        if isinstance(tags, dict):
            stated = tags.get("rotate")
    if stated is None:
        return 0
    try:
        degrees = float(str(stated))
    except ValueError:
        return None
    if not degrees.is_integer():
        return None
    normalized = int(degrees) % 360
    if normalized == 0:
        return 0
    if normalized == 90:
        return 90
    if normalized == 180:
        return 180
    if normalized == 270:
        return 270
    return None


def _codec(value: object) -> str:
    text = str(value) if value is not None else ""
    return text if _CODEC_NAME.fullmatch(text) else "unknown"


def _property(value: object) -> str | None:
    """A picture property ffprobe states by name: None when it states none or calls it unknown.

    Anything else that is not a short lowercase name is kept as "unreadable",
    so it can never pass for a property left unstated.
    """

    if value is None or value == "unknown":
        return None
    text = value if isinstance(value, str) else ""
    return text if _PROPERTY_NAME.fullmatch(text) else "unreadable"


def _count(value: object, *, minimum: int = 1) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.isdigit() and len(value) <= 10:
        number = int(value)
    else:
        return None
    return number if number >= minimum else None


def _ratio(value: object) -> Fraction | None:
    match = _RATIO.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return None
    numerator, denominator = int(match.group(1)), int(match.group(2))
    if numerator == 0 or denominator == 0:
        return None
    return Fraction(numerator, denominator)


def _offset(value: object) -> float:
    """A start time, which may be zero or slightly negative; anything unreadable counts as zero."""

    if not isinstance(value, str) or len(value) > 32:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        return 0.0
    return seconds if abs(seconds) <= MAX_DURATION_SECONDS else 0.0


def _seconds(value: object) -> float | None:
    if not isinstance(value, str) or len(value) > 32:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if seconds != seconds or seconds <= 0 or seconds == float("inf"):
        return None
    return seconds
