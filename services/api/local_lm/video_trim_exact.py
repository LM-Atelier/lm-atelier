"""Part of a stored video re-encoded so that it begins and ends on the exact frames asked for.

THE FRAMES. A part from a start to an end holds every frame on screen during
it: the frame shown at the start, then each later frame that begins before the
end. Times are on the probe's timeline and the dialog works in milliseconds, so
a frame within half a millisecond of either time counts as at it. A start
before the video's first picture starts on that picture.

HOW IT IS MADE. The plan lists the picture stream's packets by their own
timestamps, in whole ticks of the stream's time base, and names the frames by
those ticks. ffmpeg decodes from the keyframe before the one the part begins
after, keeps exactly those frames by their timestamps, and re-encodes them as
H.264 with each frame's own timing; the kept sound is cut at the same times and
re-encoded as AAC. The same run writes a digest of every frame the encoder
received, so what was encoded is measured, not predicted.

WHAT IS CHECKED BEFORE KEEPING. The encoder received exactly the planned
frames. The new file holds one H.264 picture stream of the source's size and
pixel shape, unturned, then the kept sound as AAC, and nothing else; it stores
exactly as many frames, the first a keyframe at zero, spaced as they were in
the original. Decoding the original again from the part's own keyframe gives
the same frames, bit for bit, as the decode from the keyframe before, so none of
them leans on a picture the decode began without; where each decode began is
read from its first keyframe, not assumed from where it was sought. A part in
the first group of pictures has nothing before it and is decoded from the very
beginning both times. Compared frame by frame with the original's, no frame is
below a fixed PSNR and the mean SSIM is above a fixed floor. The picture is
close to the original, never claimed identical.

FINDING KEYFRAMES. A keyframe is looked up by seeking to a time, and an MP4 is
sought by decode time: a time just before an open keyframe, whose leading
pictures are shown before it but stored after it, can still land on that
keyframe. So while a lookup lands later than wanted, the time steps back,
twice as far each time, within the lead an exact cut may decode. ffmpeg is then
sought to the very times the lookups used, so its decodes land where they did.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from .artifacts import ToolOutput
from .media_process import ToolOutputTooLarge, run_tool
from .media_tools import MediaTool
from .models import Artifact
from .video_probe import (
    DEMUXERS,
    MAX_ERROR_BYTES,
    MAX_INPUT_BYTES,
    AudioStreamFacts,
    VideoProbe,
    VideoProbeRefused,
    VideoStreamFacts,
    container_named,
    stream_facts,
)
from .video_trim import (
    CUT_ROOM,
    CUT_SECONDS,
    END_SECONDS,
    KEYFRAME_SECONDS,
    MATROSKA_STEP_BACK,
    MAX_END_BYTES,
    MAX_KEYFRAME_BYTES,
    MIN_TRIM_SECONDS,
    SEEK_PAST,
    TrimRefused,
    VideoTrimPreview,
    has_b_frames,
    keyframe_at,
    listed_fields,
    measure_output,
    packet_count,
    stated_time,
    tool_unavailable,
)

#: The most frames, and the most pixels over all of them, one exact cut re-encodes.
MAX_EXACT_FRAMES: Final = 7200
MAX_EXACT_PIXELS: Final = 1920 * 1080 * 30 * 120
#: How far before the part's first frame decoding may have to begin.
MAX_LEAD_SECONDS: Final = 60.0
#: How far past the end the frame listing reads, so pictures stored out of order are all listed.
LISTING_MARGIN: Final = 2.0
#: Closer than this, two frames could not be told apart by a time to the millisecond.
MIN_FRAME_GAP: Final = 0.001
MIN_PSNR_DB: Final = 35.0
MIN_SSIM: Final = 0.95
VIDEO_ENCODER: Final = "libx264"
AUDIO_ENCODER: Final = "aac"
PRESET: Final = "medium"
CRF: Final = 18
COMPARE_SECONDS: Final = 600.0
#: A frame digest line is about a hundred bytes; this holds every frame's twice over.
MAX_DIGEST_BYTES: Final = 2 * 1024 * 1024
MAX_COMPARE_REPORT_BYTES: Final = 1024 * 1024
#: Sound bits per second for one and two channels.
_AUDIO_BITRATES: Final = {1: 128_000, 2: 192_000}

_PACKET_ROW: Final = re.compile(r"(-?\d{1,18}),(-?\d{1,18}|N/A),([A-Z_]{1,8})")
_DIGEST_ROW: Final = re.compile(
    r"([01]),\s*(-?\d{1,18}),\s*(-?\d{1,18}),\s*(-?\d{1,18}),\s*(\d{1,12}),\s*([0-9a-f]{64})"
)
_DIGEST_TIME_BASE: Final = re.compile(r"#tb ([01]): (\d{1,10})/(\d{1,10})")
#: Two lookups or reports of one keyframe agree within this many seconds.
_SAME_TICK: Final = 2e-6
#: Transfer names ffmpeg's encoder options take for the two whose stream names they do not.
_ENCODER_TRANSFERS: Final = {"bt470m": "gamma22", "bt470bg": "gamma28"}
_PSNR: Final = re.compile(
    r"^\[psnr@aligned @ [0-9a-fA-Fx]+\] PSNR .*?average:(inf|\d{1,4}\.\d{1,6}) "
    r"min:(inf|\d{1,4}\.\d{1,6}) max:(?:inf|\d{1,4}\.\d{1,6})\s*$",
    re.M,
)
_SSIM: Final = re.compile(
    r"^\[ssim@aligned @ [0-9a-fA-Fx]+\] SSIM .*?All:(\d\.\d{1,6}) "
    r"\((?:inf|-?\d{1,4}\.\d{1,6})\)\s*$",
    re.M,
)


@dataclass(frozen=True)
class ExactTrimPlan:
    preview: VideoTrimPreview
    #: The probe's start of the video's timestamps, which every absolute time here includes.
    timeline_start: float
    video: VideoStreamFacts
    audio: tuple[AudioStreamFacts, ...]
    time_base: Fraction
    #: The part's frames, by their timestamps in ticks of ``time_base``, in the order shown.
    frames: tuple[int, ...]
    #: Where the part ends, in ticks: the next frame's time, or the last frame's end.
    end_ticks: int
    #: ffmpeg's ``-ss`` for the decode the encode reads and for the one it is checked
    #: against, counted from the start of the file's timestamps as ``-ss`` is;
    #: None decodes from the very beginning.
    decode_seek: float | None
    reference_seek: float | None
    output_bytes: int

    def absolute(self, ticks: int) -> float:
        return float(ticks * self.time_base)


@dataclass(frozen=True)
class FedFrame:
    """One frame the encoder received: its timestamp in ticks and its decoded digest."""

    ticks: int
    sha256: str


@dataclass(frozen=True)
class Decoded:
    """The frames one decode of the original gave, and the keyframe it began with, in ticks."""

    frames: list[FedFrame]
    began: int | None


@dataclass(frozen=True)
class ExactMeasurement:
    """What the new video holds, read from the new file itself."""

    video_codec: str
    video_start_seconds: float | None
    #: Where its picture ends: the last frame's end, never past the length the file states.
    video_end_seconds: float
    frames_stored: int
    audio: tuple[tuple[str, float | None, int], ...]


@dataclass(frozen=True)
class Closeness:
    """How close the new video's frames are to the original's, frame by frame."""

    ssim_mean: float
    #: None when every frame decoded identical, which PSNR states as infinite.
    psnr_mean_db: float | None
    psnr_lowest_db: float | None


async def plan_exact_trim(
    copy: Path,
    artifact: Artifact,
    probe: VideoProbe,
    ffprobe: MediaTool,
    ffmpeg: MediaTool,
    requested_start: float,
    requested_end: float,
    keep_audio: bool,
) -> ExactTrimPlan:
    """Which frames a cut on exact frames would hold, or why it cannot be made.

    ``copy`` is the file ``ArtifactStore.verified_copy`` made of the stored video.
    """

    duration = probe.duration_seconds
    if probe.artifact_sha256 != artifact.sha256 or not probe.can_trim_exact or duration is None:
        raise TrimRefused("video-trim-exact-not-offered")
    if keep_audio and probe.audio and not probe.can_keep_audio_exact:
        raise TrimRefused("video-trim-exact-audio-not-offered")
    time_base = _time_base(probe.video.time_base)
    kept = tuple(probe.audio) if keep_audio else ()
    start = round(requested_start, 6)
    end = round(requested_end, 6)
    if not (start >= 0 and end <= round(duration, 6) and end - start >= MIN_TRIM_SECONDS - 1e-9):
        raise TrimRefused("video-trim-range-invalid")
    await require_encoders(ffmpeg, keep_sound=bool(kept))

    origin = probe.start_seconds
    # Two of the stream's ticks, so a time is never rounded back onto the
    # keyframe it is meant to come before: Matroska counts in milliseconds.
    back = max(SEEK_PAST, 2 * float(time_base))
    first_keyframe = await keyframe_at(copy, probe, ffprobe, max(origin, 0.0))
    target = max(origin + start + SEEK_PAST, 0.0)
    if target < first_keyframe - _SAME_TICK:
        # Before the first keyframe shown; a stream copy can leave pictures
        # ahead of it, decoded from one its edit list hides.
        keyframe, asked = first_keyframe, target
    else:
        keyframe, asked = await _keyframe_at_or_before(
            copy, probe, ffprobe, at=target, limit=target, back=back, gap=0.0
        )
    at_first_keyframe = keyframe <= first_keyframe + _SAME_TICK
    step = (
        MATROSKA_STEP_BACK
        if probe.container == "matroska" and await has_b_frames(copy, ffprobe)
        else 0.0
    )
    if at_first_keyframe:
        decode_from = first_keyframe
        decode_seek: float | None = None
        reference_seek: float | None = None
    else:
        # Decoded from the keyframe before, so the check against a decode from
        # the part's own keyframe compares two different beginnings.
        decode_from, before = await _keyframe_at_or_before(
            copy, probe, ffprobe, at=keyframe, limit=keyframe - _SAME_TICK, back=back, gap=back
        )
        # ffmpeg's ``-ss`` counts from where the file's timestamps begin.
        decode_seek = round(max(before - origin, 0.0) + step, 6)
        reference_seek = round(max(asked - origin, 0.0) + step, 6)

    # The first group of pictures is listed from the very beginning, so
    # pictures shown before its first keyframe are listed too.
    listed = await _listing(
        copy,
        probe,
        ffprobe,
        None if at_first_keyframe else asked,
        origin + end + LISTING_MARGIN,
    )
    rows = frame_rows(listed)
    chosen = choose_frames(rows, time_base, origin + start, origin + end)
    if not at_first_keyframe and chosen.began_at_first_picture:
        # Only the video's own first keyframe may come after the start asked for.
        raise TrimRefused("video-trim-frames-unknown")
    frames = chosen.frames
    if (
        len(frames) > MAX_EXACT_FRAMES
        or len(frames) * probe.video.width * probe.video.height > MAX_EXACT_PIXELS
        or float(frames[0] * time_base) - decode_from > MAX_LEAD_SECONDS
    ):
        raise TrimRefused("video-trim-exact-too-long")

    first_seconds = round(float(frames[0] * time_base) - origin, 6)
    end_seconds = round(float(chosen.end_ticks * time_base) - origin, 6)
    preview = VideoTrimPreview(
        mode="exact",
        artifact_id=artifact.id,
        artifact_sha256=artifact.sha256,
        requested_start_seconds=start,
        requested_end_seconds=end,
        keep_audio=keep_audio,
        start_seconds=first_seconds,
        keyframe_seconds=round(keyframe - origin, 6),
        from_beginning=False,
        last_frame_seconds=round(float(frames[-1] * time_base) - origin, 6),
        end_seconds=end_seconds,
        frame_count=len(frames),
        began_at_first_picture=chosen.began_at_first_picture,
        keeps_whole_video=(
            at_first_keyframe
            and chosen.from_first_listed
            and chosen.next_ticks is None
            and end >= duration - SEEK_PAST
        ),
        audio_streams_kept=len(kept),
        omitted_streams=probe.omitted_streams,
        format="mp4",
        media_type="video/mp4",
    )
    pixels = len(frames) * probe.video.width * probe.video.height
    sound = sum(_audio_bitrate(stream) for stream in kept) * (end_seconds - first_seconds + 1) / 8
    return ExactTrimPlan(
        preview=preview,
        timeline_start=origin,
        video=probe.video,
        audio=kept,
        time_base=time_base,
        frames=frames,
        end_ticks=chosen.end_ticks,
        decode_seek=decode_seek,
        reference_seek=reference_seek,
        # Two bits a pixel is more than an H.264 encode at this quality takes.
        output_bytes=min(MAX_INPUT_BYTES, math.ceil(pixels / 4 + sound) + CUT_ROOM),
    )


async def _keyframe_at_or_before(
    copy: Path,
    probe: VideoProbe,
    ffprobe: MediaTool,
    *,
    at: float,
    limit: float,
    back: float,
    gap: float,
) -> tuple[float, float]:
    """The keyframe a lookup finds at or before ``limit``, and the time the lookup used.

    The first lookup is ``gap`` before ``at``; while one lands past ``limit``,
    the next steps back twice as far, never before the file's start nor
    further than an exact cut may decode ahead of its first frame.
    """

    floor = max(probe.start_seconds, 0.0)
    while True:
        asked = max(at - gap, floor)
        found = await keyframe_at(copy, probe, ffprobe, asked)
        if found <= limit:
            return found, asked
        if asked <= floor or gap >= MAX_LEAD_SECONDS:
            raise TrimRefused("video-trim-keyframe-unknown")
        gap = back if gap <= 0 else gap * 2


async def require_encoders(ffmpeg: MediaTool, *, keep_sound: bool) -> None:
    """Refuse unless this ffmpeg can make H.264 video, and AAC sound when sound is kept."""

    for encoder in (VIDEO_ENCODER, AUDIO_ENCODER) if keep_sound else (VIDEO_ENCODER,):
        try:
            answer = await run_tool(
                ffmpeg.executable,
                ["-hide_banner", "-nostdin", "-loglevel", "error", "-h", f"encoder={encoder}"],
                seconds=KEYFRAME_SECONDS,
                stdout_limit=MAX_KEYFRAME_BYTES,
                stderr_limit=MAX_ERROR_BYTES,
            )
        except TimeoutError:
            raise TrimRefused("video-trim-timed-out") from None
        except ToolOutputTooLarge:
            raise TrimRefused("video-trim-encoder-missing") from None
        except OSError as exc:
            raise tool_unavailable(exc) from exc
        if answer.returncode != 0 or not encoder_offered(answer.stdout, encoder):
            raise TrimRefused("video-trim-encoder-missing")


def encoder_offered(content: bytes, encoder: str) -> bool:
    """Whether ffmpeg's help for one encoder describes it, rather than saying it has none."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and lines[0].startswith(f"Encoder {encoder} [")


@dataclass(frozen=True)
class FrameRow:
    ticks: int
    #: The packet's stated length in ticks, when it states one.
    duration: int | None
    keyframe: bool


@dataclass(frozen=True)
class ChosenFrames:
    frames: tuple[int, ...]
    #: The first frame after the part, when the listing reached one.
    next_ticks: int | None
    end_ticks: int
    began_at_first_picture: bool
    #: Whether the part begins with the first frame listed.
    from_first_listed: bool


def frame_rows(content: bytes) -> list[FrameRow]:
    """ffprobe's packet listing as frames in the order shown, refusing anything unclear.

    Packets an edit list marks to be discarded are not shown, so they are left
    out; a packet marked corrupt refuses the listing.
    """

    try:
        text = content.decode("ascii")
    except UnicodeDecodeError:
        raise TrimRefused("video-trim-frames-unknown") from None
    rows: list[FrameRow] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _PACKET_ROW.fullmatch(",".join(listed_fields(line, 3)))
        if match is None:
            raise TrimRefused("video-trim-frames-unknown")
        ticks, length, flags = match.groups()
        if "C" in flags:
            raise TrimRefused("video-trim-frames-unknown")
        if "D" in flags:
            continue
        rows.append(
            FrameRow(
                ticks=int(ticks),
                duration=None if length == "N/A" else int(length),
                keyframe="K" in flags,
            )
        )
    if not rows:
        raise TrimRefused("video-trim-frames-unknown")
    rows.sort(key=lambda row: row.ticks)
    if any(earlier.ticks == later.ticks for earlier, later in zip(rows, rows[1:], strict=False)):
        raise TrimRefused("video-trim-frames-unknown")
    return rows


def choose_frames(
    rows: list[FrameRow], time_base: Fraction, start: float, end: float
) -> ChosenFrames:
    """The frames on screen from ``start`` to ``end``, both absolute, by the rule above."""

    def seconds(row: FrameRow) -> float:
        return float(row.ticks * time_base)

    if any(
        float((later.ticks - earlier.ticks) * time_base) < MIN_FRAME_GAP
        for earlier, later in zip(rows, rows[1:], strict=False)
    ):
        raise TrimRefused("video-trim-frames-unknown")
    at_start = [index for index, row in enumerate(rows) if seconds(row) <= start + SEEK_PAST]
    began_at_first_picture = not at_start
    first = at_start[-1] if at_start else 0
    before_end = [index for index, row in enumerate(rows) if seconds(row) < end - SEEK_PAST]
    if not before_end or before_end[-1] < first:
        raise TrimRefused("video-trim-range-before-picture")
    last = before_end[-1]
    following = rows[last + 1] if last + 1 < len(rows) else None
    if following is not None:
        end_ticks = following.ticks
    else:
        length = rows[last].duration
        if length is None or length <= 0:
            raise TrimRefused("video-trim-frames-unknown")
        end_ticks = rows[last].ticks + length
    return ChosenFrames(
        frames=tuple(row.ticks for row in rows[first : last + 1]),
        next_ticks=None if following is None else following.ticks,
        end_ticks=end_ticks,
        began_at_first_picture=began_at_first_picture,
        from_first_listed=first == 0,
    )


async def encode_part(
    copy: Path, plan: ExactTrimPlan, ffmpeg: MediaTool, output: ToolOutput
) -> Decoded:
    """Re-encode the planned frames into the file the store made, and return what was encoded.

    The same run states the keyframe its decode began with.
    """

    rate = f"{plan.time_base.numerator}:{plan.time_base.denominator}"
    first = plan.absolute(plan.frames[0])
    end = plan.absolute(plan.end_ticks)
    graph = (
        f"[0:{plan.video.index}]{_began_with()};"
        f"[part]{_frame_trim(plan)},split[e][h];[e]setpts=PTS-STARTPTS[v]"
    )
    sound: list[str] = []
    for number, stream in enumerate(plan.audio):
        graph += (
            f";[0:{stream.index}]atrim=start={first:.6f}:end={end:.6f},"
            f"asetpts=PTS-{first:.6f}/TB[a{number}]"
        )
        sound += ["-map", f"[a{number}]"]
    arguments = [*_input(copy, plan.decode_seek), "-filter_complex", graph, "-map", "[v]", *sound]
    arguments += [
        "-fps_mode:v",
        "passthrough",
        "-enc_time_base:v",
        rate,
        "-c:v",
        VIDEO_ENCODER,
        "-preset",
        PRESET,
        "-crf",
        str(CRF),
        "-pix_fmt",
        "yuv420p",
        *_colours(plan.video),
    ]
    for number, kept in enumerate(plan.audio):
        # Each sound track keeps its own rate and channels.
        arguments += [
            f"-c:a:{number}",
            AUDIO_ENCODER,
            f"-b:a:{number}",
            str(_audio_bitrate(kept)),
            f"-ar:a:{number}",
            str(kept.sample_rate),
            f"-ac:a:{number}",
            str(kept.channels),
        ]
    arguments += [
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-fs",
        str(output.maximum_bytes),
        "-movflags",
        "+faststart",
        "-write_tmcd",
        "0",
        "-video_track_timescale",
        str(plan.time_base.denominator),
        # The kind is named, so the private file's name never picks it; the
        # file exists already, made by the store, so it is overwritten.
        "-f",
        "mp4",
        "-y",
        f"file:{output.path}",
        # A second output in the same run: the digest of every frame the
        # encoder received, and of the keyframe the decode began with.
        "-map",
        "[h]",
        "-map",
        "[k]",
        "-fps_mode:v",
        "passthrough",
        "-enc_time_base:v",
        rate,
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
            seconds=CUT_SECONDS,
            stdout_limit=MAX_DIGEST_BYTES,
            stderr_limit=MAX_ERROR_BYTES,
        )
    except TimeoutError:
        raise TrimRefused("video-trim-timed-out") from None
    except ToolOutputTooLarge:
        raise TrimRefused("video-trim-not-encoded") from None
    except OSError as exc:
        raise tool_unavailable(exc) from exc
    if answer.returncode != 0:
        raise TrimRefused("video-trim-not-encoded")
    fed = frame_digests(answer.stdout, plan.time_base)
    if [frame.ticks for frame in fed.frames] != list(plan.frames) or fed.began is None:
        raise TrimRefused("video-trim-frames-unverified")
    return fed


def frame_digests(content: bytes, time_base: Fraction) -> Decoded:
    """The frames in ffmpeg's framehash output, and the keyframe its decode began with.

    Stream 0 holds the frames and stream 1, when there, the first keyframe;
    both must be stated in the source's time base.
    """

    try:
        text = content.decode("ascii")
    except UnicodeDecodeError:
        raise TrimRefused("video-trim-frames-unverified") from None
    stated = _DIGEST_TIME_BASE.findall(text)
    if not stated or any(Fraction(int(n), int(d)) != time_base for _index, n, d in stated):
        raise TrimRefused("video-trim-frames-unverified")
    frames: list[FedFrame] = []
    began: list[int] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = _DIGEST_ROW.fullmatch(line.strip())
        if match is None:
            raise TrimRefused("video-trim-frames-unverified")
        if match.group(1) == "0":
            frames.append(FedFrame(ticks=int(match.group(3)), sha256=match.group(6)))
        else:
            began.append(int(match.group(3)))
    if len(began) > 1:
        raise TrimRefused("video-trim-frames-unverified")
    return Decoded(frames=frames, began=began[0] if began else None)


async def check_exact_trim(
    output: Path, plan: ExactTrimPlan, ffprobe: MediaTool
) -> ExactMeasurement:
    """Measure the new video, and refuse it unless it holds the planned frames as planned."""

    facts = await measure_output(output, ffprobe)
    streams = facts.get("streams")
    if not isinstance(streams, list) or not all(isinstance(item, dict) for item in streams):
        raise TrimRefused("video-trim-unmeasured")
    if container_named(facts["format"].get("format_name")) != "mp4" or len(streams) != 1 + len(
        plan.audio
    ):
        raise TrimRefused("video-trim-output-mismatch")
    try:
        video = stream_facts(streams[0])
        kept = [stream_facts(stream) for stream in streams[1:]]
    except VideoProbeRefused:
        raise TrimRefused("video-trim-output-mismatch") from None
    if not isinstance(video, VideoStreamFacts) or not _same_picture(video, plan.video):
        raise TrimRefused("video-trim-output-mismatch")
    audio: list[tuple[str, float | None, int]] = []
    longest = float(plan.end_ticks - plan.frames[0]) * float(plan.time_base)
    for planned, stream, found in zip(plan.audio, streams[1:], kept, strict=True):
        if (
            not isinstance(found, AudioStreamFacts)
            or found.codec != AUDIO_ENCODER
            or (found.channels, found.sample_rate) != (planned.channels, planned.sample_rate)
        ):
            raise TrimRefused("video-trim-output-mismatch")
        begins = stated_time(stream.get("start_time"))
        lasts = stated_time(stream.get("duration"))
        rate = planned.sample_rate or 1
        if begins is None or (lasts is not None and lasts > longest + 1024 / rate + 0.001):
            raise TrimRefused("video-trim-output-mismatch")
        audio.append((found.codec, begins, packet_count(stream)))

    stored = await _output_frames(output, ffprobe)
    out_base = _time_base(video.time_base)
    if (
        len(stored) != len(plan.frames)
        or not stored[0].keyframe
        or [row.ticks * out_base for row in stored]
        != [(ticks - plan.frames[0]) * plan.time_base for ticks in plan.frames]
    ):
        raise TrimRefused("video-trim-frames-unverified")
    last = stored[-1]
    if last.duration:
        ends = float((last.ticks + last.duration) * out_base)
        stated = stated_time(streams[0].get("duration"))
        if stated is not None and stated > 0:
            ends = min(ends, stated)
    else:
        # Some ffmpeg builds, 6.1 among them, store no length for a re-encoded
        # file's last frame. Its frames are the planned ones, checked above to
        # the tick, so the last one lasts until the source's next frame began.
        ends = longest
    return ExactMeasurement(
        video_codec=video.codec,
        video_start_seconds=stated_time(streams[0].get("start_time")),
        video_end_seconds=ends,
        frames_stored=packet_count(streams[0]),
        audio=tuple(audio),
    )


async def compare_with_source(
    copy: Path, output: Path, plan: ExactTrimPlan, ffmpeg: MediaTool, fed: Decoded
) -> tuple[Closeness, int]:
    """Decode the original again from the part's own keyframe and compare the new video with it.

    The decode must give the frames the encoder received, bit for bit, and
    must have begun with a later keyframe than the encode's decode, unless the
    part is in the first group of pictures; then each new frame is compared
    with the original frame it was made from. Returns the closeness and the
    keyframe this decode began with, in ticks.
    """

    rate = f"{plan.time_base.numerator}:{plan.time_base.denominator}"
    graph = (
        f"[1:{plan.video.index}]{_began_with()};"
        f"[part]{_frame_trim(plan)},split[h][r];"
        "[r]setpts=PTS-STARTPTS,split[r1][r2];"
        "[0:0]setpts=PTS-STARTPTS,split[o1][o2];"
        "[o1][r1]psnr@aligned=shortest=1[x];"
        "[o2][r2]ssim@aligned=shortest=1[y]"
    )
    arguments = [
        "-hide_banner",
        "-nostdin",
        "-nostats",
        # The two comparisons state their results at this level.
        "-loglevel",
        "info",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        DEMUXERS,
        "-noautorotate",
        "-i",
        f"file:{output}",
        *_input(copy, plan.reference_seek, banner=False),
        "-filter_complex",
        graph,
        "-map",
        "[h]",
        "-map",
        "[k]",
        "-fps_mode:v",
        "passthrough",
        "-enc_time_base:v",
        rate,
        "-f",
        "framehash",
        "-hash",
        "sha256",
        "pipe:1",
        "-map",
        "[x]",
        "-f",
        "null",
        "-",
        "-map",
        "[y]",
        "-f",
        "null",
        "-",
    ]
    try:
        answer = await run_tool(
            ffmpeg.executable,
            arguments,
            seconds=COMPARE_SECONDS,
            stdout_limit=MAX_DIGEST_BYTES,
            stderr_limit=MAX_COMPARE_REPORT_BYTES,
        )
    except TimeoutError:
        raise TrimRefused("video-trim-timed-out") from None
    except ToolOutputTooLarge:
        raise TrimRefused("video-trim-unmeasured") from None
    except OSError as exc:
        raise tool_unavailable(exc) from exc
    if answer.returncode != 0:
        raise TrimRefused("video-trim-unmeasured")
    reference = frame_digests(answer.stdout, plan.time_base)
    if reference.began is None or fed.began is None:
        raise TrimRefused("video-trim-frames-unverified")
    if plan.reference_seek is not None and reference.began <= fed.began:
        # Both decodes began with the same keyframe, so comparing them proves nothing.
        raise TrimRefused("video-trim-start-unchecked")
    if reference.frames != fed.frames:
        raise TrimRefused("video-trim-start-incomplete")
    closeness = closeness_stated(answer.stderr)
    lowest = closeness.psnr_lowest_db
    if closeness.ssim_mean < MIN_SSIM or (lowest is not None and lowest < MIN_PSNR_DB):
        raise TrimRefused("video-trim-reencode-unlike")
    return closeness, reference.began


def closeness_stated(content: bytes) -> Closeness:
    """The one PSNR and one SSIM summary in ffmpeg's report, or a refusal."""

    text = content.decode("utf-8", errors="replace")
    psnr = _PSNR.findall(text)
    ssim = _SSIM.findall(text)
    if len(psnr) != 1 or len(ssim) != 1:
        raise TrimRefused("video-trim-unmeasured")
    mean, lowest = psnr[0]
    return Closeness(
        ssim_mean=float(ssim[0]),
        psnr_mean_db=None if mean == "inf" else float(mean),
        psnr_lowest_db=None if lowest == "inf" else float(lowest),
    )


def exact_trim_record(
    source: Artifact,
    plan: ExactTrimPlan,
    measured: ExactMeasurement,
    closeness: Closeness,
    fed: Decoded,
    checked_from: int,
    tool: dict[str, str],
    measured_with: dict[str, str],
) -> dict[str, Any]:
    """What was asked for, which frames were kept, and how close the new video measured.

    The requested and actual start and end are on the original's timeline; the
    ``video`` and ``audio`` entries are measured on the new file's own. Only
    ``source_artifact_id`` names another artifact, so the record keeps the
    source and nothing else.
    """

    preview = plan.preview
    first = preview.start_seconds
    digest = _digest_of("\n".join(frame.sha256 for frame in fed.frames))
    began = fed.began if fed.began is not None else plan.frames[0]
    return {
        "action": "trim",
        "mode": "exact",
        "source_artifact_id": source.id,
        "source_sha256": source.sha256,
        "requested_start_seconds": preview.requested_start_seconds,
        "requested_end_seconds": preview.requested_end_seconds,
        "actual_start_seconds": first,
        "keyframe_seconds": preview.keyframe_seconds,
        "actual_end_seconds": round(first + measured.video_end_seconds, 6),
        "from_beginning": False,
        "keep_audio": preview.keep_audio,
        "format": "mp4",
        "media_type": "video/mp4",
        "video": {
            "codec": measured.video_codec,
            "start_seconds": _rounded(measured.video_start_seconds),
            "first_shown_seconds": 0.0,
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
        "frames": {
            "first_seconds": first,
            "last_seconds": preview.last_frame_seconds,
            "count": len(plan.frames),
            "began_at_first_picture": preview.began_at_first_picture,
            "source_frames_sha256": digest,
            # Where the encode's decode and the check's began, as each decode stated.
            "decoded_from_seconds": round(plan.absolute(began) - plan.timeline_start, 6),
            "checked_from_seconds": round(plan.absolute(checked_from) - plan.timeline_start, 6),
        },
        "quality": {
            "ssim_mean": round(closeness.ssim_mean, 6),
            "psnr_mean_db": _rounded(closeness.psnr_mean_db),
            "psnr_lowest_db": _rounded(closeness.psnr_lowest_db),
        },
        "encoding": {
            "video_encoder": VIDEO_ENCODER,
            "preset": PRESET,
            "crf": CRF,
            "pixel_format": "yuv420p",
            "audio_encoder": AUDIO_ENCODER if plan.audio else None,
            "audio_bitrates": [_audio_bitrate(stream) for stream in plan.audio],
        },
        "tool": tool,
        "measured_with": measured_with,
    }


async def _listing(
    copy: Path, probe: VideoProbe, ffprobe: MediaTool, begin: float | None, end: float
) -> bytes:
    """The picture stream's packets from where ``begin`` is sought, or the start, past ``end``."""

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
                str(probe.video.index),
                "-read_intervals",
                f"%{end:.6f}" if begin is None else f"{max(begin, 0.0):.6f}%{end:.6f}",
                "-show_entries",
                "packet=pts,duration,flags",
                "-of",
                "csv=p=0",
                f"file:{copy}",
            ],
            seconds=END_SECONDS,
            stdout_limit=MAX_END_BYTES,
            stderr_limit=MAX_ERROR_BYTES,
        )
    except TimeoutError:
        raise TrimRefused("video-trim-timed-out") from None
    except ToolOutputTooLarge:
        raise TrimRefused("video-trim-exact-too-long") from None
    except OSError as exc:
        raise tool_unavailable(exc) from exc
    if answer.returncode != 0:
        raise TrimRefused("video-trim-frames-unknown")
    return answer.stdout


async def _output_frames(output: Path, ffprobe: MediaTool) -> list[FrameRow]:
    """The new video's frames in the order shown; none may be marked to be discarded."""

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
                "-show_entries",
                "packet=pts,duration,flags",
                "-of",
                "csv=p=0",
                f"file:{output}",
            ],
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
    if re.search(rb",[A-Z_]*D[A-Z_]*,*\s*$", answer.stdout, re.M):
        raise TrimRefused("video-trim-frames-unverified")
    try:
        return frame_rows(answer.stdout)
    except TrimRefused:
        raise TrimRefused("video-trim-frames-unverified") from None


def _input(copy: Path, seek: float | None, *, banner: bool = True) -> list[str]:
    """The source as ffmpeg reads it: its own timestamps kept, decoded from the keyframe sought."""

    arguments = ["-hide_banner", "-nostdin", "-nostats", "-loglevel", "error"] if banner else []
    arguments += ["-protocol_whitelist", "file", "-format_whitelist", DEMUXERS, "-noautorotate"]
    if seek is not None:
        arguments += ["-noaccurate_seek", "-ss", f"{seek:.6f}"]
    return [*arguments, "-copyts", "-i", f"file:{copy}"]


def _began_with() -> str:
    """Split a decode in two: ``[k]`` keeps the first keyframe it gives, ``[part]`` every frame."""

    return "split[all][part];[all]select=key,trim=end_frame=1[k]"


def _frame_trim(plan: ExactTrimPlan) -> str:
    """The filter that keeps exactly the planned frames, by their timestamps."""

    return f"trim=start_pts={plan.frames[0]}:end_pts={plan.frames[-1] + 1}"


def _colours(video: VideoStreamFacts) -> list[str]:
    """The source's stated colours, written into the new video the same way."""

    arguments: list[str] = []
    transfer = video.color_transfer
    for option, value in (
        ("-color_primaries", video.color_primaries),
        ("-color_trc", None if transfer is None else _ENCODER_TRANSFERS.get(transfer, transfer)),
        ("-colorspace", video.color_space),
        ("-color_range", video.color_range),
    ):
        if value is not None:
            arguments += [option, value]
    return arguments


def _same_picture(found: VideoStreamFacts, planned: VideoStreamFacts) -> bool:
    """The new picture stream is H.264 4:2:0 of the source's size and shape, in its colours."""

    stated = (
        (found.color_range, planned.color_range),
        (found.color_space, planned.color_space),
        (found.color_transfer, planned.color_transfer),
        (found.color_primaries, planned.color_primaries),
    )
    return (
        found.codec == "h264"
        and found.pixel_format == "yuv420p"
        and (found.width, found.height, found.sample_aspect)
        == (
            planned.width,
            planned.height,
            planned.sample_aspect,
        )
        and found.rotation == 0
        and all(new is None or new == old for new, old in stated)
    )


def _time_base(stated: str | None) -> Fraction:
    match = re.fullmatch(r"(\d{1,10})/(\d{1,10})", stated or "")
    if match is None or int(match.group(1)) == 0 or int(match.group(2)) == 0:
        raise TrimRefused("video-trim-frames-unknown")
    return Fraction(int(match.group(1)), int(match.group(2)))


def _audio_bitrate(stream: AudioStreamFacts) -> int:
    return _AUDIO_BITRATES.get(stream.channels or 2, _AUDIO_BITRATES[2])


def _digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)
