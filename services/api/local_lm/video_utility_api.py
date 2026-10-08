"""Read what a stored video is, save a frame of it or trim it, and run the utilities' lane."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, Literal, cast

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from .api_errors import api_error
from .artifacts import ArtifactCopyUnavailable
from .db import SessionLocal
from .domain import JobKind
from .media_tools import MediaTool, MediaToolUnavailable, ToolUnavailableCode, find_media_tool
from .models import Artifact, Job
from .queue_lane_policy import (
    LanePolicy,
    QueueLaneConflict,
    change_lane_policy,
    read_lane_policy,
)
from .schemas import JobOut, QueueControlCommand, UtilityQueuePolicyOut
from .video_probe import (
    MAX_DURATION_SECONDS,
    MAX_INPUT_BYTES,
    VideoProbe,
    VideoProbeRefused,
    probe_copy,
    probe_video,
    require_probe_input,
)
from .video_trim import TrimMode, TrimRefused, VideoTrimPreview, plan_trim
from .video_trim_exact import MAX_EXACT_FRAMES, plan_exact_trim

if TYPE_CHECKING:
    from .main import Services

router = APIRouter()

_TOOL_MESSAGES: Final[dict[ToolUnavailableCode, str]] = {
    "media-tool-missing": "FFprobe is not installed on this computer.",
    "media-tool-unreadable": "FFprobe on this computer did not answer as FFprobe.",
}
_FFMPEG_MESSAGES: Final[dict[ToolUnavailableCode, str]] = {
    "media-tool-missing": "FFmpeg is not installed on this computer.",
    "media-tool-unreadable": "FFmpeg on this computer did not answer as FFmpeg.",
}
_UNAVAILABLE: Final = (
    "The video could not be read right now. Check that the drive holding the app's data has room."
)
#: How far apart two starts may be and still be the start that was shown.
_SAME_START: Final = 1e-6


class VideoFrameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Seconds from the start of the video. The saved frame is the one shown at that time.
    requested_seconds: float = Field(ge=0, allow_inf_nan=False)


class VideoTrimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Seconds from the start of the video. A copy begins on the keyframe at or
    #: before it; an exact cut on the frame shown at it.
    start_seconds: float = Field(ge=0, le=MAX_DURATION_SECONDS, allow_inf_nan=False)
    end_seconds: float = Field(ge=0, le=MAX_DURATION_SECONDS, allow_inf_nan=False)
    keep_audio: bool
    #: The start the trim check showed, so a trim never copies a part other than the one shown.
    shown_start_seconds: float = Field(ge=0, le=MAX_DURATION_SECONDS, allow_inf_nan=False)
    mode: TrimMode = "copy"
    #: How many frames the check showed an exact cut holding; only an exact cut sends it.
    shown_frame_count: int | None = Field(default=None, ge=1, le=MAX_EXACT_FRAMES)

    @model_validator(mode="after")
    def _frames_shown_for_an_exact_cut(self) -> VideoTrimRequest:
        if (self.mode == "exact") != (self.shown_frame_count is not None):
            raise ValueError("an exact cut, and only an exact cut, sends the frames it was shown")
        return self


@router.get("/artifacts/{artifact_id}/video-probe")
async def read_video_probe(artifact_id: str, request: Request) -> VideoProbe:
    """What a stored video is and what the video utilities may do with it, or why not."""

    artifact = _stored(artifact_id)
    return await _probe(request, artifact)


@router.post("/artifacts/{artifact_id}/video-frames", status_code=202, response_model=JobOut)
async def save_video_frame(artifact_id: str, payload: VideoFrameRequest, request: Request) -> Job:
    """Queue saving the frame at a time as a new picture, once the video is known to offer it."""

    artifact = _stored(artifact_id)
    probe = await _probe(request, artifact)
    if not probe.can_save_frame:
        raise api_error(422, "video-frame-not-offered", "A frame cannot be saved from this video.")
    duration = probe.duration_seconds
    # The end itself is within the video: a player there shows its last frame.
    if duration is None or payload.requested_seconds > duration:
        raise api_error(422, "video-frame-time-outside", "That time is not within this video.")
    utilities = cast("Services", request.app.state.services).video_utilities
    with SessionLocal() as session:
        job = utilities.stage_frame(session, artifact, payload.requested_seconds)
        session.commit()
        session.refresh(job)
        session.expunge(job)
    utilities.start(job.id)
    return job


@router.get("/artifacts/{artifact_id}/video-trim-preview")
async def preview_video_trim(
    artifact_id: str,
    request: Request,
    start_seconds: float = Query(ge=0, le=MAX_DURATION_SECONDS, allow_inf_nan=False),
    end_seconds: float = Query(ge=0, le=MAX_DURATION_SECONDS, allow_inf_nan=False),
    keep_audio: bool = Query(),
    mode: TrimMode = "copy",
) -> VideoTrimPreview:
    """Where a trim of the chosen part would begin, worked out without queueing anything."""

    artifact = _stored(artifact_id)
    ffmpeg = None
    if mode == "exact":
        with _refused():
            require_probe_input(artifact)
        # Only an exact cut runs FFmpeg to plan, so only it needs FFmpeg to check.
        ffmpeg = await _ffmpeg()
    preview, _has_sound = await _preview(
        request, artifact, start_seconds, end_seconds, keep_audio, ffmpeg
    )
    return preview


@router.post("/artifacts/{artifact_id}/video-trims", status_code=202, response_model=JobOut)
async def trim_video(artifact_id: str, payload: VideoTrimRequest, request: Request) -> Job:
    """Queue cutting the chosen part into a new video, when it is the part the check showed."""

    artifact = _stored(artifact_id)
    with _refused():
        require_probe_input(artifact)
    # The job needs FFmpeg too; without it nothing is queued.
    ffmpeg = await _ffmpeg()
    preview, has_sound = await _preview(
        request,
        artifact,
        payload.start_seconds,
        payload.end_seconds,
        payload.keep_audio,
        ffmpeg if payload.mode == "exact" else None,
    )
    if abs(preview.start_seconds - payload.shown_start_seconds) > _SAME_START:
        raise api_error(
            409,
            "video-trim-preview-stale",
            "Where this cut starts has changed. Check the cut again.",
        )
    if preview.frame_count != payload.shown_frame_count:
        raise api_error(
            409,
            "video-trim-preview-stale",
            "The frames this cut keeps have changed. Check the cut again.",
        )
    if preview.keeps_whole_video:
        if preview.mode == "exact":
            # Without its sound, a copy of the whole video is a change; a silent one is not.
            advice = "Choose a later start or an earlier end." + (
                " To leave out only its sound, keep the original quality instead."
                if has_sound
                else ""
            )
            message = f"That keeps every frame of the video. {advice}"
        else:
            advice = (
                "Choose an earlier end, or leave out the sound."
                if preview.audio_streams_kept
                else "Choose an earlier end."
            )
            message = f"That would keep the whole video as it is. {advice}"
        raise api_error(422, "video-trim-keeps-whole-video", message)
    utilities = cast("Services", request.app.state.services).video_utilities
    with SessionLocal() as session:
        job = utilities.stage_trim(
            session,
            artifact,
            start_seconds=preview.requested_start_seconds,
            end_seconds=preview.requested_end_seconds,
            keep_audio=preview.keep_audio,
            shown_start_seconds=payload.shown_start_seconds,
            mode=payload.mode,
            shown_frame_count=payload.shown_frame_count,
        )
        session.commit()
        session.refresh(job)
        session.expunge(job)
    utilities.start(job.id)
    return job


@router.get("/video-utilities/jobs/{job_id}", response_model=JobOut)
async def read_video_utility_job(job_id: str) -> Job:
    """One video utility job, so a waiting page can follow it to its result."""

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is not None:
            session.expunge(job)
    if job is None or job.kind != JobKind.MEDIA_UTILITY.value:
        raise api_error(404, "job-not-found", "job not found")
    return job


@router.get("/queue/lanes/utility", response_model=UtilityQueuePolicyOut)
async def utility_queue_policy() -> UtilityQueuePolicyOut:
    result = await run_in_threadpool(_lane, None, None)
    return UtilityQueuePolicyOut.model_validate(dataclasses.asdict(result))


@router.post("/queue/lanes/utility/pause-after-current", response_model=UtilityQueuePolicyOut)
async def pause_utility_queue(
    request: Request, payload: QueueControlCommand
) -> UtilityQueuePolicyOut:
    return await _change_lane(request, "pause_after_current", payload)


@router.post("/queue/lanes/utility/resume", response_model=UtilityQueuePolicyOut)
async def resume_utility_queue(
    request: Request, payload: QueueControlCommand
) -> UtilityQueuePolicyOut:
    return await _change_lane(request, "resume", payload)


async def _change_lane(
    request: Request,
    action: Literal["pause_after_current", "resume"],
    payload: QueueControlCommand,
) -> UtilityQueuePolicyOut:
    result = await run_in_threadpool(_lane, action, payload)
    await cast("Services", request.app.state.services).scheduler.queue_control_changed("utility")
    return UtilityQueuePolicyOut.model_validate(dataclasses.asdict(result))


def _lane(
    action: Literal["pause_after_current", "resume"] | None,
    payload: QueueControlCommand | None,
) -> LanePolicy:
    try:
        with SessionLocal() as session:
            if action is None or payload is None:
                return read_lane_policy(session, "utility")
            return change_lane_policy(session, "utility", action, payload)
    except QueueLaneConflict as exc:
        raise api_error(
            409,
            "queue-lane-conflict",
            "The video utility queue changed. Refresh before trying again.",
        ) from exc


def _stored(artifact_id: str) -> Artifact:
    # The session closes before ffprobe runs, so no read is held open across it.
    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        if artifact is not None:
            session.expunge(artifact)
    if artifact is None:
        raise api_error(404, "artifact-not-found", "artifact not found")
    return artifact


async def _probe(request: Request, artifact: Artifact) -> VideoProbe:
    store = cast("Services", request.app.state.services).artifacts
    with _refused():
        # A file the probe would never read is refused before ffprobe is looked for.
        require_probe_input(artifact)
        ffprobe = await find_media_tool("ffprobe")
        return await probe_video(store, artifact, ffprobe)


async def _preview(
    request: Request,
    artifact: Artifact,
    start: float,
    end: float,
    keep_audio: bool,
    ffmpeg: MediaTool | None,
) -> tuple[VideoTrimPreview, bool]:
    """A copy's preview, or with ``ffmpeg`` an exact cut's, and whether the video has sound."""

    store = cast("Services", request.app.state.services).artifacts
    with _refused():
        require_probe_input(artifact)
        ffprobe = await find_media_tool("ffprobe")
        # One copy of exactly the verified bytes serves the probe and every lookup.
        async with store.verified_copy(artifact, maximum_bytes=MAX_INPUT_BYTES) as copy:
            probe = await probe_copy(copy, artifact, ffprobe)
            if ffmpeg is not None:
                exact = await plan_exact_trim(
                    copy, artifact, probe, ffprobe, ffmpeg, start, end, keep_audio
                )
                return exact.preview, bool(probe.audio)
            plan = await plan_trim(copy, artifact, probe, ffprobe, start, end, keep_audio)
            return plan.preview, bool(probe.audio)


async def _ffmpeg() -> MediaTool:
    try:
        return await find_media_tool("ffmpeg")
    except MediaToolUnavailable as exc:
        raise api_error(503, exc.code, _FFMPEG_MESSAGES[exc.code]) from exc


@contextmanager
def _refused() -> Iterator[None]:
    """Answer the video utilities' refusals with their own codes."""

    try:
        yield
    except (VideoProbeRefused, TrimRefused) as exc:
        # Every refusal is about the video or the part, but one: an FFmpeg that cannot encode.
        status = 503 if exc.code == "video-trim-encoder-missing" else 422
        raise api_error(status, exc.code, str(exc)) from exc
    except MediaToolUnavailable as exc:
        raise api_error(503, exc.code, _TOOL_MESSAGES[exc.code]) from exc
    except ArtifactCopyUnavailable as exc:
        # The stored file is intact; its private copy could not be made.
        raise api_error(503, "video-probe-unavailable", _UNAVAILABLE) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise api_error(
            410, "artifact-file-unreadable", "artifact file is missing or corrupt"
        ) from exc
