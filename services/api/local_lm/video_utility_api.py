"""Read what a stored video is, save one of its frames as a new picture, and run their lane."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final, Literal, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .api_errors import api_error
from .artifacts import ArtifactCopyUnavailable
from .db import SessionLocal
from .domain import JobKind
from .media_tools import MediaToolUnavailable, ToolUnavailableCode, find_media_tool
from .models import Artifact, Job
from .queue_lane_policy import (
    LanePolicy,
    QueueLaneConflict,
    change_lane_policy,
    read_lane_policy,
)
from .schemas import JobOut, QueueControlCommand, UtilityQueuePolicyOut
from .video_probe import VideoProbe, VideoProbeRefused, probe_video, require_probe_input

if TYPE_CHECKING:
    from .main import Services

router = APIRouter()

_TOOL_MESSAGES: Final[dict[ToolUnavailableCode, str]] = {
    "media-tool-missing": "FFprobe is not installed on this computer.",
    "media-tool-unreadable": "FFprobe on this computer did not answer as FFprobe.",
}
_UNAVAILABLE: Final = (
    "The video could not be read right now. Check that the drive holding the app's data has room."
)


class VideoFrameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Seconds from the start of the video. The saved frame is the one shown at that time.
    requested_seconds: float = Field(ge=0, allow_inf_nan=False)


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
    try:
        # A file the probe would never read is refused before ffprobe is looked for.
        require_probe_input(artifact)
        ffprobe = await find_media_tool("ffprobe")
        return await probe_video(store, artifact, ffprobe)
    except VideoProbeRefused as exc:
        raise api_error(422, exc.code, str(exc)) from exc
    except MediaToolUnavailable as exc:
        raise api_error(503, exc.code, _TOOL_MESSAGES[exc.code]) from exc
    except ArtifactCopyUnavailable as exc:
        # The stored file is intact; its private copy could not be made.
        raise api_error(503, "video-probe-unavailable", _UNAVAILABLE) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise api_error(
            410, "artifact-file-unreadable", "artifact file is missing or corrupt"
        ) from exc
