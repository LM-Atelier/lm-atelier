"""Save one frame of a stored video as a new picture, as a standalone durable job.

A utility job is not a conversation turn: it has no chat, run or plan. It runs
in a scheduler group of its own, one job at a time, so it never takes the
compute slot that image and video generation share. Its payload is the
immutable request; its result names the picture made, the requested and actual
times, and the ffmpeg found on the system path, by the version it reports and
its file's digest.

Retention, for now: the payload's ``source_artifact_id`` and the result's
``result_artifact_id`` are retention references, so both the video a frame came
from and the saved picture are kept while the job exists. Letting a source go
once its utilities are done needs a hold that ends with the job, which does not
exist yet; until then nothing here claims a source can be freed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from pathlib import PurePath
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .artifact_library import ensure_library_entry
from .artifacts import ArtifactCopyUnavailable, ArtifactStore
from .db import SessionLocal
from .domain import ArtifactKind, JobKind, JobStatus, new_id, utcnow
from .media_process import outlast_cancellation
from .media_tools import MediaToolUnavailable, ToolUnavailableCode, find_media_tool
from .models import Artifact, Job
from .progress import update_job_progress
from .scheduler import JobClaim, ResourceScheduler
from .video_frames import DecodedFrame, FrameRefused, decode_frame
from .video_probe import (
    MAX_INPUT_BYTES,
    VideoProbeRefused,
    probe_copy,
    require_probe_input,
)

#: The scheduler group utility jobs take turns in. No generation job uses it.
UTILITY_GROUP: Final = "cpu_media"
_FINISHED: Final = frozenset(
    {JobStatus.COMPLETE.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}
)

logger = logging.getLogger(__name__)


class FrameRequest(BaseModel):
    """The immutable request a frame job keeps: which video, by digest, and when."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    action: Literal["extract_frame"] = "extract_frame"
    source_artifact_id: str = Field(min_length=1, max_length=80)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_seconds: float = Field(ge=0, allow_inf_nan=False)


_TOOL_MESSAGES: Final[dict[ToolUnavailableCode, str]] = {
    "media-tool-missing": "FFmpeg and FFprobe are needed and one of them is not installed.",
    "media-tool-unreadable": "FFmpeg or FFprobe on this computer did not answer as expected.",
}


class _SourceChanged(Exception):
    code: Final = "video-utility-source-changed"

    def __str__(self) -> str:
        return "The video this job was asked for is no longer the one stored."


class VideoUtilityManager:
    """Start, recover, cancel and stop frame jobs; each runs as its own task."""

    def __init__(self, store: ArtifactStore, scheduler: ResourceScheduler) -> None:
        self._store = store
        self._scheduler = scheduler
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def stage_frame(self, session: Session, artifact: Artifact, requested_seconds: float) -> Job:
        """Add a queued frame job to the caller's transaction, without committing or starting it."""

        request = FrameRequest(
            source_artifact_id=artifact.id,
            source_sha256=artifact.sha256,
            requested_seconds=requested_seconds,
        )
        job = Job(
            kind=JobKind.MEDIA_UTILITY.value,
            status=JobStatus.QUEUED.value,
            queue_resource=UTILITY_GROUP,
            queue_group=UTILITY_GROUP,
            queue_ticket=new_id("ticket"),
            enqueued_at=utcnow(),
            payload_json=request.model_dump(mode="json"),
        )
        update_job_progress(job, stage="queued", queue_resource=UTILITY_GROUP, indeterminate=True)
        session.add(job)
        return job

    def start(self, job_id: str) -> None:
        previous = self._tasks.get(job_id)
        if previous is not None and not previous.done():
            return
        task = asyncio.create_task(self._run(job_id), name=f"video-utility-{job_id}")
        self._tasks[job_id] = task

        def discard(done: asyncio.Task[None]) -> None:
            if self._tasks.get(job_id) is done:
                self._tasks.pop(job_id, None)

        task.add_done_callback(discard)

    def recover(self) -> None:
        """Start every frame job a previous run left waiting or interrupted.

        A frame job's request is immutable and its only output is committed in
        one step, so an interrupted job is started again from its request
        rather than resumed.
        """

        waiting: list[str] = []
        with SessionLocal() as session:
            session.execute(text("UPDATE jobs SET status = status WHERE 0"))
            jobs = session.scalars(
                select(Job)
                .where(
                    Job.kind == JobKind.MEDIA_UTILITY.value,
                    Job.status.in_((JobStatus.QUEUED.value, JobStatus.INTERRUPTED.value)),
                    Job.claim_owner.is_(None),
                )
                .order_by(Job.enqueued_at, Job.queue_ticket, Job.created_at, Job.id)
            ).all()
            for job in jobs:
                if job.status == JobStatus.INTERRUPTED.value:
                    _requeue(job, "queued")
                waiting.append(job.id)
            session.commit()
        for job_id in waiting:
            self.start(job_id)

    def stage_retry(self, job: Job) -> None:
        """Queue an unsuccessful frame job again from its unchanged request."""

        _requeue(job, "retry queued")

    async def cancel(self, job_id: str) -> bool:
        """Stop the job's task first, then mark the row cancelled; never the reverse.

        Marking the row while the task still runs would let a frame that was
        about to be saved write a completed job over a cancelled one. The task
        ends only once its tools have exited and any write it had begun has
        finished, so a job whose picture was already being saved ends complete
        and is not cancelled.
        """

        with SessionLocal() as session:
            initial = session.get(Job, job_id)
            if initial is None:
                return False
            attempt, ticket = initial.attempt, initial.queue_ticket
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        with SessionLocal() as session:
            session.execute(text("UPDATE jobs SET status = status WHERE 0"))
            job = session.get(Job, job_id)
            if (
                job is None
                or job.attempt != attempt
                or job.queue_ticket != ticket
                or job.status in _FINISHED
            ):
                return False
            job.status = JobStatus.CANCELLED.value
            job.completed_at = utcnow()
            update_job_progress(job, stage="cancelled", indeterminate=True)
            session.commit()
        return True

    async def close(self) -> None:
        """Stop every running frame job and mark unfinished ones interrupted for the next start."""

        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        with SessionLocal() as session:
            session.execute(text("UPDATE jobs SET status = status WHERE 0"))
            for job in session.scalars(
                select(Job).where(
                    Job.kind == JobKind.MEDIA_UTILITY.value,
                    Job.status.in_((JobStatus.QUEUED.value, JobStatus.RUNNING.value)),
                )
            ).all():
                job.status = JobStatus.INTERRUPTED.value
                job.claim_owner = None
                update_job_progress(job, stage="interrupted", indeterminate=True)
            session.commit()

    async def _run(self, job_id: str) -> None:
        execution = asyncio.current_task()

        def displaced() -> None:
            # Another claimant owns the job now; this attempt writes nothing more.
            if execution is not None:
                execution.cancel()

        async with self._scheduler.job_lease(
            job_id,
            resource=UTILITY_GROUP,
            group=UTILITY_GROUP,
            capacity=1,
            on_claim_lost=displaced,
        ) as claim:
            try:
                request, artifact = await asyncio.to_thread(self._accepted, job_id, claim)
                ffprobe = await find_media_tool("ffprobe")
                ffmpeg = await find_media_tool("ffmpeg")
                require_probe_input(artifact)
                await asyncio.to_thread(self._report, job_id, claim, "Reading the video")
                # One copy of exactly the verified bytes serves both tools.
                async with self._store.verified_copy(
                    artifact, maximum_bytes=MAX_INPUT_BYTES
                ) as copy:
                    probe = await probe_copy(copy, artifact, ffprobe)
                    await asyncio.to_thread(self._report, job_id, claim, "Decoding the frame")
                    frame = await decode_frame(
                        copy, artifact, probe, ffmpeg, request.requested_seconds
                    )
                record = _frame_record(request, frame, ffmpeg.record())
                await _write(lambda: self._finish(job_id, claim, artifact, frame, record))
            except MediaToolUnavailable as exc:
                failure: str = exc.code
                message = _TOOL_MESSAGES[exc.code]
                await _write(lambda: self._fail(job_id, claim, failure, message))
            except (VideoProbeRefused, FrameRefused, _SourceChanged) as exc:
                failure, message = exc.code, str(exc)
                await _write(lambda: self._fail(job_id, claim, failure, message))
            except ArtifactCopyUnavailable:
                await _write(
                    lambda: self._fail(
                        job_id,
                        claim,
                        "video-utility-copy-unavailable",
                        "The video could not be copied for reading. "
                        "Check that the drive holding the app's data has room.",
                    )
                )
            except (FileNotFoundError, ValueError):
                await _write(
                    lambda: self._fail(
                        job_id,
                        claim,
                        "artifact-file-unreadable",
                        "The stored video is missing or corrupt.",
                    )
                )
            except Exception:
                # Anything else, a full disk while copying for one, still ends
                # the job; a job left running would hold its place for good.
                logger.exception("A video utility job failed")
                await _write(
                    lambda: self._fail(
                        job_id, claim, "video-utility-failed", "The frame could not be saved."
                    )
                )

    def _accepted(self, job_id: str, claim: JobClaim) -> tuple[FrameRequest, Artifact]:
        with SessionLocal() as session:
            job = _current(session, job_id, claim)
            if job is None:
                raise asyncio.CancelledError
            try:
                request = FrameRequest.model_validate(job.payload_json)
            except ValidationError:
                raise _SourceChanged from None
            artifact = session.get(Artifact, request.source_artifact_id)
            if artifact is None or artifact.sha256 != request.source_sha256:
                raise _SourceChanged
            session.expunge(artifact)
            return request, artifact

    def _report(self, job_id: str, claim: JobClaim, stage: str) -> None:
        with SessionLocal() as session:
            job = _current(session, job_id, claim)
            if job is None:
                raise asyncio.CancelledError
            update_job_progress(job, stage=stage, indeterminate=True)
            session.commit()

    def _finish(
        self,
        job_id: str,
        claim: JobClaim,
        source: Artifact,
        frame: DecodedFrame,
        record: dict[str, Any],
    ) -> None:
        # The picture is stored first, in its own transaction, so no file is
        # written while the job row's writer is held. A picture whose job then
        # turns out to be displaced has no library entry and nothing refers to
        # it, so ordinary retention removes it.
        with SessionLocal() as session:
            picture = self._store.ingest_bytes(
                session,
                frame.png,
                kind=ArtifactKind.IMAGE,
                media_type="image/png",
                original_name=_frame_name(source, frame),
                metadata={"video_frame": record},
            )
            session.commit()
            picture_id = picture.id
        with SessionLocal() as session:
            job = _current(session, job_id, claim)
            if job is None:
                return
            saved = session.get(Artifact, picture_id)
            if saved is None:
                raise FileNotFoundError(picture_id)
            ensure_library_entry(session, saved)
            job.result_json = {"result_artifact_id": picture_id, **record}
            job.status = JobStatus.COMPLETE.value
            job.completed_at = utcnow()
            job.progress = 1.0
            update_job_progress(job, stage="Frame saved", overall_progress=1.0)
            session.commit()

    def _fail(self, job_id: str, claim: JobClaim, code: str, message: str) -> None:
        with SessionLocal() as session:
            job = _current(session, job_id, claim)
            if job is None:
                return
            job.status = JobStatus.FAILED.value
            job.error = message
            job.result_json = {"failure_code": code}
            job.completed_at = utcnow()
            update_job_progress(job, stage="Frame not saved", indeterminate=True)
            session.commit()


async def _write(work: Callable[[], None]) -> None:
    """Run a job's final write to its end, even when the job is cancelled meanwhile.

    The write runs in a thread, which a cancellation cannot stop. Waiting for
    it means a cancelled job's task ends only once nothing of the job runs.
    """

    await outlast_cancellation(asyncio.ensure_future(asyncio.to_thread(work)))


def _requeue(job: Job, stage: str) -> None:
    job.status = JobStatus.QUEUED.value
    job.progress = 0.0
    job.error = None
    job.result_json = {}
    job.started_at = None
    job.completed_at = None
    job.enqueued_at = utcnow()
    job.claim_owner = None
    job.claim_expires_at = None
    job.heartbeat_at = None
    update_job_progress(job, stage=stage, queue_resource=UTILITY_GROUP, indeterminate=True)


def _current(session: Session, job_id: str, claim: JobClaim) -> Job | None:
    """The job, only while this claim still owns it and it is running; the writer is reserved."""

    session.execute(text("UPDATE jobs SET status = status WHERE 0"))
    job = session.get(Job, job_id, populate_existing=True)
    if (
        job is None
        or job.claim_owner != claim.token
        or job.attempt != claim.attempt
        or job.status != JobStatus.RUNNING.value
    ):
        return None
    return job


def _frame_record(
    request: FrameRequest, frame: DecodedFrame, tool: dict[str, str]
) -> dict[str, Any]:
    return {
        "action": request.action,
        "source_artifact_id": request.source_artifact_id,
        "source_sha256": request.source_sha256,
        "requested_seconds": frame.requested_seconds,
        "actual_seconds": frame.actual_seconds,
        "width": frame.width,
        "height": frame.height,
        "rotation_applied": frame.rotation_applied,
        "sample_aspect": frame.sample_aspect,
        "tool": tool,
    }


def _frame_name(source: Artifact, frame: DecodedFrame) -> str:
    stem = PurePath(source.original_name or "video").stem[:80] or "video"
    return f"{stem} frame {frame.actual_seconds:.3f}s.png"
