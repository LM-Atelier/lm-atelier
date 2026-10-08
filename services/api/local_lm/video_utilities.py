"""Save a frame of a stored video as a new picture, or trim it into a new video, as durable jobs.

A utility job is not a conversation turn: it has no chat, run or plan. It runs
in a scheduler group of its own, one job at a time, so it never takes the
compute slot that image and video generation share. Its payload is the
immutable request; its result names what was made, the requested and actual
times, and the ffmpeg found on the system path, by the version it reports and
its file's digest.

Retention, for now: the payload's ``source_artifact_id`` and the result's
``result_artifact_id`` are retention references, so both the video a utility
read and what it made are kept while the job exists. Letting a source go once
its utilities are done needs a hold that ends with the job, which does not
exist yet; until then nothing here claims a source can be freed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from pathlib import PurePath
from typing import Annotated, Any, Final, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .artifact_library import ensure_library_entry
from .artifacts import ArtifactCopyUnavailable, ArtifactStore, ToolOutputChanged
from .db import SessionLocal
from .domain import ArtifactKind, JobKind, JobStatus, new_id, utcnow
from .media_process import outlast_cancellation
from .media_tools import MediaTool, MediaToolUnavailable, ToolUnavailableCode, find_media_tool
from .models import Artifact, Job
from .progress import update_job_progress
from .scheduler import JobClaim, ResourceScheduler
from .video_frames import DecodedFrame, FrameRefused, decode_frame
from .video_probe import (
    MAX_DURATION_SECONDS,
    MAX_INPUT_BYTES,
    VideoProbeRefused,
    probe_copy,
    require_probe_input,
)
from .video_trim import (
    CUT_ROOM,
    TrimMode,
    TrimRefused,
    check_trim,
    cut_video,
    keyframe_picture,
    plan_trim,
    trim_record,
)
from .video_trim_exact import (
    MAX_EXACT_FRAMES,
    check_exact_trim,
    compare_with_source,
    encode_part,
    exact_trim_record,
    plan_exact_trim,
)

#: The scheduler group utility jobs take turns in. No generation job uses it.
UTILITY_GROUP: Final = "cpu_media"
_FINISHED: Final = frozenset(
    {JobStatus.COMPLETE.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}
)
#: How far apart two times read from the same file may be and still be the same time.
_SAME_START: Final = 1e-6

logger = logging.getLogger(__name__)


class FrameRequest(BaseModel):
    """The immutable request a frame job keeps: which video, by digest, and when."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    action: Literal["extract_frame"] = "extract_frame"
    source_artifact_id: str = Field(min_length=1, max_length=80)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_seconds: float = Field(ge=0, allow_inf_nan=False)


class TrimRequest(BaseModel):
    """The immutable request a trim job keeps, with the start the person was shown.

    The start is worked out again when the job runs; a job whose start would
    differ from the one shown fails rather than copying another part. A cut on
    exact frames also keeps how many frames it was shown to hold.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    action: Literal["trim"] = "trim"
    source_artifact_id: str = Field(min_length=1, max_length=80)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_start_seconds: float = Field(ge=0, le=MAX_DURATION_SECONDS, allow_inf_nan=False)
    requested_end_seconds: float = Field(gt=0, le=MAX_DURATION_SECONDS, allow_inf_nan=False)
    keep_audio: bool
    shown_start_seconds: float = Field(ge=0, le=MAX_DURATION_SECONDS, allow_inf_nan=False)
    mode: TrimMode = "copy"
    shown_frame_count: int | None = Field(default=None, ge=1, le=MAX_EXACT_FRAMES)

    @model_validator(mode="after")
    def _frames_shown_for_an_exact_cut(self) -> TrimRequest:
        if (self.mode == "exact") != (self.shown_frame_count is not None):
            raise ValueError("an exact cut, and only an exact cut, keeps the frames it was shown")
        return self


UtilityRequest = Annotated[FrameRequest | TrimRequest, Field(discriminator="action")]
_REQUEST: Final[TypeAdapter[FrameRequest | TrimRequest]] = TypeAdapter(UtilityRequest)


class _Outcome(NamedTuple):
    done: str
    not_done: str
    failed: str


_OUTCOMES: Final[dict[str, _Outcome]] = {
    "extract_frame": _Outcome("Frame saved", "Frame not saved", "The frame could not be saved."),
    "trim": _Outcome("Video trimmed", "Video not trimmed", "The video could not be trimmed."),
}
#: For a job whose request cannot be read, so its action is not known.
_UNKNOWN: Final = _Outcome("Finished", "Not finished", "The video utility could not finish.")

_TOOL_MESSAGES: Final[dict[ToolUnavailableCode, str]] = {
    "media-tool-missing": "FFmpeg and FFprobe are needed and one of them is not installed.",
    "media-tool-unreadable": "FFmpeg or FFprobe on this computer did not answer as expected.",
}


class _SourceChanged(Exception):
    code: Final = "video-utility-source-changed"

    def __init__(self, action: object = None) -> None:
        super().__init__()
        self.outcome = _OUTCOMES.get(str(action), _UNKNOWN)

    def __str__(self) -> str:
        return "The video this job was asked for is no longer the one stored."


class VideoUtilityManager:
    """Start, recover, cancel and stop utility jobs; each runs as its own task."""

    def __init__(self, store: ArtifactStore, scheduler: ResourceScheduler) -> None:
        self._store = store
        self._scheduler = scheduler
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def stage_frame(self, session: Session, artifact: Artifact, requested_seconds: float) -> Job:
        """Add a queued frame job to the caller's transaction, without committing or starting it."""

        return _stage(
            session,
            FrameRequest(
                source_artifact_id=artifact.id,
                source_sha256=artifact.sha256,
                requested_seconds=requested_seconds,
            ),
        )

    def stage_trim(
        self,
        session: Session,
        artifact: Artifact,
        *,
        start_seconds: float,
        end_seconds: float,
        keep_audio: bool,
        shown_start_seconds: float,
        mode: TrimMode = "copy",
        shown_frame_count: int | None = None,
    ) -> Job:
        """Add a queued trim job to the caller's transaction, without committing or starting it."""

        return _stage(
            session,
            TrimRequest(
                source_artifact_id=artifact.id,
                source_sha256=artifact.sha256,
                requested_start_seconds=start_seconds,
                requested_end_seconds=end_seconds,
                keep_audio=keep_audio,
                shown_start_seconds=shown_start_seconds,
                mode=mode,
                shown_frame_count=shown_frame_count,
            ),
        )

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
        """Start every utility job a previous run left waiting or interrupted.

        A utility job's request is immutable and its only output is committed
        in one step, so an interrupted job is started again from its request
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
        """Queue an unsuccessful utility job again from its unchanged request."""

        _requeue(job, "retry queued")

    async def cancel(self, job_id: str) -> bool:
        """Stop the job's task first, then mark the row cancelled; never the reverse.

        Marking the row while the task still runs would let a result that was
        about to be saved write a completed job over a cancelled one. The task
        ends only once its tools have exited and any write it had begun has
        finished, so a job whose result was already being saved ends complete
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
        """Stop every running utility job; unfinished ones are taken up at the next start."""

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
            outcome = _UNKNOWN
            try:
                request, artifact = await asyncio.to_thread(self._accepted, job_id, claim)
                outcome = _OUTCOMES[request.action]
                ffprobe = await find_media_tool("ffprobe")
                ffmpeg = await find_media_tool("ffmpeg")
                require_probe_input(artifact)
                if isinstance(request, TrimRequest):
                    await self._trim(job_id, claim, request, artifact, ffprobe, ffmpeg)
                else:
                    await self._frame(job_id, claim, request, artifact, ffprobe, ffmpeg)
            except MediaToolUnavailable as exc:
                failure: str = exc.code
                message = _TOOL_MESSAGES[exc.code]
                await _write(lambda: self._fail(job_id, claim, failure, message, outcome))
            except _SourceChanged as exc:
                failure, message, outcome = exc.code, str(exc), exc.outcome
                await _write(lambda: self._fail(job_id, claim, failure, message, outcome))
            except (VideoProbeRefused, FrameRefused, TrimRefused) as exc:
                failure, message = exc.code, str(exc)
                await _write(lambda: self._fail(job_id, claim, failure, message, outcome))
            except ArtifactCopyUnavailable:
                await _write(
                    lambda: self._fail(
                        job_id,
                        claim,
                        "video-utility-copy-unavailable",
                        "The video utilities could not make their working files. "
                        "Check that the drive holding the app's data has room.",
                        outcome,
                    )
                )
            except (FileNotFoundError, ValueError):
                await _write(
                    lambda: self._fail(
                        job_id,
                        claim,
                        "artifact-file-unreadable",
                        "The stored video is missing or corrupt.",
                        outcome,
                    )
                )
            except Exception:
                # Anything else, a full disk while copying for one, still ends
                # the job; a job left running would hold its place for good.
                logger.exception("A video utility job failed")
                await _write(
                    lambda: self._fail(
                        job_id, claim, "video-utility-failed", outcome.failed, outcome
                    )
                )

    async def _frame(
        self,
        job_id: str,
        claim: JobClaim,
        request: FrameRequest,
        artifact: Artifact,
        ffprobe: MediaTool,
        ffmpeg: MediaTool,
    ) -> None:
        await asyncio.to_thread(self._report, job_id, claim, "Reading the video")
        # One copy of exactly the verified bytes serves both tools.
        async with self._store.verified_copy(artifact, maximum_bytes=MAX_INPUT_BYTES) as copy:
            probe = await probe_copy(copy, artifact, ffprobe)
            await asyncio.to_thread(self._report, job_id, claim, "Decoding the frame")
            frame = await decode_frame(copy, artifact, probe, ffmpeg, request.requested_seconds)
        record = _frame_record(request, frame, ffmpeg.record())

        def store_picture(session: Session) -> Artifact:
            return self._store.ingest_bytes(
                session,
                frame.png,
                kind=ArtifactKind.IMAGE,
                media_type="image/png",
                original_name=_frame_name(artifact, frame),
                metadata={"video_frame": record},
            )

        done = _OUTCOMES[request.action].done
        await _write(lambda: self._finish(job_id, claim, store_picture, record, done))

    async def _trim(
        self,
        job_id: str,
        claim: JobClaim,
        request: TrimRequest,
        artifact: Artifact,
        ffprobe: MediaTool,
        ffmpeg: MediaTool,
    ) -> None:
        if request.mode == "exact":
            await self._exact_trim(job_id, claim, request, artifact, ffprobe, ffmpeg)
            return
        await asyncio.to_thread(self._report, job_id, claim, "Reading the video")
        # The new video is written into a file the store made, which outlives
        # the copy it is cut from so that it can be checked and kept after the
        # copy is gone; both are removed however the job ends.
        async with self._store.tool_output(maximum_bytes=artifact.size_bytes + CUT_ROOM) as output:
            async with self._store.verified_copy(artifact, maximum_bytes=MAX_INPUT_BYTES) as copy:
                probe = await probe_copy(copy, artifact, ffprobe)
                await asyncio.to_thread(self._report, job_id, claim, "Finding the keyframe")
                plan = await plan_trim(
                    copy,
                    artifact,
                    probe,
                    ffprobe,
                    request.requested_start_seconds,
                    request.requested_end_seconds,
                    request.keep_audio,
                )
                if abs(plan.preview.start_seconds - request.shown_start_seconds) > _SAME_START:
                    raise TrimRefused("video-trim-start-moved")
                keyframe = await keyframe_picture(copy, plan, ffmpeg)
                await asyncio.to_thread(self._report, job_id, claim, "Copying the chosen part")
                await cut_video(copy, plan, ffmpeg, output)
            written = await asyncio.to_thread(self._store.tool_output_size, output)
            # ffmpeg stops at its size limit and still exits cleanly, so a file
            # that reached the limit is a part, not the whole.
            if not 0 < written < output.maximum_bytes:
                raise TrimRefused("video-trim-not-written")
            await asyncio.to_thread(self._report, job_id, claim, "Checking the new video")
            # The checks read the new video by name, so it is sealed and held
            # unchanged until it is kept, and only what was sealed is kept.
            try:
                async with self._store.sealed_tool_output(output) as seal:
                    measured = await check_trim(output.path, plan, ffprobe, ffmpeg, keyframe)
                    record = trim_record(
                        artifact, plan, measured, ffmpeg.record(), ffprobe.record()
                    )
                    name = _trim_name(artifact, record, plan.suffix)

                    def store_video(session: Session) -> Artifact:
                        return self._store.ingest_tool_output(
                            session,
                            output,
                            seal=seal,
                            kind=ArtifactKind.VIDEO,
                            media_type=plan.preview.media_type,
                            original_name=name,
                            metadata={"video_trim": record},
                        )

                    done = _OUTCOMES[request.action].done
                    await _write(lambda: self._finish(job_id, claim, store_video, record, done))
            except ToolOutputChanged:
                raise TrimRefused("video-trim-output-mismatch") from None

    async def _exact_trim(
        self,
        job_id: str,
        claim: JobClaim,
        request: TrimRequest,
        artifact: Artifact,
        ffprobe: MediaTool,
        ffmpeg: MediaTool,
    ) -> None:
        await asyncio.to_thread(self._report, job_id, claim, "Reading the video")
        # The original is read again to compare the new video with it, so its
        # copy lasts until the new video is kept; both files are removed
        # however the job ends.
        async with self._store.verified_copy(artifact, maximum_bytes=MAX_INPUT_BYTES) as copy:
            probe = await probe_copy(copy, artifact, ffprobe)
            await asyncio.to_thread(self._report, job_id, claim, "Finding the frames")
            plan = await plan_exact_trim(
                copy,
                artifact,
                probe,
                ffprobe,
                ffmpeg,
                request.requested_start_seconds,
                request.requested_end_seconds,
                request.keep_audio,
            )
            if abs(plan.preview.start_seconds - request.shown_start_seconds) > _SAME_START:
                raise TrimRefused("video-trim-start-moved")
            if plan.preview.frame_count != request.shown_frame_count:
                raise TrimRefused("video-trim-frames-moved")
            async with self._store.tool_output(maximum_bytes=plan.output_bytes) as output:
                await asyncio.to_thread(self._report, job_id, claim, "Re-encoding the chosen part")
                fed = await encode_part(copy, plan, ffmpeg, output)
                written = await asyncio.to_thread(self._store.tool_output_size, output)
                # ffmpeg stops at its size limit and still exits cleanly, so a
                # file that reached the limit is a part, not the whole.
                if not 0 < written < output.maximum_bytes:
                    raise TrimRefused("video-trim-not-encoded")
                await asyncio.to_thread(self._report, job_id, claim, "Checking the new video")
                # The checks and the comparison read the new video by name, so
                # it is sealed and held unchanged until it is kept, and only
                # what was sealed is kept.
                try:
                    async with self._store.sealed_tool_output(output) as seal:
                        measured = await check_exact_trim(output.path, plan, ffprobe)
                        await asyncio.to_thread(
                            self._report, job_id, claim, "Comparing it with the original"
                        )
                        closeness, checked_from = await compare_with_source(
                            copy, output.path, plan, ffmpeg, fed
                        )
                        record = exact_trim_record(
                            artifact,
                            plan,
                            measured,
                            closeness,
                            fed,
                            checked_from,
                            ffmpeg.record(),
                            ffprobe.record(),
                        )
                        name = _trim_name(artifact, record, ".mp4")

                        def store_video(session: Session) -> Artifact:
                            return self._store.ingest_tool_output(
                                session,
                                output,
                                seal=seal,
                                kind=ArtifactKind.VIDEO,
                                media_type="video/mp4",
                                original_name=name,
                                metadata={"video_trim": record},
                            )

                        done = _OUTCOMES[request.action].done
                        await _write(lambda: self._finish(job_id, claim, store_video, record, done))
                except ToolOutputChanged:
                    raise TrimRefused("video-trim-output-mismatch") from None

    def _accepted(
        self, job_id: str, claim: JobClaim
    ) -> tuple[FrameRequest | TrimRequest, Artifact]:
        with SessionLocal() as session:
            job = _current(session, job_id, claim)
            if job is None:
                raise asyncio.CancelledError
            action = job.payload_json.get("action") if isinstance(job.payload_json, dict) else None
            try:
                request = _REQUEST.validate_python(job.payload_json)
            except ValidationError:
                raise _SourceChanged(action) from None
            artifact = session.get(Artifact, request.source_artifact_id)
            if artifact is None or artifact.sha256 != request.source_sha256:
                raise _SourceChanged(action)
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
        store_result: Callable[[Session], Artifact],
        record: dict[str, Any],
        stage: str,
    ) -> None:
        """Keep what the job made, then complete the job with its record.

        What was made is stored first, in its own transaction, so no file is
        written while the job row's writer is held. A result whose job then
        turns out to be displaced has no library entry and nothing refers to
        it, so ordinary retention removes it.

        A result byte for byte the same as one stored before is that same
        artifact, which keeps the metadata it was first stored with; the job's
        result is therefore the record of this job.
        """

        with SessionLocal() as session:
            result = store_result(session)
            session.commit()
            result_id = result.id
        with SessionLocal() as session:
            job = _current(session, job_id, claim)
            if job is None:
                return
            saved = session.get(Artifact, result_id)
            if saved is None:
                raise FileNotFoundError(result_id)
            entry = ensure_library_entry(session, saved)
            job.result_json = {
                "result_artifact_id": result_id,
                # False when the same file is already in Recently Deleted.
                "in_library": entry is not None and entry.state == "visible",
                **record,
            }
            job.status = JobStatus.COMPLETE.value
            job.completed_at = utcnow()
            job.progress = 1.0
            update_job_progress(job, stage=stage, overall_progress=1.0)
            session.commit()

    def _fail(
        self, job_id: str, claim: JobClaim, code: str, message: str, outcome: _Outcome
    ) -> None:
        with SessionLocal() as session:
            job = _current(session, job_id, claim)
            if job is None:
                return
            job.status = JobStatus.FAILED.value
            job.error = message
            job.result_json = {"failure_code": code}
            job.completed_at = utcnow()
            update_job_progress(job, stage=outcome.not_done, indeterminate=True)
            session.commit()


async def _write(work: Callable[[], None]) -> None:
    """Run a job's final write to its end, even when the job is cancelled meanwhile.

    The write runs in a thread, which a cancellation cannot stop. Waiting for
    it means a cancelled job's task ends only once nothing of the job runs.
    """

    await outlast_cancellation(asyncio.ensure_future(asyncio.to_thread(work)))


def _stage(session: Session, request: FrameRequest | TrimRequest) -> Job:
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


def _stem(source: Artifact) -> str:
    return PurePath(source.original_name or "video").stem[:80] or "video"


def _frame_name(source: Artifact, frame: DecodedFrame) -> str:
    return f"{_stem(source)} frame {frame.actual_seconds:.3f}s.png"


def _trim_name(source: Artifact, record: dict[str, Any], suffix: str) -> str:
    start, end = record["actual_start_seconds"], record["actual_end_seconds"]
    return f"{_stem(source)} trim {start:.3f}-{end:.3f}s{suffix}"
