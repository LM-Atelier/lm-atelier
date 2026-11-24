"""How a run ended is recorded faithfully while another writer holds the database.

A failure or a stop that the database keeps out is tried again, in the
execution and then in the background, for about ten minutes in all. These
tests pin what that record may and may not do: it is the outcome the run
really had, it lands on the job it ended and on no later retry of it, it
survives a second stop, it queues no retry once a stop has cut it short, and
once it lands the run is announced and its setup check finished as it would
have been at once.

The writer is held for real, from a second connection, as in
test_finished_renders_land_on_a_held_database.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS, wait_for_terminal_status, wait_until
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from test_finished_renders_land_on_a_held_database import (
    QUICK_PAUSES,
    _database,
    _executions_end,
    _hold_from_completion,
    _hold_terminal_writes,
    _job,
    _OtherWriter,
    _outcome_settled,
    _outcome_tasks,
    _record_events,
    _retry_quickly,
    _shown,
    _start_render,
)
from test_generation_retry import choose_retries, failing_media, read_run
from test_renders_outlast_a_busy_database import _short_busy_wait
from test_setup_verification import seed_ready_role

from local_lm import orchestrator as orchestrator_module
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.mock import MockMediaAdapter
from local_lm.artifacts import StagedArtifactFile
from local_lm.config import Settings
from local_lm.db import SessionLocal, database_is_contended
from local_lm.domain import JobKind, JobStatus, utcnow
from local_lm.image_edit_verification import (
    ImageEditVerificationJobPayload,
    image_edit_verification_job_id,
)
from local_lm.media_worker_use import MediaWorkerBusy
from local_lm.models import Artifact, Job, Run, SetupVerification
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.progress import update_job_progress
from local_lm.scheduler import ResourceScheduler

#: The owner a job moves to when a later attempt takes it.
LATER_OWNER = "a-later-attempt"


class _ThreadedWriter:
    """A second connection holding SQLite's writer, which any thread may let go.

    Let go from a timer thread, it frees the database on time even while the
    event loop is blocked in another write.
    """

    def __init__(self, database: Path) -> None:
        self._connection = sqlite3.connect(
            database, timeout=5, check_same_thread=False, isolation_level=None
        )
        self._lock = threading.Lock()
        self.released = threading.Event()

    def take(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    def release(self) -> None:
        with self._lock:
            if self.released.is_set():
                return
            with contextlib.suppress(sqlite3.OperationalError):
                self._connection.execute("ROLLBACK")
            self.released.set()

    def close(self) -> None:
        self.release()
        self._connection.close()


async def _job_left(job_id: str, status: str) -> Job:
    """Wait until the job is no longer in `status`, and return it."""

    async def read() -> Job:
        return _job(job_id)

    return await wait_until(
        read, lambda job: job.status != status, what=f"job {job_id} leaving {status}"
    )


def _supersede(database: Path, job_id: str) -> None:
    """Move the job to a later attempt another execution owns, as an expiry and a new claim do."""

    connection = sqlite3.connect(database, timeout=5)
    try:
        connection.execute(
            "UPDATE jobs SET attempt = attempt + 1, claim_owner = ? WHERE id = ?",
            (LATER_OWNER, job_id),
        )
        connection.commit()
    finally:
        connection.close()


def _run_reader(client: AsyncClient, run_id: str) -> Callable[[], Awaitable[dict[str, Any]]]:
    async def read() -> dict[str, Any]:
        return await read_run(client, run_id)

    return read


def _block_the_render(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Make every render wait, once it has reported progress, until it is stopped."""

    rendering = asyncio.Event()

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        yield MediaEvent(type="progress", progress=0.1, phase="loading", engine_sourced=True)
        rendering.set()
        await asyncio.sleep(PATIENCE_SECONDS)

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    return rendering


def _job_snapshots(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record each job snapshot the application publishes: the job, and the status it shows."""

    shown: list[tuple[str, str]] = []
    events = app.state.services.events
    publish = events.publish

    async def observed(event_type: str, entity_id: Any = None, payload: Any = None) -> Any:
        if event_type == "job.progress" and isinstance(payload, dict):
            shown.append((str(entity_id), str(payload["job"]["status"])))
        return await publish(event_type, entity_id, payload)

    monkeypatch.setattr(events, "publish", observed)
    return shown


def _hold_stops_from_the_first_try(
    monkeypatch: pytest.MonkeyPatch, writer: _OtherWriter, *, let_go_after: int | None
) -> list[str]:
    """Take the writer at a stopped execution's first try to record the stop, and keep it.

    With `let_go_after`, the writer is let go once that many tries have given
    up on it, so a later try lands in the execution itself.
    """

    tries: list[str] = []
    original = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator,
        session: Session,
        job_id: str,
        claim: Any,
        **values: Any,
    ) -> bool:
        if "status" in values:
            return original(self, session, job_id, claim, **values)
        if not tries:
            writer.take()
        try:
            landed = original(self, session, job_id, claim, **values)
        except OperationalError as error:
            if database_is_contended(error):
                tries.append("held")
                if let_go_after is not None and len(tries) >= let_go_after:
                    writer.release()
            raise
        tries.append("landed")
        return landed

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    return tries


async def test_a_failure_beside_a_queued_render_is_recorded_once_a_held_database_frees(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failure, and the retry it queues, outlast a claim release the database refused.

    Another writer takes the database as the render fails and keeps it past the
    release of the render's claim, then lets go two and a half seconds after
    that release gave up. Another render waits behind, and each of its passes
    clears abandoned claims, which would end this job as interrupted. Every
    wait is the application's own: the tries and their pauses, and the busy
    timeout of every other write.
    """

    writer = _ThreadedWriter(_database(settings))
    rendering = asyncio.Event()
    fail_now = asyncio.Event()
    renders: list[str] = []
    original_generate = MockMediaAdapter.generate

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        renders.append(request.run_id)
        if len(renders) == 1:
            yield MediaEvent(type="progress", progress=0.1, phase="loading", engine_sourced=True)
            rendering.set()
            await fail_now.wait()
            writer.take()
            raise RuntimeError("Temporary media engine failure")
        async for media_event in original_generate(self, request):
            yield media_event

    gave_up: list[str] = []
    original_release = ResourceScheduler._release_job

    async def release_job(self: ResourceScheduler, job_id: str, token: str, group: str) -> None:
        try:
            await original_release(self, job_id, token, group)
        except OperationalError:
            if not gave_up:
                gave_up.append(job_id)
                threading.Timer(2.5, writer.release).start()
            raise

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    monkeypatch.setattr(ResourceScheduler, "_release_job", release_job)
    await choose_retries(client, 1)

    async def run_to_its_end(run_id: str) -> dict[str, Any]:
        async def read() -> dict[str, Any]:
            return await read_run(client, run_id)

        return dict(await wait_for_terminal_status(read, what=f"run {run_id}", expected=None))

    async def scenario() -> tuple[str, str, str]:
        run_id, job_id = await _start_render(client, "failing-beside-a-queue")
        await rendering.wait()
        queued_run, _queued_job = await _start_render(client, "queued-behind-a-failure")
        fail_now.set()
        await run_to_its_end(run_id)
        await run_to_its_end(queued_run)
        return run_id, job_id, queued_run

    try:
        with caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
            run_id, job_id, queued_run = await asyncio.wait_for(scenario(), timeout=120)
    finally:
        writer.close()

    assert gave_up == [job_id], "the claim's release has to have met the held writer"
    run = await read_run(client, run_id)
    assert run["status"] == "complete", (run["status"], run["error"])
    assert run["provenance_json"]["failure_retries"]["used"] == 1
    assert _job(job_id).attempt == 2 and len(renders) == 3
    assert (await read_run(client, queued_run))["status"] == "complete"
    tries = [
        record.getMessage()
        for record in caplog.records
        if "could not record its failure on try" in record.getMessage()
    ]
    # The application's own schedule: eight tries, each waiting about a second.
    assert tries and all(" of 8: " in line for line in tries), tries
    waited = [float(line.rsplit(" after ", 1)[1].split(" ms", 1)[0]) for line in tries]
    assert all(500 <= each < 2000 for each in waited), tries


def _fail_as_the_database_is_taken(
    monkeypatch: pytest.MonkeyPatch, writer: _OtherWriter
) -> tuple[asyncio.Event, asyncio.Event, list[str], list[str]]:
    """Make the first render fail, on cue, as the other writer takes the database.

    Returns the event set once that render is rendering, the event that makes
    it fail, the runs asked to render, and the jobs whose claim release met the
    held writer.
    """

    rendering = asyncio.Event()
    fail_now = asyncio.Event()
    renders: list[str] = []
    original_generate = MockMediaAdapter.generate

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        renders.append(request.run_id)
        if len(renders) == 1:
            yield MediaEvent(type="progress", progress=0.1, phase="loading", engine_sourced=True)
            rendering.set()
            await fail_now.wait()
            writer.take()
            raise RuntimeError("Temporary media engine failure")
        async for media_event in original_generate(self, request):
            yield media_event

    gave_up: list[str] = []
    original_release = ResourceScheduler._release_job

    async def release_job(self: ResourceScheduler, job_id: str, token: str, group: str) -> None:
        try:
            await original_release(self, job_id, token, group)
        except OperationalError:
            gave_up.append(job_id)
            raise

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    monkeypatch.setattr(ResourceScheduler, "_release_job", release_job)
    return rendering, fail_now, renders, gave_up


async def test_a_failure_left_to_the_background_beside_a_queued_render_lands_as_it_happened(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failure every try in the execution met the held database lands later, and says so.

    The other writer keeps the database past the release of the render's claim
    and every try to record its failure, so the record goes on in the
    background. Another render waits behind, and each of its passes clears
    abandoned claims, which would end this job as interrupted and refuse the
    record. The failure lands instead, the retry it queues runs, and the log
    says it was recorded.
    """

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=1.0)
    rendering, fail_now, renders, gave_up = _fail_as_the_database_is_taken(monkeypatch, writer)
    await choose_retries(client, 1)
    try:
        with _short_busy_wait(), caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
            run_id, job_id = await _start_render(client, "failing-beside-a-queue")
            await asyncio.wait_for(rendering.wait(), timeout=PATIENCE_SECONDS)
            queued_run, _queued_job = await _start_render(client, "queued-behind-a-failure")
            fail_now.set()

            async def left_to_the_background() -> int:
                return len(_outcome_tasks(job_id))

            await wait_until(
                left_to_the_background,
                lambda count: count >= 1,
                what="the failure left to the background",
            )
            writer.release()
            run = await wait_for_terminal_status(
                _run_reader(client, run_id), what="the failed run", expected=None
            )
            queued = await wait_for_terminal_status(
                _run_reader(client, queued_run), what="the queued run", expected=None
            )
    finally:
        writer.release()

    assert gave_up == [job_id], "the claim's release has to have met the held writer"
    assert run["status"] == "complete", (run["status"], run["error"])
    assert run["provenance_json"]["failure_retries"]["used"] == 1
    assert _job(job_id).attempt == 2 and len(renders) == 3
    assert queued["status"] == "complete"
    said = [record.getMessage() for record in caplog.records]
    assert f"Job {job_id} recorded its failure once the database was free" in said, said
    assert not [line for line in said if "dropped its failure" in line], said


async def test_a_failure_the_background_gives_up_on_lets_its_claim_go(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Once the background stops trying to record a failure, the claim it kept is let go.

    The other writer keeps the database past the release of the render's claim
    and every try to record its failure, in the execution and in the
    background. Once the database frees, a pass of the render queued behind
    ends the job as interrupted, as it ends any claim whose release could not
    be written, rather than leave it running until the next start.
    """

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch)
    monkeypatch.setattr(orchestrator_module, "OUTCOME_RECORD_RETRY_DELAYS_SECONDS", (0.05,) * 3)
    rendering, fail_now, _renders, gave_up = _fail_as_the_database_is_taken(monkeypatch, writer)
    await choose_retries(client, 0)
    try:
        with _short_busy_wait(), caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
            _run_id, job_id = await _start_render(client, "failing-beside-a-queue")
            await asyncio.wait_for(rendering.wait(), timeout=PATIENCE_SECONDS)
            queued_run, _queued_job = await _start_render(client, "queued-behind-a-failure")
            fail_now.set()
            gave_up_saying = f"Job {job_id} could not record its failure: another writer held"

            async def background_gave_up() -> bool:
                return any(
                    record.getMessage().startswith(gave_up_saying) for record in caplog.records
                )

            await wait_until(background_gave_up, bool, what="the background giving up")
            await _outcome_settled(job_id)
            writer.release()
            job = await _job_left(job_id, JobStatus.RUNNING.value)
            queued = await wait_for_terminal_status(
                _run_reader(client, queued_run), what="the queued run", expected=None
            )
    finally:
        writer.release()

    assert gave_up == [job_id], "the claim's release has to have met the held writer"
    assert job.status == JobStatus.INTERRUPTED.value, (job.status, job.error)
    assert job.error == "The dispatcher lease expired before this job completed."
    assert queued["status"] == "complete"


async def test_a_queued_render_stopped_while_the_database_is_held_is_cancelled_once_it_frees(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A render still waiting for its turn is stopped while the database is held.

    The stop's own write fails, and so does every try of the stopped
    execution, so its record goes on trying later. Nothing has queued the job
    again since, so the record lands once the database frees, and clients are
    told the run was cancelled.
    """

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=0.3)
    published = _record_events(app, monkeypatch)
    busy = asyncio.Event()
    let_busy_finish = asyncio.Event()
    renders: list[str] = []
    original = MockMediaAdapter.generate

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        renders.append(request.run_id)
        if len(renders) == 1:
            yield MediaEvent(type="progress", progress=0.1, phase="loading", engine_sourced=True)
            busy.set()
            await let_busy_finish.wait()
        async for media_event in original(self, request):
            yield media_event

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    await choose_retries(client, 0)
    with _short_busy_wait(), caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
        busy_run, _busy_job = await _start_render(client, "busy-render")
        await asyncio.wait_for(busy.wait(), timeout=PATIENCE_SECONDS)
        run_id, job_id = await _start_render(client, "waiting-render")
        await wait_until(
            lambda: asyncio.sleep(0.05, result=_job(job_id)),
            lambda job: job.status == JobStatus.QUEUED.value and job.claim_owner is None,
            what="the second render waiting unclaimed",
        )

        writer.take()
        try:
            with pytest.raises(OperationalError):
                await client.post(f"/api/jobs/{job_id}/cancel")

            async def deferred() -> int:
                return len(_outcome_tasks(job_id))

            await wait_until(deferred, lambda count: count == 1, what="the stop left for later")
        finally:
            writer.release()
        job = await _job_left(job_id, JobStatus.QUEUED.value)
        await _outcome_settled(job_id)
        let_busy_finish.set()
        await wait_for_terminal_status(
            _run_reader(client, busy_run), what="the busy run", expected="complete"
        )

    assert job.status == JobStatus.CANCELLED.value, (job.status, job.error)
    assert (await read_run(client, run_id))["status"] == "cancelled"
    assert ("run.cancelled", run_id) in published
    assert renders == [busy_run], "the stopped render was asked to render"
    lines = [record.getMessage() for record in caplog.records]
    assert f"Job {job_id} recorded its cancellation once the database was free" in lines, lines


async def test_a_stop_recorded_late_leaves_alone_the_retry_asked_for_since(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A queued render is stopped while the database is held, then stopped again and retried.

    The first stop's own write fails, and so does every try of the stopped
    execution, so its record goes on trying later. It acted on the job as it
    was queued then; the retry queues the job again, and that is left alone.
    """

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=2.0)
    busy = asyncio.Event()
    let_busy_finish = asyncio.Event()
    renders: list[str] = []
    original = MockMediaAdapter.generate

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        renders.append(request.run_id)
        if len(renders) == 1:
            yield MediaEvent(type="progress", progress=0.1, phase="loading", engine_sourced=True)
            busy.set()
            await let_busy_finish.wait()
        async for media_event in original(self, request):
            yield media_event

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    await choose_retries(client, 0)
    with _short_busy_wait(), caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
        busy_run, _busy_job = await _start_render(client, "busy-render")
        await asyncio.wait_for(busy.wait(), timeout=PATIENCE_SECONDS)
        run_id, job_id = await _start_render(client, "waiting-render")
        waiting = await wait_until(
            lambda: asyncio.sleep(0.05, result=_job(job_id)),
            lambda job: job.status == JobStatus.QUEUED.value and job.claim_owner is None,
            what="the second render waiting unclaimed",
        )
        assert waiting.attempt == 0

        writer.take()
        try:
            with pytest.raises(OperationalError):
                await client.post(f"/api/jobs/{job_id}/cancel")

            async def deferred() -> int:
                return len(_outcome_tasks(job_id))

            await wait_until(deferred, lambda count: count == 1, what="the stop left for later")
        finally:
            writer.release()
        assert _job(job_id).status == JobStatus.QUEUED.value

        stopped = await client.post(f"/api/jobs/{job_id}/cancel")
        assert stopped.status_code == 200 and stopped.json()["status"] == "cancelled"
        retried = await client.post(f"/api/jobs/{job_id}/retry")
        assert retried.status_code == 200 and retried.json()["status"] == "queued"
        await _outcome_settled(job_id)

        assert _job(job_id).status == JobStatus.QUEUED.value, "the late stop ended the retry"
        let_busy_finish.set()
        for each in (busy_run, run_id):
            await wait_for_terminal_status(
                _run_reader(client, each), what=f"run {each}", expected="complete"
            )
    lines = [record.getMessage() for record in caplog.records]
    dropped = f"Job {job_id} dropped its cancellation once the database was free"
    assert f"{dropped}: the job was queued again since" in lines, lines
    assert not any(f"Job {job_id} recorded its cancellation" in line for line in lines)


async def test_a_late_failure_of_a_waiting_render_leaves_alone_the_retry_asked_for_since(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A render is refused its turn, and its failure meets a held database.

    It fails before any claim, so its failure may end only the job it saw
    queued. Every try of the failure meets the held writer, so it goes on
    trying later; meanwhile the job is stopped and queued again behind another
    render, and the late failure leaves that retry alone.
    """

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=0.3)
    busy = asyncio.Event()
    let_busy_finish = asyncio.Event()
    renders: list[str] = []
    original = MockMediaAdapter.generate

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        renders.append(request.run_id)
        if len(renders) == 1:
            yield MediaEvent(type="progress", progress=0.1, phase="loading", engine_sourced=True)
            busy.set()
            await let_busy_finish.wait()
        async for media_event in original(self, request):
            yield media_event

    refused: list[str] = []
    admit = ConversationOrchestrator._admit_media_run

    async def admit_or_refuse(
        self: ConversationOrchestrator, job_id: str, run_id: str
    ) -> str | None:
        if refused:
            return await admit(self, job_id, run_id)
        # ComfyUI is in someone else's use, and the render gives up waiting.
        refused.append(run_id)
        raise MediaWorkerBusy("prompts", late=False, limit_seconds=90)

    free = asyncio.Event()
    write_failure = ConversationOrchestrator._write_failure

    def held_until_free(self: ConversationOrchestrator, session: Session, *, outcome: Any) -> Any:
        if free.is_set():
            return write_failure(self, session, outcome=outcome)
        writer.take()
        try:
            return write_failure(self, session, outcome=outcome)
        finally:
            writer.release()

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    monkeypatch.setattr(ConversationOrchestrator, "_admit_media_run", admit_or_refuse)
    monkeypatch.setattr(ConversationOrchestrator, "_write_failure", held_until_free)
    await choose_retries(client, 0)
    with _short_busy_wait(), caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
        try:
            run_id, job_id = await _start_render(client, "refused-render")

            async def deferred() -> int:
                return len(_outcome_tasks(job_id))

            await wait_until(deferred, lambda count: count == 1, what="the failure left for later")
            assert _job(job_id).status == JobStatus.QUEUED.value

            stopped = await client.post(f"/api/jobs/{job_id}/cancel")
            assert stopped.status_code == 200 and stopped.json()["status"] == "cancelled"
            # Another render takes the turn, so the job queued again below waits.
            busy_run, _busy_job = await _start_render(client, "busy-render")
            await asyncio.wait_for(busy.wait(), timeout=PATIENCE_SECONDS)
            retried = await client.post(f"/api/jobs/{job_id}/retry")
            assert retried.status_code == 200 and retried.json()["status"] == "queued"
            await wait_until(
                lambda: asyncio.sleep(0.05, result=_job(job_id)),
                lambda job: job.status == JobStatus.QUEUED.value and job.claim_owner is None,
                what="the retry waiting unclaimed",
            )
        finally:
            free.set()
        await _outcome_settled(job_id)

        assert _job(job_id).status == JobStatus.QUEUED.value, "the late failure ended the retry"
        let_busy_finish.set()
        for each in (busy_run, run_id):
            await wait_for_terminal_status(
                _run_reader(client, each), what=f"run {each}", expected="complete"
            )
    assert refused == [run_id]
    assert renders == [busy_run, run_id]
    lines = [record.getMessage() for record in caplog.records]
    dropped = f"Job {job_id} dropped its failure once the database was free"
    assert f"{dropped}: the job was queued again since" in lines, lines


async def test_a_late_failure_gives_way_to_a_stop_pressed_since(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A render fails while the database is held through every try, then is stopped.

    The stop's own write lands, so the failure that goes on trying later finds
    the job cancelled, leaves it so, and says why.
    """

    writer = _OtherWriter(_database(settings))
    tries_per_write = _retry_quickly(monkeypatch, later=2.0)
    failing_media(app, monkeypatch, 1)
    published = _record_events(app, monkeypatch)
    tries = _hold_terminal_writes(
        monkeypatch,
        writer,
        status=JobStatus.FAILED.value,
        hold=lambda attempt: attempt < tries_per_write,
    )
    await choose_retries(client, 0)
    with _short_busy_wait(), caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
        run_id, job_id = await _start_render(client, "stopped-after-failing")

        async def deferred() -> int:
            return len(_outcome_tasks(job_id))

        await wait_until(deferred, lambda count: count == 1, what="the failure left for later")
        stopped = await client.post(f"/api/jobs/{job_id}/cancel")
        assert stopped.status_code == 200 and stopped.json()["status"] == "cancelled"
        await _outcome_settled(job_id)

    assert tries[:tries_per_write] == [True] * tries_per_write and len(tries) > tries_per_write
    assert _job(job_id).status == JobStatus.CANCELLED.value
    assert (await read_run(client, run_id))["status"] == "cancelled"
    assert ("run.failed", run_id) not in published
    lines = [record.getMessage() for record in caplog.records]
    dropped = f"Job {job_id} dropped its failure once the database was free"
    assert f"{dropped}: a stop had already cancelled the job" in lines, lines


async def test_stopping_a_chats_work_announces_each_stop_once(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopping all of a chat's work ends its render, and that stop is announced once.

    The stopped execution records the stop itself, as every stopped execution
    does, and leaves announcing it to the request that stopped the chat's work.
    """

    rendering = _block_the_render(monkeypatch)
    published = _record_events(app, monkeypatch)
    run_id, job_id = await _start_render(client, "stopped-with-its-chat")
    await asyncio.wait_for(rendering.wait(), timeout=PATIENCE_SECONDS)
    chat_id = (await read_run(client, run_id))["chat_id"]

    stopped = await client.post(f"/api/chats/{chat_id}/cancel-all")
    assert stopped.status_code == 204, stopped.text
    await _executions_end(app)
    await _outcome_settled(job_id)

    assert _job(job_id).status == JobStatus.CANCELLED.value
    assert published.count(("run.cancelled", run_id)) == 1, published


async def test_a_stop_while_a_failure_waits_for_a_held_database_still_records_the_failure(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render fails, and is stopped while its failure waits out a held database.

    The stop's own write fails too. The failure record goes on trying, so the
    job does not read running once the database is free.
    """

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=0.3)
    failing_media(app, monkeypatch, 1)
    tries: list[bool] = []
    original = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator,
        session: Session,
        job_id: str,
        claim: Any,
        **values: Any,
    ) -> bool:
        if values.get("status") == JobStatus.FAILED.value and not tries:
            # Taken at the failure's first try and kept.
            writer.take()
        tries.append(writer.holding)
        return original(self, session, job_id, claim, **values)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    await choose_retries(client, 0)
    orchestrator = app.state.services.orchestrator
    try:
        with _short_busy_wait():
            run_id, job_id = await _start_render(client, "stopped-while-failing")
            await wait_until(
                lambda: asyncio.sleep(0.01, result=len(tries)),
                lambda count: count >= 1,
                what="the failure's first try",
            )
            execution = orchestrator._tasks[job_id]
            with pytest.raises(OperationalError):
                await client.post(f"/api/jobs/{job_id}/cancel")
            await asyncio.wait({execution}, timeout=PATIENCE_SECONDS)
            # Ended by the stop, in the middle of the failure's tries, rather
            # than by running out of them.
            assert execution.cancelled()
            assert _job(job_id).status == JobStatus.RUNNING.value
            writer.release()
            ended = await _job_left(job_id, JobStatus.RUNNING.value)
    finally:
        writer.release()

    assert ended.status == JobStatus.FAILED.value
    assert ended.error == "Temporary media engine failure"
    run = await read_run(client, run_id)
    assert run["status"] == "failed"


async def test_a_stop_while_a_failure_waits_withdraws_the_retry_it_would_have_queued(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render with retries left fails, and is stopped while its failure waits out a held database.

    The stop's own write fails too. The failure is still what is recorded,
    with its own error, but not the retry it would have queued: nothing runs
    again after the stop.
    """

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=0.3)
    renders = failing_media(app, monkeypatch, 1)
    tries: list[bool] = []
    original = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator,
        session: Session,
        job_id: str,
        claim: Any,
        **values: Any,
    ) -> bool:
        if values.get("status") == JobStatus.FAILED.value and not tries:
            # Taken at the failure's first try and kept.
            writer.take()
        tries.append(writer.holding)
        return original(self, session, job_id, claim, **values)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    allowance = await client.get("/api/settings/generation-retries")
    assert allowance.json()["max_retries"] == 3, "the allowance a new installation starts with"
    orchestrator = app.state.services.orchestrator
    try:
        with _short_busy_wait():
            run_id, job_id = await _start_render(client, "stopped-while-failing-with-retries")
            await wait_until(
                lambda: asyncio.sleep(0.01, result=len(tries)),
                lambda count: count >= 1,
                what="the failure's first try",
            )
            execution = orchestrator._tasks[job_id]
            with pytest.raises(OperationalError):
                await client.post(f"/api/jobs/{job_id}/cancel")
            await asyncio.wait({execution}, timeout=PATIENCE_SECONDS)
            assert execution.cancelled()
            writer.release()
            ended = await _job_left(job_id, JobStatus.RUNNING.value)
            await _outcome_settled(job_id)
    finally:
        writer.release()

    assert ended.status == JobStatus.FAILED.value, (ended.status, ended.error)
    assert ended.error == "Temporary media engine failure"
    assert ended.attempt == 1
    run = await read_run(client, run_id)
    assert run["status"] == "failed"
    assert run["provenance_json"]["failure_retries"] == {"limit": 3, "used": 0, "pending": False}
    assert len(renders) == 1, "the render was asked for again after the stop"
    assert job_id not in orchestrator._tasks


async def test_a_stop_refused_while_a_failure_waits_in_the_background_withdraws_its_retry(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render with retries left fails, and every try of its failure meets a held database.

    The failure goes on trying in the background once its execution has
    ended. Then the render is stopped, and the stop's own write is refused too.
    The failure is still recorded, with its own error, once the database
    frees, but not the retry it would have queued: nothing runs again after
    the stop.
    """

    writer = _OtherWriter(_database(settings))
    tries_per_write = _retry_quickly(monkeypatch, later=0.3)
    renders = failing_media(app, monkeypatch, 1)
    tries: list[bool] = []
    original = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator,
        session: Session,
        job_id: str,
        claim: Any,
        **values: Any,
    ) -> bool:
        if values.get("status") == JobStatus.FAILED.value and not tries:
            # Taken at the failure's first try and kept until the test lets go.
            writer.take()
        tries.append(writer.holding)
        return original(self, session, job_id, claim, **values)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    allowance = await client.get("/api/settings/generation-retries")
    assert allowance.json()["max_retries"] == 3, "the allowance a new installation starts with"
    orchestrator = app.state.services.orchestrator
    try:
        with _short_busy_wait():
            run_id, job_id = await _start_render(client, "stopped-after-its-execution-ended")

            async def left_to_the_background() -> int:
                return len(_outcome_tasks(job_id))

            await wait_until(
                left_to_the_background, lambda count: count == 1, what="the failure left for later"
            )
            await _executions_end(app)
            assert len(tries) >= tries_per_write and all(tries[:tries_per_write]), tries
            assert _job(job_id).status == JobStatus.RUNNING.value
            with pytest.raises(OperationalError):
                await client.post(f"/api/jobs/{job_id}/cancel")
            writer.release()
            ended = await _job_left(job_id, JobStatus.RUNNING.value)
            await _outcome_settled(job_id)
    finally:
        writer.release()

    assert ended.status == JobStatus.FAILED.value, (ended.status, ended.error)
    assert ended.error == "Temporary media engine failure"
    assert ended.attempt == 1
    run = await read_run(client, run_id)
    assert run["status"] == "failed"
    assert run["provenance_json"]["failure_retries"] == {"limit": 3, "used": 0, "pending": False}
    assert len(renders) == 1, "the render was asked for again after the stop"
    assert job_id not in orchestrator._tasks


async def test_an_image_edit_check_that_gave_way_is_recorded_as_interrupted_however_late(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that gives way to new work was interrupted, not cancelled.

    That is decided when it gives way. Its record is held out through every
    try and lands in the background, after the work that displaced it has
    finished, and still says so. The check has no run to announce, so the
    job as it now reads is what clients are shown.
    """

    writer = _OtherWriter(_database(settings))
    tries_per_write = _retry_quickly(monkeypatch, later=0.3)
    snapshots = _job_snapshots(app, monkeypatch)
    await choose_retries(client, 0)
    source_run_id, source_job_id = await _start_render(client, "check-source")
    await _executions_end(app)
    source = await read_run(client, source_run_id)
    assert source["status"] == "complete", source["error"]
    picture = source["provenance_json"]["outputs"][0]["artifact_id"]

    # A check of that render, queued as a finished edit queues one.
    check_id = image_edit_verification_job_id(source_run_id)
    payload = ImageEditVerificationJobPayload(
        chat_id=source["chat_id"],
        source_run_id=source_run_id,
        source_job_id=source_job_id,
        source_artifact_id=picture,
        result_artifact_id=picture,
        vision_profile_id="profile-vision",
    )
    with SessionLocal() as session:
        now = utcnow()
        check = Job(
            id=check_id,
            kind=JobKind.EDIT_VERIFY.value,
            status=JobStatus.QUEUED.value,
            progress=0.0,
            progress_json={},
            queue_resource="interactive_compute",
            queue_group="primary",
            queue_priority=-10,
            queue_ticket=check_id,
            enqueued_at=now,
            payload_json=payload.model_dump(mode="json"),
            cancellable=True,
        )
        update_job_progress(
            check,
            stage="Waiting to check image edit",
            queue_resource=check.queue_resource,
            indeterminate=True,
            now=now,
        )
        session.add(check)
        session.commit()

    checking = asyncio.Event()

    async def assess(self: ConversationOrchestrator, job_id: str, claim: Any) -> None:
        checking.set()
        await asyncio.sleep(PATIENCE_SECONDS)

    monkeypatch.setattr(ConversationOrchestrator, "_execute_image_edit_verification", assess)
    orchestrator = app.state.services.orchestrator
    orchestrator.start(check_id, None)
    await asyncio.wait_for(checking.wait(), timeout=PATIENCE_SECONDS)

    free = asyncio.Event()
    tries = _hold_terminal_writes(
        monkeypatch, writer, status=None, hold=lambda _attempt: not free.is_set()
    )
    run_id, _job_id = await _start_render(client, "displacing-work")
    await wait_for_terminal_status(
        lambda: read_run(client, run_id), what="the displacing render", expected="complete"
    )
    assert len(tries) >= tries_per_write and all(tries), tries
    assert _job(check_id).status == JobStatus.RUNNING.value

    free.set()
    await _job_left(check_id, JobStatus.RUNNING.value)
    await _outcome_settled(check_id)

    check_after = _job(check_id)
    record = (await read_run(client, source_run_id))["provenance_json"].get(
        "image_edit_verification"
    )
    assert check_after.status == JobStatus.COMPLETE.value, check_after.result_json
    assert check_after.result_json["reason"] == "assessment_interrupted"
    assert record is not None and record["reason"] == "assessment_interrupted"
    assert (check_id, JobStatus.COMPLETE.value) in snapshots, snapshots


async def test_a_second_stop_while_the_first_is_being_recorded_keeps_the_record(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render is stopped twice while its first stop waits out a held database.

    The second stop arrives in a pause between the first stop's tries. It does
    not end the record: the stop is recorded once the database is free.
    """

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=0.3)
    monkeypatch.setattr(orchestrator_module, "HELD_DATABASE_RETRY_DELAYS_SECONDS", (0.3,) * 7)
    rendering = _block_the_render(monkeypatch)
    tries = _hold_stops_from_the_first_try(monkeypatch, writer, let_go_after=None)
    orchestrator = app.state.services.orchestrator
    try:
        with _short_busy_wait():
            run_id, job_id = await _start_render(client, "stopped-twice")
            await asyncio.wait_for(rendering.wait(), timeout=PATIENCE_SECONDS)
            execution = orchestrator._tasks[job_id]
            execution.cancel()
            await wait_until(
                lambda: asyncio.sleep(0.01, result=len(tries)),
                lambda count: count >= 1,
                what="the first stop's first try",
            )
            execution.cancel()
            await asyncio.wait({execution}, timeout=PATIENCE_SECONDS)
            assert execution.cancelled()
            assert _job(job_id).status == JobStatus.RUNNING.value
            writer.release()
            ended = await _job_left(job_id, JobStatus.RUNNING.value)
    finally:
        writer.release()

    assert ended.status == JobStatus.CANCELLED.value
    assert (await read_run(client, run_id))["status"] == "cancelled"


@pytest.mark.parametrize("late", ["in-the-execution", "in-the-background"])
async def test_a_stop_recorded_late_finishes_its_setup_check_and_announces_it(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    late: str,
) -> None:
    """A setup check is stopped while the database is held, so the stop's own write fails.

    The stopped execution records the stop instead, on a later try or in the
    background, and then does what the stop would have: the check is finished,
    and the job as it now reads and the run's end are announced.
    """

    seed_ready_role(settings, "image", operation="text_to_image")
    writer = _OtherWriter(_database(settings))
    tries_per_write = _retry_quickly(monkeypatch, later=0.3)
    rendering = _block_the_render(monkeypatch)
    published = _record_events(app, monkeypatch)
    snapshots = _job_snapshots(app, monkeypatch)
    tries = _hold_stops_from_the_first_try(
        monkeypatch, writer, let_go_after=2 if late == "in-the-execution" else None
    )
    try:
        with _short_busy_wait():
            assert (await client.post("/api/setup/verify/image")).status_code == 202
            await asyncio.wait_for(rendering.wait(), timeout=PATIENCE_SECONDS)
            with SessionLocal() as session:
                verification = session.scalar(select(SetupVerification))
                assert verification is not None and verification.job_id is not None
                job_id, run_id = verification.job_id, verification.run_id
            writer.take()
            with pytest.raises(OperationalError):
                await client.post(f"/api/jobs/{job_id}/cancel")
            if late == "in-the-background":
                await wait_until(
                    lambda: asyncio.sleep(0.01, result=len(tries)),
                    lambda count: count >= tries_per_write,
                    what="every try of the stop",
                )
                await _executions_end(app)
                writer.release()

            async def verification_state() -> str:
                with SessionLocal() as session:
                    current = session.scalar(select(SetupVerification))
                    assert current is not None
                    return current.state

            await wait_until(
                verification_state,
                lambda state: state not in {"queued", "running"},
                what="the setup check finishing",
            )
    finally:
        writer.release()

    with SessionLocal() as session:
        verification = session.scalar(select(SetupVerification))
        assert verification is not None
        assert verification.state == "failed"
        assert verification.failure_code == "generation_cancelled"
        assert session.scalars(select(Job)).all() == []
    assert ("run.cancelled", run_id) in published
    # The stop's own request published nothing; only the record that landed
    # can have shown the job cancelled.
    assert (job_id, JobStatus.CANCELLED.value) in snapshots, snapshots
    assert any(kind == "setup.verification.completed" for kind, _entity in published)
    assert "landed" in tries


async def test_the_pauses_between_held_tries_leave_the_event_loop_free(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Other work goes on while a save that met a held database pauses before its next try.

    The pauses here are a second long, and the event loop must keep serving
    everything else through them.
    """

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch)
    monkeypatch.setattr(orchestrator_module, "HELD_DATABASE_RETRY_DELAYS_SECONDS", (1.0,) * 7)
    _hold_from_completion(monkeypatch, writer)
    refused: list[str] = []
    original = ConversationOrchestrator._claim_owns_row

    def claim_owns_row(
        self: ConversationOrchestrator, session: Session, job_id: str, claim: Any
    ) -> bool:
        if not writer.holding:
            return original(self, session, job_id, claim)
        try:
            return original(self, session, job_id, claim)
        except OperationalError:
            refused.append(job_id)
            if len(refused) >= 2:
                writer.release()
            raise

    monkeypatch.setattr(ConversationOrchestrator, "_claim_owns_row", claim_owns_row)
    await choose_retries(client, 0)
    stalls: list[float] = []

    async def tick() -> None:
        last = asyncio.get_running_loop().time()
        while True:
            await asyncio.sleep(0.01)
            now = asyncio.get_running_loop().time()
            stalls.append(now - last - 0.01)
            last = now

    ticking = asyncio.create_task(tick())
    try:
        with _short_busy_wait():
            run_id, _job_id = await _start_render(client, "pausing-save")
            await wait_for_terminal_status(
                lambda: read_run(client, run_id), what="the render", expected="complete"
            )
    finally:
        ticking.cancel()
        await asyncio.gather(ticking, return_exceptions=True)
        writer.release()

    assert len(refused) == 2, "the save has to have paused twice"
    assert stalls and max(stalls) < 0.7, f"the event loop stopped for {max(stalls):.2f}s"


async def test_a_failure_recorded_later_finishes_its_setup_check(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A setup check's render fails while the database is held through every try.

    The failure lands in the background, and the check is finished with it.
    """

    seed_ready_role(settings, "image", operation="text_to_image")
    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=0.3)
    failing_media(app, monkeypatch, 1)
    free = asyncio.Event()
    tries = _hold_terminal_writes(
        monkeypatch,
        writer,
        status=JobStatus.FAILED.value,
        hold=lambda _attempt: not free.is_set(),
    )
    with _short_busy_wait():
        assert (await client.post("/api/setup/verify/image")).status_code == 202
        await _executions_end(app)
        assert len(tries) >= len(QUICK_PAUSES) + 1 and all(tries)
        free.set()

        async def state() -> str:
            with SessionLocal() as session:
                current = session.scalar(select(SetupVerification))
                assert current is not None
                return current.state

        await wait_until(
            state,
            lambda current: current not in {"queued", "running"},
            what="the setup check finishing",
        )

    with SessionLocal() as session:
        verification = session.scalar(select(SetupVerification))
        assert verification is not None
        assert verification.state == "failed"
        assert verification.failure_code == "generation_failed"
        assert session.scalars(select(Job)).all() == []


async def test_a_failure_refused_once_its_job_moved_on_is_not_announced(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While a failure pauses between tries, a later attempt takes the job.

    The next try is refused, and nothing tells clients the run failed: the run
    belongs to the attempt that took it.
    """

    database = _database(settings)
    writer = _OtherWriter(database)
    _retry_quickly(monkeypatch)
    failing_media(app, monkeypatch, 1)
    published = _record_events(app, monkeypatch)
    tries: list[bool] = []
    original = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator,
        session: Session,
        job_id: str,
        claim: Any,
        **values: Any,
    ) -> bool:
        if values.get("status") != JobStatus.FAILED.value or tries:
            tries.append(False)
            return original(self, session, job_id, claim, **values)
        tries.append(True)
        writer.take()
        try:
            return original(self, session, job_id, claim, **values)
        finally:
            writer.release()
            _supersede(database, job_id)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    await choose_retries(client, 0)
    with _short_busy_wait():
        run_id, job_id = await _start_render(client, "failure-overtaken")
        await _executions_end(app)

    job = _job(job_id)
    assert tries[:2] == [True, False], "the failure has to have been tried again"
    assert (job.status, job.attempt, job.claim_owner) == ("running", 2, LATER_OWNER)
    assert ("run.failed", run_id) not in published
    assert ("run.retrying", run_id) not in published


async def test_a_completion_refused_once_its_job_moved_on_leaves_the_later_attempt_alone(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While the completion pauses between tries, a later attempt takes the job.

    The next try is refused: the run is not completed, and no outputs are
    shown for an attempt that is no longer this execution's.
    """

    database = _database(settings)
    writer = _OtherWriter(database)
    _retry_quickly(monkeypatch)
    tries: list[bool] = []
    original = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator,
        session: Session,
        job_id: str,
        claim: Any,
        **values: Any,
    ) -> bool:
        if values.get("status") != JobStatus.COMPLETE.value or tries:
            tries.append(False)
            return original(self, session, job_id, claim, **values)
        tries.append(True)
        writer.take()
        try:
            return original(self, session, job_id, claim, **values)
        finally:
            writer.release()
            _supersede(database, job_id)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    await choose_retries(client, 0)
    with _short_busy_wait():
        run_id, job_id = await _start_render(client, "completion-overtaken")
        await _executions_end(app)

    job = _job(job_id)
    assert tries[:2] == [True, False], "the completion has to have been tried again"
    assert (job.status, job.attempt, job.claim_owner) == ("running", 2, LATER_OWNER)
    assert not job.result_json
    run = await read_run(client, run_id)
    assert run["status"] == "running"
    assert "outputs" not in run["provenance_json"]
    # Only the streamed preview is shown, never the finished picture.
    assert not [
        part
        for part in (await _shown(client, run))["parts"]
        if part["type"] == "image" and not part["metadata_json"].get("preview")
    ]


async def test_a_start_refused_once_its_job_moved_on_does_not_start_the_run(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While the start of a run pauses between tries, a later attempt takes the job.

    The next try is refused: this execution neither marks the run running nor
    asks the engine for anything.
    """

    database = _database(settings)
    writer = _OtherWriter(database)
    _retry_quickly(monkeypatch)
    renders: list[str] = []

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        renders.append(request.run_id)
        yield MediaEvent(type="cancelled", phase="cancelled")

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    original = ConversationOrchestrator._resolve_step_inputs
    started: list[str] = []

    def resolve(session: Session, run: Run) -> None:
        # The start reads the run's inputs before it writes anything.
        if not started:
            writer.take()
        started.append(run.id)
        original(session, run)

    original_status = ConversationOrchestrator._set_work_status

    def set_work_status(session: Session, run: Run, status: str, **values: Any) -> None:
        # Read first: a write the held database refused leaves the run expired.
        run_id = run.id
        try:
            original_status(session, run, status, **values)
        except OperationalError:
            if writer.holding:
                writer.release()
                with SessionLocal() as other:
                    job_id = other.scalars(select(Job.id).where(Job.run_id == run_id)).one()
                _supersede(database, job_id)
            raise

    monkeypatch.setattr(ConversationOrchestrator, "_resolve_step_inputs", staticmethod(resolve))
    monkeypatch.setattr(ConversationOrchestrator, "_set_work_status", staticmethod(set_work_status))
    await choose_retries(client, 0)
    with _short_busy_wait():
        run_id, job_id = await _start_render(client, "start-overtaken")
        await _executions_end(app)

    assert started == [run_id, run_id], "the start has to have been tried again"
    job = _job(job_id)
    assert (job.attempt, job.claim_owner) == (2, LATER_OWNER)
    assert (await read_run(client, run_id))["status"] == "queued"
    assert renders == []


@pytest.mark.parametrize("derived", ["proxy", "poster"])
async def test_a_video_derivative_refused_once_its_job_moved_on_is_not_attached(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    derived: str,
) -> None:
    """While a video's proxy or poster pauses between tries, a later attempt takes the job.

    The next try is refused: the video is not pointed at it, and the render
    stops there.
    """

    database = _database(settings)
    writer = _OtherWriter(database)
    _retry_quickly(monkeypatch)
    artifacts = app.state.services.artifacts
    staged_path = artifacts.root / "overtaken-proxy.mp4"

    async def proxy(_artifact: Artifact) -> StagedArtifactFile | None:
        if derived != "proxy":
            return None
        staged_path.write_bytes(b"browser-playable-proxy")
        writer.take()
        return StagedArtifactFile(
            path=staged_path, media_type="video/mp4", original_name="overtaken-proxy.mp4"
        )

    async def poster(_artifact: Artifact) -> bytes | None:
        if derived != "poster":
            return None
        writer.take()
        return b"\xff\xd8\xff\xe0poster-frame"

    monkeypatch.setattr(artifacts, "browser_video_proxy", proxy)
    monkeypatch.setattr(artifacts, "video_poster", poster)
    checks: list[bool] = []
    original = ConversationOrchestrator._claim_owns_row

    def claim_owns_row(
        self: ConversationOrchestrator, session: Session, job_id: str, claim: Any
    ) -> bool:
        if not writer.holding:
            owned = original(self, session, job_id, claim)
            checks.append(owned)
            return owned
        try:
            return original(self, session, job_id, claim)
        finally:
            writer.release()
            _supersede(database, job_id)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_owns_row", claim_owns_row)
    await choose_retries(client, 0)
    with _short_busy_wait():
        run_id, job_id = await _start_render(client, f"{derived}-overtaken", mode="video")
        await _executions_end(app)

    assert checks and checks[-1] is False, "the derivative has to have been tried again"
    job = _job(job_id)
    assert (job.attempt, job.claim_owner) == (2, LATER_OWNER)
    run = await read_run(client, run_id)
    assert run["status"] == "running"
    with SessionLocal() as session:
        video = session.scalars(
            select(Artifact).where(Artifact.kind == "video").order_by(Artifact.created_at.desc())
        ).first()
        assert video is not None
        assert "browser_proxy_artifact_id" not in video.metadata_json
        assert "poster_artifact_id" not in video.metadata_json
    assert not staged_path.exists()


def test_the_retry_schedule_is_the_one_described() -> None:
    """The schedule the module describes: about a second per try, eight tries over about
    forty seconds, then tries in the background from five seconds to a minute apart for
    about ten minutes."""

    pauses = orchestrator_module.HELD_DATABASE_RETRY_DELAYS_SECONDS
    later = orchestrator_module.OUTCOME_RECORD_RETRY_DELAYS_SECONDS
    assert orchestrator_module.HELD_DATABASE_TRY_WAIT_MS == 1000
    assert len(pauses) + 1 == 8
    assert list(pauses) == sorted(pauses) and pauses[0] == 0.25 and pauses[-1] == 15.0
    assert 30 <= sum(pauses) <= 32
    assert later[0] == 5.0 and max(later) == 60.0 and list(later) == sorted(later)
    assert 600 <= sum(later) <= 630
