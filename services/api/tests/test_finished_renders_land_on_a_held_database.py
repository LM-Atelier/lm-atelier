"""A finished render, and every run's outcome, waits for a held database before giving up.

SQLite has a single writer. A write that finds another connection holding it
waits out the busy timeout and fails. Once the engine has finished, the
pictures it made exist only in memory, and a run whose outcome cannot be
written goes on reading running. So these writes wait for the database
instead: a try that meets the held writer gives up after a second, holds
nothing and blocks nothing while it pauses, and the next try starts afresh.
A write gets eight tries, about forty seconds, and a run whose own writes run
out of them fails in plain words; an outcome still unwritten then goes on
trying in the background for about ten minutes.

The writer is held for real, from a second connection. Each held try, and the
application's other writes, give up after a tenth of a second instead, except
where a test measures how long a try waits, and the pauses between tries are
always shortened.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS, wait_until
from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from test_generation_retry import choose_retries, failing_media, read_run
from test_renders_outlast_a_busy_database import _short_busy_wait
from test_scheduler import _real_sqlite_error
from test_setup_verification import seed_ready_role

from local_lm import db as database_module
from local_lm import orchestrator as orchestrator_module
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.mock import MockMediaAdapter
from local_lm.artifacts import StagedArtifactFile
from local_lm.config import Settings
from local_lm.db import SessionLocal, database_is_contended
from local_lm.domain import JobStatus
from local_lm.models import Artifact, Job, Run, SetupVerification
from local_lm.orchestrator import ConversationOrchestrator

#: What a run says when another writer held the database through every try.
HELD_TOO_LONG = "This run stopped because another task held the database for too long. Try again."

#: Short pauses between the tries a write gets.
QUICK_PAUSES = (0.02,) * 7

#: How long a held try waits in these tests, as `_short_busy_wait` sets for
#: every other write.
QUICK_TRY_WAIT_MS = 100


def _database(settings: Settings) -> Path:
    return settings.state_dir / "local-lm.sqlite3"


def _retry_quickly(monkeypatch: pytest.MonkeyPatch, *, later: float = PATIENCE_SECONDS) -> int:
    """Shorten each try and the pauses between them, and return how many tries a write gets.

    A failure or a cancellation that outlasts those tries waits `later`
    between each further try. By default that is longer than any test waits,
    so a test about the tries an execution makes is never passed by a later one.
    """

    monkeypatch.setattr(
        orchestrator_module, "HELD_DATABASE_TRY_WAIT_MS", QUICK_TRY_WAIT_MS, raising=False
    )
    monkeypatch.setattr(
        orchestrator_module, "HELD_DATABASE_RETRY_DELAYS_SECONDS", QUICK_PAUSES, raising=False
    )
    monkeypatch.setattr(
        orchestrator_module, "OUTCOME_RECORD_RETRY_DELAYS_SECONDS", (later,) * 100, raising=False
    )
    return len(QUICK_PAUSES) + 1


class _OtherWriter:
    """A second connection that takes SQLite's writer and lets it go when told."""

    def __init__(self, database: Path) -> None:
        self._database = database
        self._connection: sqlite3.Connection | None = None

    @property
    def holding(self) -> bool:
        return self._connection is not None

    def take(self) -> None:
        if self._connection is not None:
            return
        connection = sqlite3.connect(self._database, timeout=5)
        connection.execute("BEGIN IMMEDIATE")
        self._connection = connection

    def release(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.rollback()
            connection.close()


def _let_go_after_refused_claims(
    monkeypatch: pytest.MonkeyPatch, writer: _OtherWriter, refusals: int
) -> list[str]:
    """Let the other writer go once `refusals` claim-checked writes have given up on it.

    A render checks its claim first in every write it stores something with,
    so this counts the tries of exactly those writes.
    """

    refused: list[str] = []
    original = ConversationOrchestrator._claim_owns_row

    def claim_owns_row(
        self: ConversationOrchestrator, session: Session, job_id: str, claim: Any
    ) -> bool:
        if not writer.holding:
            return original(self, session, job_id, claim)
        try:
            return original(self, session, job_id, claim)
        except OperationalError as error:
            if database_is_contended(error):
                refused.append(job_id)
                if len(refused) >= refusals:
                    writer.release()
            raise

    monkeypatch.setattr(ConversationOrchestrator, "_claim_owns_row", claim_owns_row)
    return refused


@contextmanager
def _let_go_at_first_refusal(writer: _OtherWriter) -> Iterator[list[str]]:
    """Let the other writer go as soon as one of the application's writes gives up on it.

    The writer is taken just before the write under test, with nothing awaited
    in between, so that write is the first to meet it.
    """

    refused: list[str] = []
    engine = database_module.engine

    def gave_up(context: Any) -> None:
        code = getattr(context.original_exception, "sqlite_errorcode", None)
        if writer.holding and isinstance(code, int) and code & 0xFF == sqlite3.SQLITE_BUSY:
            refused.append(str(context.statement))
            writer.release()

    event.listen(engine, "handle_error", gave_up)
    try:
        yield refused
    finally:
        event.remove(engine, "handle_error", gave_up)


def _hold_terminal_writes(
    monkeypatch: pytest.MonkeyPatch,
    writer: _OtherWriter,
    *,
    status: str | None,
    hold: Callable[[int], bool],
) -> list[bool]:
    """Hold the other writer across the chosen tries to end a job, and record each try.

    `status` picks the writes: a failure names its status, while stopping a
    job only asserts its attempt and names none. `hold` is asked about each
    try in turn, counted from zero.
    """

    tries: list[bool] = []
    original = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator,
        session: Session,
        job_id: str,
        claim: Any,
        **values: Any,
    ) -> bool:
        if values.get("status") != status:
            return original(self, session, job_id, claim, **values)
        holding = hold(len(tries))
        tries.append(holding)
        if not holding:
            return original(self, session, job_id, claim, **values)
        writer.take()
        try:
            return original(self, session, job_id, claim, **values)
        finally:
            writer.release()

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    return tries


def _hold_from_completion(monkeypatch: pytest.MonkeyPatch, writer: _OtherWriter) -> list[str]:
    """Take the other writer just before the engine reports its finished render."""

    renders: list[str] = []
    original = MockMediaAdapter.generate

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        renders.append(request.run_id)
        async for media_event in original(self, request):
            if media_event.type == "complete":
                writer.take()
            yield media_event

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    return renders


def _count_renders(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record each render the engine is asked for."""

    renders: list[str] = []
    original = MockMediaAdapter.generate

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        renders.append(request.run_id)
        async for media_event in original(self, request):
            yield media_event

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    return renders


async def _start_render(client: AsyncClient, key: str, *, mode: str = "image") -> tuple[str, str]:
    """Ask for a render without waiting for it: the run's id and its job's id."""

    chat = await client.post("/api/chats", json={"title": "Render beside a held database"})
    assert chat.status_code == 201
    text = (
        "Create a short video of a turning gear"
        if mode == "video"
        else "Create an image of a purple observatory"
    )
    response = await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={"text": text, "mode": mode, "idempotency_key": key},
    )
    assert response.status_code == 202, response.text
    run_id = str(response.json()["run"]["id"])
    with SessionLocal() as session:
        job_id = session.scalars(select(Job.id).where(Job.run_id == run_id)).one()
    return run_id, job_id


async def _executions_end(app: FastAPI) -> None:
    """Wait until no execution is left, however each one ended."""

    tasks = app.state.services.orchestrator._tasks

    async def running() -> int:
        return len(tasks)

    await wait_until(running, lambda count: count == 0, what="every execution ending")


def _job(job_id: str) -> Job:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        return job


#: The ways an execution's outcome is recorded after the execution ends.
OUTCOME_KINDS = ("failure", "cancellation", "interruption")


def _outcome_tasks(job_id: str) -> list[asyncio.Task[Any]]:
    """The tasks still recording, or finishing, how this job's execution ended."""

    names = {f"local-lm-{kind}-{job_id}" for kind in OUTCOME_KINDS}
    return [task for task in asyncio.all_tasks() if task.get_name() in names and not task.done()]


async def _outcome_settled(job_id: str) -> None:
    async def pending() -> int:
        return len(_outcome_tasks(job_id))

    await wait_until(pending, lambda count: count == 0, what=f"the outcome of job {job_id}")


def _record_events(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    """Record each event the application publishes, with the entity it is about."""

    published: list[tuple[str, Any]] = []
    events = app.state.services.events
    publish = events.publish

    async def observed(event_type: str, entity_id: Any = None, payload: Any = None) -> Any:
        published.append((event_type, entity_id))
        return await publish(event_type, entity_id, payload)

    monkeypatch.setattr(events, "publish", observed)
    return published


async def _shown(client: AsyncClient, run: dict[str, Any]) -> dict[str, Any]:
    response = await client.get(f"/api/messages/{run['assistant_message_id']}")
    assert response.status_code == 200
    message = response.json()
    assert isinstance(message, dict)
    return message


async def test_a_finished_render_is_saved_once_a_database_held_at_its_completion_frees(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine finished, then another writer held the database past the busy timeout.

    The first try to save the picture gives up on it. The next, once it is
    free, saves the picture, and the run completes on its only render.
    """

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch)
    renders = _hold_from_completion(monkeypatch, writer)
    refused = _let_go_after_refused_claims(monkeypatch, writer, refusals=1)
    await choose_retries(client, 0)
    try:
        with _short_busy_wait():
            run_id, job_id = await _start_render(client, "held-at-completion")
            await _executions_end(app)
    finally:
        writer.release()

    run = await read_run(client, run_id)
    assert run["status"] == "complete", run["error"]
    assert len(refused) == 1, "the save has to have met the held database"
    job = _job(job_id)
    assert job.status == JobStatus.COMPLETE.value
    assert job.attempt == 1 and renders == [run_id], "the finished render was lost and made again"
    picture = next(part for part in (await _shown(client, run))["parts"] if part["type"] == "image")
    content = await client.get(f"/api/artifacts/{picture['artifact_id']}/content")
    assert content.status_code == 200
    assert run["provenance_json"]["outputs"][0]["artifact_id"] == picture["artifact_id"]


async def test_a_try_that_meets_a_held_database_holds_up_everything_else_for_about_a_second(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A try waits for the other writer on the event loop, which serves everything else.

    So it gives up after about a second, not the five of the busy timeout, and
    the pauses between tries wait without blocking. Each wait for the writer
    here is the application's own; only the pauses are shortened.
    """

    writer = _OtherWriter(_database(settings))
    monkeypatch.setattr(
        orchestrator_module, "HELD_DATABASE_RETRY_DELAYS_SECONDS", QUICK_PAUSES, raising=False
    )
    _hold_from_completion(monkeypatch, writer)
    refused = _let_go_after_refused_claims(monkeypatch, writer, refusals=1)
    await choose_retries(client, 0)
    try:
        with caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
            run_id, _job_id = await _start_render(client, "held-try-gives-up-soon")
            await _executions_end(app)
    finally:
        writer.release()

    run = await read_run(client, run_id)
    assert run["status"] == "complete", run["error"]
    assert len(refused) == 1, "the save has to have met the held database"
    tries = [
        record.getMessage()
        for record in caplog.records
        if "could not record its output" in record.getMessage()
    ]
    assert len(tries) == 1, tries
    # How long the try waited for the other writer, which is how long it held
    # the event loop.
    waited_ms = float(tries[0].rsplit(" after ", 1)[1].split(" ms", 1)[0])
    assert 500 <= waited_ms < 2500, tries[0]


async def test_a_render_completes_once_a_database_held_as_it_completes_frees(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The picture is saved; the write that completes the run then meets the held writer."""

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch)
    renders = _count_renders(monkeypatch)
    tries = _hold_terminal_writes(
        monkeypatch, writer, status=JobStatus.COMPLETE.value, hold=lambda attempt: attempt < 1
    )
    await choose_retries(client, 0)
    with _short_busy_wait():
        run_id, job_id = await _start_render(client, "held-as-it-completes")
        await _executions_end(app)

    run = await read_run(client, run_id)
    assert run["status"] == "complete", run["error"]
    assert tries == [True, False], "the completion has to have met the held database, then landed"
    job = _job(job_id)
    assert job.status == JobStatus.COMPLETE.value
    assert job.attempt == 1 and len(renders) == 1, "the finished render was lost and made again"
    message = await _shown(client, run)
    assert [part["type"] for part in message["parts"]] == ["image", "generation_metadata"]
    picture = message["parts"][0]
    assert run["provenance_json"]["outputs"][0]["artifact_id"] == picture["artifact_id"]


async def test_a_render_held_out_through_every_try_fails_in_plain_words_and_is_not_left_running(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _OtherWriter(_database(settings))
    tries = _retry_quickly(monkeypatch)
    _hold_from_completion(monkeypatch, writer)
    refused = _let_go_after_refused_claims(monkeypatch, writer, refusals=tries)
    await choose_retries(client, 0)
    try:
        with _short_busy_wait():
            run_id, job_id = await _start_render(client, "held-through-every-try")
            await _executions_end(app)
    finally:
        writer.release()

    job = _job(job_id)
    assert job.status == JobStatus.FAILED.value, "a held database left the job running"
    assert job.error == HELD_TOO_LONG
    assert len(refused) == tries, "the save was not tried as often as it may be"
    run = await read_run(client, run_id)
    assert run["status"] == "failed"
    assert run["error"] == HELD_TOO_LONG
    message = await _shown(client, run)
    assert [part["text"] for part in message["parts"] if part["type"] == "error"] == [HELD_TOO_LONG]
    shown = json.dumps(message) + json.dumps(run)
    assert "sqlite3" not in shown and "[SQL:" not in shown and "database is locked" not in shown


async def test_a_failure_is_recorded_once_its_first_tries_have_met_a_held_database(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch)
    failing_media(app, monkeypatch, 1)
    tries = _hold_terminal_writes(
        monkeypatch, writer, status=JobStatus.FAILED.value, hold=lambda attempt: attempt < 2
    )
    await choose_retries(client, 0)
    with _short_busy_wait():
        run_id, job_id = await _start_render(client, "failure-after-held-tries")
        await _executions_end(app)

    job = _job(job_id)
    assert job.status == JobStatus.FAILED.value, "a held database left the job running"
    assert tries == [True, True, False]
    assert job.error == "Temporary media engine failure"
    run = await read_run(client, run_id)
    assert run["status"] == "failed"
    assert run["error"] == "Temporary media engine failure"


async def test_a_failure_held_out_through_every_try_is_recorded_once_the_database_frees(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The execution ends with its failure still unwritten; a later try writes it.

    The later tries go on while the database stays held: three of them are
    held out too, and the fourth lands. Clients are then told the run failed.
    """

    writer = _OtherWriter(_database(settings))
    tries_per_write = _retry_quickly(monkeypatch, later=0.05)
    failing_media(app, monkeypatch, 1)
    published = _record_events(app, monkeypatch)
    free = asyncio.Event()
    tries = _hold_terminal_writes(
        monkeypatch, writer, status=JobStatus.FAILED.value, hold=lambda _attempt: not free.is_set()
    )
    await choose_retries(client, 0)
    with _short_busy_wait():
        run_id, job_id = await _start_render(client, "failure-recorded-later")
        await _executions_end(app)
        assert len(tries) >= tries_per_write and all(tries), "the failure was not tried again"
        assert _job(job_id).status == JobStatus.RUNNING.value

        async def tried() -> int:
            return len(tries)

        await wait_until(
            tried, lambda count: count >= tries_per_write + 3, what="three later held tries"
        )
        assert all(tries) and _job(job_id).status == JobStatus.RUNNING.value
        free.set()

        async def status() -> str:
            return _job(job_id).status

        await wait_until(
            status, lambda current: current != JobStatus.RUNNING.value, what="the failure landing"
        )
        await _outcome_settled(job_id)

    job = _job(job_id)
    assert job.status == JobStatus.FAILED.value
    assert job.error == "Temporary media engine failure"
    assert tries.count(True) >= tries_per_write + 3 and tries[-1] is False
    run = await read_run(client, run_id)
    assert run["status"] == "failed"
    message = await _shown(client, run)
    assert [part["text"] for part in message["parts"] if part["type"] == "error"] == [
        "Temporary media engine failure"
    ]
    assert ("run.failed", run_id) in published


async def test_a_retry_queued_by_a_failure_recorded_later_is_started(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The execution that failed has ended by the time its failure lands, so nothing else would."""

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=0.05)
    requests = failing_media(app, monkeypatch, 1)
    published = _record_events(app, monkeypatch)
    free = asyncio.Event()
    _hold_terminal_writes(
        monkeypatch, writer, status=JobStatus.FAILED.value, hold=lambda _attempt: not free.is_set()
    )
    await choose_retries(client, 1)
    with _short_busy_wait():
        run_id, job_id = await _start_render(client, "retry-after-a-failure-recorded-later")
        await _executions_end(app)
        assert _job(job_id).status == JobStatus.RUNNING.value

        free.set()

        async def status() -> str:
            return str((await read_run(client, run_id))["status"])

        await wait_until(
            status,
            lambda current: current in {"complete", "failed", "cancelled"},
            what="the retried run ending",
        )

    run = await read_run(client, run_id)
    assert run["status"] == "complete", run["error"]
    assert run["provenance_json"]["failure_retries"]["used"] == 1
    assert len(requests) == 2
    assert _job(job_id).attempt == 2
    assert ("run.retrying", run_id) in published


async def test_a_failure_still_waiting_for_the_database_ends_when_the_application_closes(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing goes on trying once the application has closed; the next start marks the job."""

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch)
    failing_media(app, monkeypatch, 1)
    _hold_terminal_writes(
        monkeypatch, writer, status=JobStatus.FAILED.value, hold=lambda _attempt: True
    )
    await choose_retries(client, 0)
    orchestrator = app.state.services.orchestrator
    with _short_busy_wait():
        _run_id, job_id = await _start_render(client, "failure-given-up-at-close")
        await _executions_end(app)
        waiting = [
            task for task in asyncio.all_tasks() if task.get_name() == f"local-lm-failure-{job_id}"
        ]
        assert waiting, "nothing went on trying to record the failure"

        await asyncio.wait_for(orchestrator.close(), timeout=PATIENCE_SECONDS)

    assert all(task.cancelled() for task in waiting)
    assert _job(job_id).status == JobStatus.RUNNING.value


async def test_a_render_starts_once_a_database_held_as_it_starts_frees(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch)
    resolved: list[str] = []
    original = ConversationOrchestrator._resolve_step_inputs

    def resolve(session: Session, run: Run) -> None:
        # The run's start reads its inputs before it writes anything, so the
        # writer taken here is held when that start first writes.
        if not resolved:
            writer.take()
        resolved.append(run.id)
        original(session, run)

    monkeypatch.setattr(ConversationOrchestrator, "_resolve_step_inputs", staticmethod(resolve))
    await choose_retries(client, 0)
    try:
        with _short_busy_wait(), _let_go_at_first_refusal(writer) as refused:
            run_id, job_id = await _start_render(client, "held-at-start")
            await _executions_end(app)
    finally:
        writer.release()

    run = await read_run(client, run_id)
    assert run["status"] == "complete", run["error"]
    assert refused, "the start has to have met the held database"
    assert resolved == [run_id, run_id], "the start was not tried again"
    assert _job(job_id).attempt == 1


async def test_a_render_goes_on_once_a_database_held_as_its_phase_is_shown_frees(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The phase is the claim check before a worker is loaded or stopped: it is waited out.

    So the phase's own write is tried again once the writer frees and lands,
    rather than left out as a progress update would be.
    """

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch)
    phases: list[str] = []
    original = ConversationOrchestrator._set_media_phase

    async def set_phase(
        self: ConversationOrchestrator, job_id: str, run_id: str, label: str, claim: Any
    ) -> bool:
        if not phases:
            writer.take()
        phases.append(label)
        return await original(self, job_id, run_id, label, claim)

    phase_writes: list[str] = []
    original_write = ConversationOrchestrator._write_media_phase

    def write_media_phase(self: ConversationOrchestrator, session: Session, **values: Any) -> bool:
        try:
            written = original_write(self, session, **values)
        except OperationalError as error:
            phase_writes.append("held" if database_is_contended(error) else "failed")
            raise
        phase_writes.append("landed" if written else "refused")
        return written

    monkeypatch.setattr(ConversationOrchestrator, "_set_media_phase", set_phase)
    monkeypatch.setattr(ConversationOrchestrator, "_write_media_phase", write_media_phase)
    refused = _let_go_after_refused_claims(monkeypatch, writer, refusals=1)
    await choose_retries(client, 0)
    try:
        with _short_busy_wait():
            run_id, job_id = await _start_render(client, "held-at-phase")
            await _executions_end(app)
    finally:
        writer.release()

    run = await read_run(client, run_id)
    assert run["status"] == "complete", run["error"]
    assert len(refused) == 1, "the phase has to have met the held database"
    assert phase_writes[:2] == ["held", "landed"], phase_writes
    assert phases and _job(job_id).attempt == 1


async def test_a_render_stopped_as_the_application_closes_is_cancelled_once_the_database_frees(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch)
    rendering = asyncio.Event()

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        yield MediaEvent(type="progress", progress=0.1, phase="loading", engine_sourced=True)
        rendering.set()
        # Closing the application stops the render here.
        await asyncio.sleep(PATIENCE_SECONDS)

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    tries = _hold_terminal_writes(
        monkeypatch, writer, status=None, hold=lambda attempt: attempt < 1
    )
    with _short_busy_wait():
        run_id, job_id = await _start_render(client, "held-as-it-closes")
        await asyncio.wait_for(rendering.wait(), timeout=PATIENCE_SECONDS)
        await asyncio.wait_for(app.state.services.orchestrator.close(), timeout=PATIENCE_SECONDS)

    job = _job(job_id)
    assert job.status == JobStatus.CANCELLED.value, "a held database left the job running"
    assert tries == [True, False]
    run = await read_run(client, run_id)
    assert run["status"] == "cancelled"


async def test_a_stop_held_out_through_every_try_is_recorded_once_the_database_frees(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render stopped while the application keeps running is not left running.

    The execution is stopped as deleting its chat, or an image edit check
    giving way to new work, stops one. Every try to record the stop meets the
    held writer, so the execution ends with its job still reading running, and
    the record lands later, once the writer lets go.
    """

    writer = _OtherWriter(_database(settings))
    tries_per_write = _retry_quickly(monkeypatch, later=0.05)
    rendering = asyncio.Event()

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        yield MediaEvent(type="progress", progress=0.1, phase="loading", engine_sourced=True)
        rendering.set()
        # The stop ends the render here.
        await asyncio.sleep(PATIENCE_SECONDS)

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    free = asyncio.Event()
    tries = _hold_terminal_writes(
        monkeypatch, writer, status=None, hold=lambda _attempt: not free.is_set()
    )
    orchestrator = app.state.services.orchestrator
    with _short_busy_wait():
        run_id, job_id = await _start_render(client, "stop-recorded-later")
        await asyncio.wait_for(rendering.wait(), timeout=PATIENCE_SECONDS)
        execution = orchestrator._tasks[job_id]
        execution.cancel()
        await asyncio.wait({execution}, timeout=PATIENCE_SECONDS)
        assert execution.cancelled(), "the stop did not end the execution"
        assert len(tries) >= tries_per_write and all(tries), "the stop was not tried again"
        assert _job(job_id).status == JobStatus.RUNNING.value

        free.set()

        async def status() -> str:
            return _job(job_id).status

        await wait_until(
            status, lambda current: current != JobStatus.RUNNING.value, what="the stop landing"
        )

    assert _job(job_id).status == JobStatus.CANCELLED.value
    assert tries[-1] is False
    run = await read_run(client, run_id)
    assert run["status"] == "cancelled"


async def test_a_setup_check_is_finished_once_a_database_held_as_it_finishes_frees(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_ready_role(settings, "image", operation="text_to_image")
    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch)
    tries: list[bool] = []
    original = ConversationOrchestrator._claim_still_finalizes

    def still_finalizes(
        self: ConversationOrchestrator, session: Session, job_id: str, claim: Any
    ) -> bool:
        holding = not tries
        tries.append(holding)
        if not holding:
            return original(self, session, job_id, claim)
        writer.take()
        try:
            return original(self, session, job_id, claim)
        finally:
            writer.release()

    monkeypatch.setattr(ConversationOrchestrator, "_claim_still_finalizes", still_finalizes)
    with _short_busy_wait():
        assert (await client.post("/api/setup/verify/image")).status_code == 202
        await _executions_end(app)

    with SessionLocal() as session:
        verification = session.scalar(select(SetupVerification))
        assert verification is not None
        assert verification.state == "ready", "a held database left the setup check unfinished"
        assert session.scalars(select(Job)).all() == []
    assert tries == [True, False]


async def test_a_videos_proxy_and_poster_are_saved_once_a_database_held_as_each_is_saved_frees(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each is made once; only the write that stores it is tried again."""

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch)
    artifacts = app.state.services.artifacts
    staged_path = artifacts.root / "held-proxy.mp4"
    made: list[str] = []

    async def staged_proxy(_artifact: Artifact) -> StagedArtifactFile:
        made.append("proxy")
        staged_path.write_bytes(b"browser-playable-proxy")
        writer.take()
        return StagedArtifactFile(
            path=staged_path, media_type="video/mp4", original_name="held-proxy.mp4"
        )

    async def poster(_artifact: Artifact) -> bytes:
        made.append("poster")
        writer.take()
        return b"\xff\xd8\xff\xe0poster-frame"

    monkeypatch.setattr(artifacts, "browser_video_proxy", staged_proxy)
    monkeypatch.setattr(artifacts, "video_poster", poster)
    refused = _let_go_after_refused_claims(monkeypatch, writer, refusals=1)
    await choose_retries(client, 0)
    try:
        with _short_busy_wait():
            run_id, _job_id = await _start_render(client, "held-video-playback", mode="video")
            await _executions_end(app)
    finally:
        writer.release()

    run = await read_run(client, run_id)
    assert run["status"] == "complete", run["error"]
    assert made == ["proxy", "poster"]
    assert len(refused) == 2, "the proxy and the poster have to have met the held database"
    output = run["provenance_json"]["outputs"][0]
    assert output["browser_proxy_artifact_id"] and output["poster_artifact_id"]
    assert not staged_path.exists()
    with SessionLocal() as session:
        video = session.get(Artifact, output["artifact_id"])
        assert video is not None
        assert (
            video.metadata_json["browser_proxy_artifact_id"] == output["browser_proxy_artifact_id"]
        )
        assert video.metadata_json["poster_artifact_id"] == output["poster_artifact_id"]
    delivered = await client.get(f"/api/artifacts/{output['browser_proxy_artifact_id']}/content")
    assert delivered.status_code == 200
    assert delivered.content == b"browser-playable-proxy"


async def test_a_save_that_fails_for_another_reason_is_not_tried_again(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only a held database is waited out; a broken one still ends the render at once."""

    _retry_quickly(monkeypatch)
    broken = _real_sqlite_error(tmp_path / "broken", contended=False)
    finished: list[str] = []
    original_generate = MockMediaAdapter.generate

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        async for media_event in original_generate(self, request):
            if media_event.type == "complete":
                finished.append(request.run_id)
            yield media_event

    tries: list[str] = []
    original = ConversationOrchestrator._claim_owns_row

    def claim_owns_row(
        self: ConversationOrchestrator, session: Session, job_id: str, claim: Any
    ) -> bool:
        if finished:
            tries.append(job_id)
            raise broken
        return original(self, session, job_id, claim)

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    monkeypatch.setattr(ConversationOrchestrator, "_claim_owns_row", claim_owns_row)
    await choose_retries(client, 0)
    run_id, job_id = await _start_render(client, "broken-save")
    await _executions_end(app)

    assert len(tries) == 1, "a failure other than a held database was tried again"
    job = _job(job_id)
    assert job.status == JobStatus.FAILED.value
    assert job.error == "This run stopped because of a database error."
    run = await read_run(client, run_id)
    assert run["status"] == "failed"
