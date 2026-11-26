"""Cancelling a chat's work leaves no failure of its own able to start a job again.

A render with retries left fails while another writer holds the database, so
its failure goes on trying, in its execution and then in the background. Once
the chat's work is cancelled, that failure is recorded with its own error but
without the retry it would have queued, however its landing falls: after the
cancel's own write was refused and the database freed, while the cancel was
still waiting for the engine to stop another of the chat's runs, or after its
execution's own tries ran out during that stop. Nothing of the chat runs again
after its work is cancelled, as after a stop of the one job. Another chat's
failure keeps its retry, even one landing while the engine stops this chat's
render, and a retry someone asks for afterwards runs as any retry does.

The writer is held for real, from a second connection.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from test_finished_renders_land_on_a_held_database import (
    _database,
    _executions_end,
    _job,
    _OtherWriter,
    _outcome_settled,
    _outcome_tasks,
    _record_events,
    _retry_quickly,
    _start_render,
)
from test_generation_retry import failing_media, read_run
from test_outcomes_outlast_a_held_database import _job_left
from test_renders_outlast_a_busy_database import _short_busy_wait

from local_lm import orchestrator as orchestrator_module
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.config import Settings
from local_lm.db import database_is_contended
from local_lm.domain import JobStatus
from local_lm.orchestrator import ConversationOrchestrator


def _hold_the_first_failure(monkeypatch: pytest.MonkeyPatch, writer: _OtherWriter) -> list[bool]:
    """Have the writer taken at a failure's first try and kept; each try's view of it, in order."""

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
            writer.take()
        tries.append(writer.holding)
        return original(self, session, job_id, claim, **values)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    return tries


async def _failure_left_to_the_background(job_id: str) -> None:
    async def left() -> int:
        return len(_outcome_tasks(job_id))

    await wait_until(left, lambda count: count == 1, what="the failure left for later")


async def test_cancelling_a_chats_work_withdraws_the_retry_a_waiting_failure_would_queue(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _OtherWriter(_database(settings))
    tries_per_write = _retry_quickly(monkeypatch, later=0.3)
    renders = failing_media(app, monkeypatch, 1)
    tries = _hold_the_first_failure(monkeypatch, writer)
    allowance = await client.get("/api/settings/generation-retries")
    assert allowance.json()["max_retries"] == 3, "the allowance a new installation starts with"
    orchestrator = app.state.services.orchestrator
    try:
        with _short_busy_wait():
            run_id, job_id = await _start_render(client, "cancelled-with-its-chat")
            chat_id = (await read_run(client, run_id))["chat_id"]
            await _failure_left_to_the_background(job_id)
            await _executions_end(app)
            assert len(tries) >= tries_per_write and all(tries[:tries_per_write]), tries
            assert _job(job_id).status == JobStatus.RUNNING.value
            with pytest.raises(OperationalError) as refused:
                await client.post(f"/api/chats/{chat_id}/cancel-all")
            assert database_is_contended(refused.value)
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
    assert len(renders) == 1, "the render was asked for again after its chat's work was cancelled"
    assert job_id not in orchestrator._tasks


async def test_a_failure_landing_while_the_engine_stops_another_run_of_the_chat_queues_no_retry(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cancel waits for the engine to stop the chat's other render; the failure lands then."""

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=0.3)
    orchestrator = app.state.services.orchestrator
    adapter = app.state.services.engines.media
    original_generate = adapter.generate
    original_cancel = adapter.cancel
    renders: list[str] = []

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        renders.append(request.run_id)
        if len(renders) == 1:
            raise RuntimeError("Temporary media engine failure")
        if len(renders) == 2:
            yield MediaEvent(type="progress", progress=0.1, phase="sampling", engine_sourced=True)
            await asyncio.Event().wait()
        async for event in original_generate(request):
            yield event

    monkeypatch.setattr(adapter, "generate", generate)
    # The writer is held around each of the first render's failure tries, until the stop.
    hold = [True]
    original_transition = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator, session: Session, job_id: str, claim: Any, **values: Any
    ) -> bool:
        if values.get("status") == JobStatus.FAILED.value and hold[0]:
            writer.take()
            try:
                return original_transition(self, session, job_id, claim, **values)
            finally:
                writer.release()
        return original_transition(self, session, job_id, claim, **values)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    stops: list[str] = []
    try:
        with _short_busy_wait():
            run_a, job_a = await _start_render(client, "fails-while-the-database-is-held")
            chat_id = (await read_run(client, run_a))["chat_id"]
            await _failure_left_to_the_background(job_a)
            second = await client.post(
                f"/api/chats/{chat_id}/turns",
                json={
                    "text": "Create an image of a green lighthouse",
                    "mode": "image",
                    "idempotency_key": "generates-while-the-work-is-cancelled",
                },
            )
            assert second.status_code == 202, second.text
            run_b = second.json()["run"]["id"]

            async def generating() -> bool:
                return run_b in orchestrator._media_generating

            await wait_until(generating, bool, what="the second render generating")
            assert _job(job_a).status == JobStatus.RUNNING.value, "the failure still waits"

            async def slow_stop(run_id: str) -> None:
                # The database frees as the stop begins, and the stop takes a while.
                stops.append(run_id)
                hold[0] = False
                await asyncio.sleep(1.0)
                await original_cancel(run_id)

            monkeypatch.setattr(adapter, "cancel", slow_stop)
            events = _record_events(app, monkeypatch)
            cancelled = await client.post(f"/api/chats/{chat_id}/cancel-all")
            assert cancelled.status_code == 204, cancelled.text
            await _outcome_settled(job_a)
            await _executions_end(app)
    finally:
        writer.release()

    assert stops == [run_b]
    run = await read_run(client, run_a)
    assert _job(job_a).status == JobStatus.FAILED.value, (_job(job_a).status, _job(job_a).error)
    assert run["provenance_json"]["failure_retries"]["used"] == 0, run["provenance_json"]
    assert ("run.retrying", run_a) not in events, events
    assert len(renders) == 2, renders


async def test_cancelling_another_chats_work_leaves_this_chats_failure_its_retry(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=0.3)
    renders = failing_media(app, monkeypatch, 1)
    _hold_the_first_failure(monkeypatch, writer)
    other = (await client.post("/api/chats", json={"title": "Another chat"})).json()
    try:
        with _short_busy_wait():
            run_id, job_id = await _start_render(client, "kept-while-another-chat-is-cancelled")
            await _failure_left_to_the_background(job_id)
            await _executions_end(app)
            cancelled = await client.post(f"/api/chats/{other['id']}/cancel-all")
            assert cancelled.status_code == 204, cancelled.text
            writer.release()
            ended = await _job_left(job_id, JobStatus.RUNNING.value)
            await _outcome_settled(job_id)
    finally:
        writer.release()

    assert ended.status != JobStatus.FAILED.value, (ended.status, ended.error)

    async def status() -> str:
        return str((await read_run(client, run_id))["status"])

    await wait_until(status, lambda value: value == "complete", what="the retry finishing")
    run = await read_run(client, run_id)
    assert run["provenance_json"]["failure_retries"]["used"] == 1, run["provenance_json"]
    assert len(renders) == 2


async def test_another_chats_failure_landing_while_this_chats_work_is_cancelled_keeps_its_retry(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cancel waits for the engine to stop its own chat's render; the failure lands then."""

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=0.3)
    orchestrator = app.state.services.orchestrator
    adapter = app.state.services.engines.media
    original_generate = adapter.generate
    original_cancel = adapter.cancel
    renders: list[str] = []

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        renders.append(request.run_id)
        if len(renders) == 1:
            raise RuntimeError("Temporary media engine failure")
        if len(renders) == 2:
            yield MediaEvent(type="progress", progress=0.1, phase="sampling", engine_sourced=True)
            await asyncio.Event().wait()
        async for event in original_generate(request):
            yield event

    monkeypatch.setattr(adapter, "generate", generate)
    # The writer is held around each of the first render's failure tries, until the stop.
    hold = [True]
    original_transition = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator, session: Session, job_id: str, claim: Any, **values: Any
    ) -> bool:
        if values.get("status") == JobStatus.FAILED.value and hold[0]:
            writer.take()
            try:
                return original_transition(self, session, job_id, claim, **values)
            finally:
                writer.release()
        return original_transition(self, session, job_id, claim, **values)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    cancelled_chat = (await client.post("/api/chats", json={"title": "Cancelled chat"})).json()
    stops: list[str] = []
    landed_during_the_stop: list[str] = []
    try:
        with _short_busy_wait():
            run_a, job_a = await _start_render(client, "kept-while-another-chat-stops")
            await _failure_left_to_the_background(job_a)
            second = await client.post(
                f"/api/chats/{cancelled_chat['id']}/turns",
                json={
                    "text": "Create an image of a green lighthouse",
                    "mode": "image",
                    "idempotency_key": "stopped-while-another-chat-fails",
                },
            )
            assert second.status_code == 202, second.text
            run_b = second.json()["run"]["id"]

            async def generating() -> bool:
                return run_b in orchestrator._media_generating

            await wait_until(generating, bool, what="the cancelled chat's render generating")
            assert _job(job_a).status == JobStatus.RUNNING.value, "the failure still waits"

            async def slow_stop(run_id: str) -> None:
                # The database frees as the stop begins; the stop lasts until the failure lands.
                stops.append(run_id)
                hold[0] = False
                landed_during_the_stop.append((await _job_left(job_a, "running")).status)
                await original_cancel(run_id)

            monkeypatch.setattr(adapter, "cancel", slow_stop)
            cancelled = await client.post(f"/api/chats/{cancelled_chat['id']}/cancel-all")
            assert cancelled.status_code == 204, cancelled.text
    finally:
        writer.release()

    assert stops == [run_b]
    assert landed_during_the_stop == [JobStatus.QUEUED.value], landed_during_the_stop

    async def status() -> str:
        return str((await read_run(client, run_a))["status"])

    await wait_until(status, lambda value: value == "complete", what="the retry finishing")
    run = await read_run(client, run_a)
    assert run["provenance_json"]["failure_retries"]["used"] == 1, run["provenance_json"]
    assert len(renders) == 3, renders


async def test_a_retry_asked_for_after_the_chats_work_was_cancelled_keeps_its_own_retry(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old failure still waits when someone asks for the retry; that retry runs as any does."""

    writer = _OtherWriter(_database(settings))
    _retry_quickly(monkeypatch, later=0.3)
    renders = failing_media(app, monkeypatch, 2)
    held_token: list[str] = []
    held = [True]
    original = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator, session: Session, job_id: str, claim: Any, **values: Any
    ) -> bool:
        if values.get("status") == JobStatus.FAILED.value:
            if not held_token:
                held_token.append(claim.token)
            if claim.token == held_token[0] and held[0]:
                writer.take()
                try:
                    return original(self, session, job_id, claim, **values)
                finally:
                    writer.release()
        return original(self, session, job_id, claim, **values)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    try:
        with _short_busy_wait():
            run_id, job_id = await _start_render(client, "retried-after-its-chat-was-cancelled")
            chat_id = (await read_run(client, run_id))["chat_id"]
            await _failure_left_to_the_background(job_id)
            await _executions_end(app)
            cancelled = await client.post(f"/api/chats/{chat_id}/cancel-all")
            assert cancelled.status_code == 204, cancelled.text
            assert _job(job_id).status == JobStatus.CANCELLED.value
            assert len(_outcome_tasks(job_id)) == 1, "the old failure still waits"
            retried = await client.post(f"/api/jobs/{job_id}/retry")
            assert retried.status_code == 200, retried.text

            async def status() -> str:
                return str((await read_run(client, run_id))["status"])

            await wait_until(
                status, lambda value: value == "complete", what="the asked-for retry finishing"
            )
            held[0] = False
            await _outcome_settled(job_id)
    finally:
        writer.release()

    run = await read_run(client, run_id)
    assert len(renders) == 3, "the asked-for retry failed once and its own retry rendered"
    assert run["provenance_json"]["failure_retries"]["used"] == 1
    assert _job(job_id).status == JobStatus.COMPLETE.value


def _fail_while_the_second_render_waits(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, *, ending_seconds: float = 0.0
) -> tuple[list[str], asyncio.Event]:
    """The first render fails once let go; the second generates until stopped.

    Each failure write gets many short tries in its execution, so the failure
    is still trying there when the chat's work is cancelled, and long pauses in
    the background after. `ending_seconds` is how long the stopped second
    render takes to end. Returns the renders asked for and the event that lets
    the first one fail.
    """

    monkeypatch.setattr(orchestrator_module, "HELD_DATABASE_TRY_WAIT_MS", 100)
    monkeypatch.setattr(orchestrator_module, "HELD_DATABASE_RETRY_DELAYS_SECONDS", (0.05,) * 40)
    monkeypatch.setattr(orchestrator_module, "OUTCOME_RECORD_RETRY_DELAYS_SECONDS", (0.3,) * 400)
    adapter = app.state.services.engines.media
    original_generate = adapter.generate
    renders: list[str] = []
    fail_first = asyncio.Event()

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        renders.append(request.run_id)
        if len(renders) == 1:
            yield MediaEvent(type="progress", progress=0.1, phase="sampling", engine_sourced=True)
            await fail_first.wait()
            raise RuntimeError("Temporary media engine failure")
        if len(renders) == 2:
            yield MediaEvent(type="progress", progress=0.1, phase="sampling", engine_sourced=True)
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(ending_seconds)
        async for event in original_generate(request):
            yield event

    monkeypatch.setattr(adapter, "generate", generate)
    return renders, fail_first


async def _two_renders_in_one_chat(
    app: FastAPI,
    client: AsyncClient,
    fail_first: asyncio.Event,
    label: str,
    first_job: list[str],
) -> tuple[str, str, str, str]:
    """Start two renders in one chat, let the first fail, and wait for the second to generate.

    `first_job` is given the first render's job before that render may fail,
    so a hold on that job's failure covers its first try. Returns the chat,
    the first render's run and job, and the second render's run.
    """

    orchestrator = app.state.services.orchestrator
    run_a, job_a = await _start_render(client, f"{label}-first")
    first_job.append(job_a)
    chat_id = (await read_run(client, run_a))["chat_id"]

    async def a_generating() -> bool:
        return run_a in orchestrator._media_generating

    await wait_until(a_generating, bool, what="the first render generating")
    second = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={
            "text": "Create an image of a green lighthouse",
            "mode": "image",
            "idempotency_key": f"{label}-second",
        },
    )
    assert second.status_code == 202, second.text
    run_b = second.json()["run"]["id"]
    fail_first.set()

    async def b_generating() -> bool:
        return run_b in orchestrator._media_generating

    await wait_until(b_generating, bool, what="the second render generating")
    return chat_id, run_a, job_a, run_b


async def test_a_failure_whose_tries_run_out_during_the_stop_queues_no_retry_once_it_lands(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Still trying in its execution as the cancel begins, it is left for later during the stop.

    The database is then held for good, so the cancel's own marking is refused;
    the failure lands once the database frees, after the cancel has ended.
    """

    writer = _OtherWriter(_database(settings))
    renders, fail_first = _fail_while_the_second_render_waits(app, monkeypatch)
    job_a_box: list[str] = []
    refusing = [True]
    persistent = [False]
    tries: list[float] = []
    original_transition = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator, session: Session, job_id: str, claim: Any, **values: Any
    ) -> bool:
        if values.get("status") == JobStatus.FAILED.value and job_a_box == [job_id] and refusing[0]:
            tries.append(time.monotonic())
            if persistent[0]:
                return original_transition(self, session, job_id, claim, **values)
            writer.take()
            try:
                return original_transition(self, session, job_id, claim, **values)
            finally:
                writer.release()
        return original_transition(self, session, job_id, claim, **values)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    adapter = app.state.services.engines.media
    original_cancel = adapter.cancel
    stops: list[str] = []
    waiting: list[int] = []
    try:
        with _short_busy_wait():
            chat_id, run_a, job_a, run_b = await _two_renders_in_one_chat(
                app, client, fail_first, "runs-out-during-the-stop", job_a_box
            )
            waiting.append(len(_outcome_tasks(job_a)))
            assert _job(job_a).status == JobStatus.RUNNING.value

            async def slow_stop(run_id: str) -> None:
                stops.append(run_id)
                waiting.append(len(_outcome_tasks(job_a)))
                await _failure_left_to_the_background(job_a)
                waiting.append(len(_outcome_tasks(job_a)))
                # Held from here, so the cancel's marking is refused.
                persistent[0] = True
                writer.take()
                await original_cancel(run_id)

            monkeypatch.setattr(adapter, "cancel", slow_stop)
            events = _record_events(app, monkeypatch)
            with pytest.raises(OperationalError) as refused:
                await client.post(f"/api/chats/{chat_id}/cancel-all")
            assert database_is_contended(refused.value)
            writer.release()
            persistent[0] = False
            refusing[0] = False
            await _outcome_settled(job_a)
            await _executions_end(app)
            await _outcome_settled(job_a)
    finally:
        writer.release()

    # Not yet left for later as the cancel began, nor as the stop began; left during it.
    assert waiting == [0, 0, 1], waiting
    assert stops == [run_b]
    run = await read_run(client, run_a)
    assert _job(job_a).status == JobStatus.FAILED.value, (_job(job_a).status, _job(job_a).error)
    assert run["provenance_json"]["failure_retries"]["used"] == 0, run["provenance_json"]
    assert ("run.retrying", run_a) not in events, events
    assert len(renders) == 2, renders


async def test_a_failure_landing_while_the_cancel_waits_for_a_stopped_render_queues_no_retry(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Left for later during the stop, it lands while the stopped render is still ending."""

    writer = _OtherWriter(_database(settings))
    renders, fail_first = _fail_while_the_second_render_waits(app, monkeypatch, ending_seconds=1.5)
    job_a_box: list[str] = []
    refusing = [True]
    original_transition = ConversationOrchestrator._claim_terminal_transition

    def transition(
        self: ConversationOrchestrator, session: Session, job_id: str, claim: Any, **values: Any
    ) -> bool:
        if values.get("status") == JobStatus.FAILED.value and job_a_box == [job_id] and refusing[0]:
            writer.take()
            try:
                return original_transition(self, session, job_id, claim, **values)
            finally:
                writer.release()
        return original_transition(self, session, job_id, claim, **values)

    monkeypatch.setattr(ConversationOrchestrator, "_claim_terminal_transition", transition)
    adapter = app.state.services.engines.media
    original_cancel = adapter.cancel
    stops: list[str] = []
    waiting: list[int] = []
    try:
        with _short_busy_wait():
            chat_id, run_a, job_a, run_b = await _two_renders_in_one_chat(
                app, client, fail_first, "lands-while-the-cancel-waits", job_a_box
            )
            waiting.append(len(_outcome_tasks(job_a)))

            async def slow_stop(run_id: str) -> None:
                stops.append(run_id)
                waiting.append(len(_outcome_tasks(job_a)))
                await _failure_left_to_the_background(job_a)
                waiting.append(len(_outcome_tasks(job_a)))
                # The database frees as the stop ends; the failure's next try follows.
                refusing[0] = False
                await original_cancel(run_id)

            monkeypatch.setattr(adapter, "cancel", slow_stop)
            events = _record_events(app, monkeypatch)
            cancelled = await client.post(f"/api/chats/{chat_id}/cancel-all")
            assert cancelled.status_code == 204, cancelled.text
            await _outcome_settled(job_a)
            await _executions_end(app)
            await _outcome_settled(job_a)
    finally:
        writer.release()

    assert waiting == [0, 0, 1], waiting
    assert stops == [run_b]
    run = await read_run(client, run_a)
    assert run["provenance_json"]["failure_retries"]["used"] == 0, run["provenance_json"]
    assert ("run.retrying", run_a) not in events, events
    assert ("run.created", run_a) not in events, events
    assert len(renders) == 2, renders
