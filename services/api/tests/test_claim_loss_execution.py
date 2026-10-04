"""A displaced media execution stops without rewriting its replacement's rows."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm import scheduler as scheduler_module
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.mock import MockMediaAdapter
from local_lm.db import SessionLocal
from local_lm.models import Job, Run
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.scheduler import ResourceScheduler


@asynccontextmanager
async def _running_image(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cleanup_error: bool = False,
) -> AsyncIterator[tuple[str, str, asyncio.Task[None], asyncio.Task[Any], asyncio.Event]]:
    entered = asyncio.Event()
    exited = asyncio.Event()

    async def silent_media(
        self: MockMediaAdapter, request: MediaRequest
    ) -> AsyncIterator[MediaEvent]:
        del self, request
        try:
            entered.set()
            yield MediaEvent(type="progress", progress=0.05, phase="loading", engine_sourced=True)
            await asyncio.Event().wait()
        finally:
            exited.set()
            if cleanup_error:
                raise RuntimeError("The media cleanup failed")

    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(MockMediaAdapter, "generate", silent_media)
    chat_response = await client.post("/api/chats", json={"title": "Garden work"})
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]
    accepted = await client.post(
        f"/api/chats/{chat_id}/turns", json={"text": "Draw a garden path", "mode": "image"}
    )
    assert accepted.status_code == 202
    run_id: str = accepted.json()["run"]["id"]
    await asyncio.wait_for(entered.wait(), timeout=5)
    with SessionLocal() as session:
        job_id = session.scalar(select(Job.id).where(Job.run_id == run_id))
        assert job_id is not None
    orchestrator: ConversationOrchestrator = app.state.services.orchestrator
    execution = orchestrator._tasks[job_id]
    heartbeat = next(
        task for task in asyncio.all_tasks() if task.get_name() == f"job-heartbeat-{job_id}"
    )
    try:
        yield job_id, run_id, execution, heartbeat, exited
    finally:
        if not execution.done():
            execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)


@pytest.mark.parametrize("replacement", [False, True])
async def test_lost_claim_stops_the_exact_media_execution_and_keeps_current_rows(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, replacement: bool
) -> None:
    async with _running_image(client, app, monkeypatch) as (
        job_id,
        run_id,
        execution,
        heartbeat,
        exited,
    ):
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            original_token = job.claim_owner
            assert original_token is not None
            job.claim_owner = "garden-replacement" if replacement else None
            job.claim_expires_at = None
            if replacement:
                job.attempt += 1
            expected = job.claim_owner, job.attempt, job.status, run.status
            session.commit()
        await asyncio.wait_for(asyncio.shield(heartbeat), timeout=5)
        try:
            await asyncio.wait_for(exited.wait(), timeout=2)
        except TimeoutError:
            pytest.fail("The media execution continued after its heartbeat lost ownership")
        outcome = await asyncio.wait_for(
            asyncio.gather(execution, return_exceptions=True), timeout=5
        )
        assert outcome[0] is None or isinstance(outcome[0], asyncio.CancelledError)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            assert (job.claim_owner, job.attempt, job.status, run.status) == expected


async def test_a_status_change_with_the_same_claim_keeps_execution_running(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _running_image(client, app, monkeypatch) as (
        job_id,
        run_id,
        execution,
        heartbeat,
        exited,
    ):
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.claim_owner is not None
            token = job.claim_owner
            job.status = "paused"
            session.commit()
        await asyncio.wait_for(asyncio.shield(heartbeat), timeout=5)
        assert not exited.is_set() and not execution.done()
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            assert job.claim_owner == token and job.status == "paused" and run.status == "running"


async def test_a_reclaimed_generation_keeps_running_when_the_older_execution_stops(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = [asyncio.Event(), asyncio.Event()]
    exited = [asyncio.Event(), asyncio.Event()]
    requests = 0

    async def two_generations(
        self: MockMediaAdapter, request: MediaRequest
    ) -> AsyncIterator[MediaEvent]:
        nonlocal requests
        del self, request
        index = requests
        requests += 1
        assert index < 2
        try:
            entered[index].set()
            yield MediaEvent(type="progress", progress=0.05, phase="loading", engine_sourced=True)
            await asyncio.Event().wait()
        finally:
            exited[index].set()

    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(MockMediaAdapter, "generate", two_generations)
    chat = await client.post("/api/chats", json={"title": "Garden attempts"})
    assert chat.status_code == 201
    accepted = await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={"text": "Draw a garden path", "mode": "image"},
    )
    assert accepted.status_code == 202
    run_id: str = accepted.json()["run"]["id"]
    await asyncio.wait_for(entered[0].wait(), timeout=5)
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        assert job is not None and job.claim_owner is not None
        job_id, older_token = job.id, job.claim_owner
        job.status = "queued"
        job.claim_owner = None
        job.claim_expires_at = None
        session.commit()
    original: ConversationOrchestrator = app.state.services.orchestrator
    older = original._tasks[job_id]
    older_heartbeat = next(
        task for task in asyncio.all_tasks() if task.get_name() == f"job-heartbeat-{job_id}"
    )
    replacement = ConversationOrchestrator(
        original.engines,
        original.artifacts,
        original.events,
        ResourceScheduler(session_factory=original.session_factory),
        original.processes,
        session_factory=original.session_factory,
    )
    replacement.start(job_id, run_id)
    newer = replacement._tasks[job_id]
    try:
        await asyncio.wait_for(entered[1].wait(), timeout=5)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            assert job.claim_owner is not None and job.claim_owner != older_token
            expected = job.claim_owner, job.attempt, job.status, run.status
        await asyncio.wait_for(asyncio.shield(older_heartbeat), timeout=5)
        assert not newer.done() and not exited[1].is_set()
        try:
            await asyncio.wait_for(exited[0].wait(), timeout=2)
        except TimeoutError:
            pytest.fail("The older execution continued after its replacement began generating")
        outcome = await asyncio.wait_for(asyncio.gather(older, return_exceptions=True), timeout=5)
        assert outcome[0] is None or isinstance(outcome[0], asyncio.CancelledError)
        assert not newer.done() and not exited[1].is_set()
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            assert (job.claim_owner, job.attempt, job.status, run.status) == expected
    finally:
        older.cancel()
        newer.cancel()
        await asyncio.gather(older, newer, return_exceptions=True)
        await replacement.close()
