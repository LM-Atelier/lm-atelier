"""Silent engines stop when the durable execution claim no longer exists."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_claim_loss_execution import _running_image

from local_lm import scheduler as scheduler_module
from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.adapters.mock import MockChatAdapter
from local_lm.db import SessionLocal
from local_lm.domain import JobStatus, RunStatus
from local_lm.models import Job, Run
from local_lm.orchestrator import ConversationOrchestrator


async def test_removing_the_job_stops_its_waiting_media_execution(
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
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            expected = run.status
            session.delete(job)
            session.commit()
        await asyncio.wait_for(asyncio.shield(heartbeat), timeout=5)
        try:
            await asyncio.wait_for(exited.wait(), timeout=2)
        except TimeoutError:
            pytest.fail("The media execution continued after its job was removed")
        outcome = await asyncio.wait_for(
            asyncio.gather(execution, return_exceptions=True), timeout=5
        )
        assert outcome[0] is None or isinstance(outcome[0], asyncio.CancelledError)
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert session.get(Job, job_id) is None
            assert run is not None and run.status == expected


async def test_a_silent_chat_stream_stops_after_its_claim_moves(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()
    exited = asyncio.Event()

    async def silent_chat(self: MockChatAdapter, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        del self, request
        try:
            yield ChatEvent(type="delta", text="A garden path.")
            entered.set()
            await asyncio.Event().wait()
        finally:
            exited.set()

    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(MockChatAdapter, "stream", silent_chat)
    chat = await client.post("/api/chats", json={"title": "Garden conversation"})
    assert chat.status_code == 201
    accepted = await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={"text": "Describe a garden path", "mode": "text"},
    )
    assert accepted.status_code == 202
    run_id: str = accepted.json()["run"]["id"]
    await asyncio.wait_for(entered.wait(), timeout=5)
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        run = session.get(Run, run_id)
        assert job is not None and job.claim_owner is not None and run is not None
        job_id = job.id
        job.claim_owner = "garden-chat-replacement"
        job.attempt += 1
        expected = job.claim_owner, job.attempt, job.status, run.status
        session.commit()
    orchestrator: ConversationOrchestrator = app.state.services.orchestrator
    execution = orchestrator._tasks[job_id]
    global_changes: list[str] = []
    release_restart = orchestrator._release_deferred_media_restart
    settle_prewarm = orchestrator._settle_step_prewarm

    def track_restart_release() -> None:
        global_changes.append("restart")
        release_restart()

    async def track_prewarm_settlement(current_job_id: str) -> None:
        global_changes.append("prewarm")
        await settle_prewarm(current_job_id)

    monkeypatch.setattr(orchestrator, "_release_deferred_media_restart", track_restart_release)
    monkeypatch.setattr(orchestrator, "_settle_step_prewarm", track_prewarm_settlement)
    orchestrator._displaced_chat_profile_id = "garden-displaced-profile"
    heartbeat = next(
        task for task in asyncio.all_tasks() if task.get_name() == f"job-heartbeat-{job_id}"
    )
    try:
        await asyncio.wait_for(asyncio.shield(heartbeat), timeout=5)
        try:
            await asyncio.wait_for(exited.wait(), timeout=2)
        except TimeoutError:
            pytest.fail("The chat stream continued after its heartbeat lost ownership")
        outcome = await asyncio.wait_for(
            asyncio.gather(execution, return_exceptions=True), timeout=5
        )
        assert outcome[0] is None or isinstance(outcome[0], asyncio.CancelledError)
        assert global_changes == []
        assert orchestrator._displaced_chat_profile_id == "garden-displaced-profile"
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            assert (job.claim_owner, job.attempt, job.status, run.status) == expected
    finally:
        if not execution.done():
            execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)


async def test_a_finished_generation_keeps_its_claim_during_a_slow_handoff(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()
    finish = asyncio.Event()
    orchestrator: ConversationOrchestrator = app.state.services.orchestrator

    async def prepare_handoff(*_args: object, **_kwargs: object) -> str:
        return "garden-chat-profile"

    async def complete_handoff(profile_id: str) -> None:
        assert profile_id == "garden-chat-profile"
        entered.set()
        await finish.wait()

    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(orchestrator, "_prepare_device_handoff", prepare_handoff)
    monkeypatch.setattr(orchestrator, "_complete_media_handoff", complete_handoff)
    chat = await client.post("/api/chats", json={"title": "Garden handoff"})
    assert chat.status_code == 201
    accepted = await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={"text": "Draw a garden path", "mode": "image"},
    )
    assert accepted.status_code == 202
    run_id: str = accepted.json()["run"]["id"]
    await asyncio.wait_for(entered.wait(), timeout=5)
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        run = session.get(Run, run_id)
        assert job is not None and job.claim_owner is not None and run is not None
        assert job.status == JobStatus.COMPLETE.value and run.status == RunStatus.COMPLETE.value
        job_id, token, heartbeat_at = job.id, job.claim_owner, job.heartbeat_at
    execution = orchestrator._tasks[job_id]

    async def heartbeat_advanced() -> None:
        while True:
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                assert job is not None and job.claim_owner == token
                if job.heartbeat_at != heartbeat_at:
                    return
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(heartbeat_advanced(), timeout=2)
        assert not execution.done()
        finish.set()
        await asyncio.wait_for(execution, timeout=5)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            assert job.claim_owner is None
            assert job.status == JobStatus.COMPLETE.value and run.status == RunStatus.COMPLETE.value
    finally:
        finish.set()
        if not execution.done():
            execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)
