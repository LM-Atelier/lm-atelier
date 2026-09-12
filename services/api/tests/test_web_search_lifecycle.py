"""Generation waits release execution ownership without completing the response."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Job, Run, WorkPlan, WorkStep
from local_lm.scheduler import JobClaim


async def test_a_query_wait_can_release_its_real_generation_lease(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = app.state.services.orchestrator
    captured: list[JobClaim] = []

    async def pause_after_query(self: Any, job_id: str, run_id: str, claim: JobClaim) -> None:
        with SessionLocal() as session:
            assert self._claim_terminal_transition(
                session, job_id, claim, status="paused", phase="awaiting search approval"
            )
            run = session.get(Run, run_id)
            assert run is not None
            run.status = "queued"
            self._set_work_status(session, run, "paused")
            session.commit()
        captured.append(claim)

    monkeypatch.setattr(type(orchestrator), "_execute_chat", pause_after_query)
    monkeypatch.setattr(type(orchestrator), "start", lambda self, job_id, run_id: None)
    seen: list[tuple[str, str]] = []
    for index in range(2):
        created = await client.post("/api/chats", json={"title": f"Approval wait {index}"})
        assert created.status_code == 201
        accepted = await client.post(
            f"/api/chats/{created.json()['id']}/turns",
            json={"text": "Compare two neutral materials", "mode": "text"},
        )
        assert accepted.status_code == 202
        run_id = accepted.json()["run"]["id"]
        with SessionLocal() as session:
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None and job.status == "queued"
            job_id = job.id
        async with asyncio.timeout(30):
            await orchestrator._execute(job_id, run_id)
        seen.append((job_id, run_id))
        assert len(captured) == index + 1

    # The second real execution cannot acquire its slot if the first wait kept
    # the generation lease. Both must remain resumable after their scope exits.
    with SessionLocal() as session:
        for job_id, run_id in seen:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            assert job.status == "paused"
            assert job.claim_owner is None
            assert job.claim_expires_at is None
            assert job.heartbeat_at is None
            assert job.attempt == 1
            assert run.status == "queued"
            assert run.completed_at is None
            assert run.work_step_id is not None and run.work_plan_id is not None
            step = session.get(WorkStep, run.work_step_id)
            plan = session.get(WorkPlan, run.work_plan_id)
            assert step is not None and step.status == "paused"
            assert plan is not None and plan.status == "paused"


@pytest.mark.parametrize("approval_order", ["before_exit", "after_exit", "cancel_before_exit"])
async def test_search_approval_resumes_after_the_actual_pausing_scope(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    approval_order: str,
) -> None:
    from local_lm.models import Chat
    from local_lm.web_search_consent import decide_search, pause_for_search

    orchestrator = app.state.services.orchestrator
    paused, leave_pause, resumed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    claims: list[JobClaim] = []
    proposal: list[Any] = []

    async def execute_at_search(self: Any, job_id: str, run_id: str, claim: JobClaim) -> None:
        claims.append(claim)
        if len(claims) == 1:
            with SessionLocal() as session:
                proposal.append(
                    pause_for_search(
                        session,
                        job_id,
                        claim,
                        query="Compare two materials",
                        provider_endpoint="https://search.example.test",
                        provider_revision="provider-one",
                        installation_enabled=True,
                    )
                )
            paused.set()
            await leave_pause.wait()
        else:
            with SessionLocal() as session:
                assert self._claim_terminal_transition(session, job_id, claim, status="complete")
                run = session.get(Run, run_id)
                run.status = "complete"
                self._set_work_status(session, run, "complete")
                session.commit()
            resumed.set()

    monkeypatch.setattr(type(orchestrator), "_execute_chat", execute_at_search)
    created = await client.post("/api/chats", json={"title": "Search resumption"})
    assert created.status_code == 201
    with SessionLocal() as session:
        session.get(Chat, created.json()["id"]).web_settings_json = {"allow_search": True}
        session.commit()
    accepted = await client.post(
        f"/api/chats/{created.json()['id']}/turns",
        json={"text": "Compare two neutral materials", "mode": "text"},
    )
    assert accepted.status_code == 202
    async with asyncio.timeout(30):
        await paused.wait()
        current = proposal[0]
        previous = orchestrator._tasks[current.job_id]
        with SessionLocal() as session:
            decide_search(session, current.job_id, current.revision, "approve")
        if approval_order == "after_exit":
            leave_pause.set()
            await previous
        orchestrator.resume_search(current.job_id)
        orchestrator.resume_search(current.job_id)
        if approval_order == "cancel_before_exit":
            assert await orchestrator.cancel(current.job_id)
            leave_pause.set()
            await asyncio.gather(previous, return_exceptions=True)
            await asyncio.sleep(0)
            assert not resumed.is_set() and len(claims) == 1
        else:
            leave_pause.set()
            await resumed.wait()
            assert len(claims) == 2
            assert claims[0].attempt == 1 and claims[1].attempt == 2
            assert claims[0].token != claims[1].token
            finishing = orchestrator._tasks.get(current.job_id)
            if finishing is not None:
                await finishing


@pytest.mark.parametrize("ending", ["dispatch", "cancel", "close"])
async def test_automatic_search_wait_releases_compute_and_uses_normal_task_teardown(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    ending: str,
) -> None:
    from datetime import timedelta

    from local_lm.domain import utcnow
    from local_lm.models import Chat, WebSearchProposal
    from local_lm.web_search import CrwSearchProvider
    from local_lm.web_search_configuration import search_provider_revision
    from local_lm.web_search_consent import claim_search_dispatch, pause_for_search

    orchestrator = app.state.services.orchestrator
    orchestrator.engines.settings.web_access_enabled = True
    orchestrator.engines.settings.crw_endpoint = "https://search.example.test"
    provider = CrwSearchProvider("https://search.example.test")
    revision = search_provider_revision(provider)
    entered_wait, release_wait, other_finished, dispatched = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    proposal: list[Any] = []
    claims: list[JobClaim] = []
    auto_run: list[str] = []
    first_paused = asyncio.Event()
    first_tasks: list[asyncio.Task[Any]] = []

    async def wait_for_dispatch(self: Any, deadline: Any) -> None:
        assert deadline is not None
        entered_wait.set()
        await release_wait.wait()

    async def execute_at_search(self: Any, job_id: str, run_id: str, claim: JobClaim) -> None:
        if not auto_run:
            auto_run.append(run_id)
        if run_id == auto_run[0]:
            claims.append(claim)
            if len(claims) == 1:
                first_tasks.append(asyncio.current_task())
                with SessionLocal() as session:
                    proposal.append(
                        pause_for_search(
                            session,
                            job_id,
                            claim,
                            query="Compare two neutral materials",
                            provider_endpoint=provider.endpoint,
                            provider_revision=revision,
                            installation_enabled=True,
                        )
                    )
                self.resume_search(job_id)
                first_paused.set()
                return
            with SessionLocal() as session:
                result = claim_search_dispatch(
                    session,
                    job_id,
                    proposal[0].revision,
                    claim,
                    provider_endpoint=provider.endpoint,
                    provider_revision=revision,
                    installation_enabled=True,
                )
                assert result.state == "dispatching"
            dispatched.set()
        with SessionLocal() as session:
            assert self._claim_terminal_transition(session, job_id, claim, status="complete")
            run = session.get(Run, run_id)
            run.status = "complete"
            self._set_work_status(session, run, "complete")
            session.commit()
        if run_id != auto_run[0]:
            other_finished.set()

    monkeypatch.setattr(type(orchestrator), "_execute_chat", execute_at_search)
    monkeypatch.setattr(
        type(orchestrator),
        "_wait_for_search_dispatch",
        wait_for_dispatch,
        raising=False,
    )

    async def submit(automatic: bool) -> str:
        created = await client.post("/api/chats", json={"title": "Automatic search lifecycle"})
        assert created.status_code == 201
        with SessionLocal() as session:
            session.get(Chat, created.json()["id"]).web_settings_json = {
                "allow_search": True,
                "allow_search_without_asking": automatic,
            }
            session.commit()
        accepted = await client.post(
            f"/api/chats/{created.json()['id']}/turns",
            json={"text": "Compare two neutral materials", "mode": "text"},
        )
        assert accepted.status_code == 202
        return accepted.json()["run"]["id"]

    async with asyncio.timeout(30):
        await submit(True)
        await first_paused.wait()
        await first_tasks[0]
        await asyncio.sleep(0)
        assert entered_wait.is_set(), "the persisted automatic wait was never resumed"
        job_id = proposal[0].job_id
        timer = orchestrator._tasks[job_id]
        orchestrator.resume_search(job_id)
        assert orchestrator._tasks[job_id] is timer
        await submit(False)
        await other_finished.wait()
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job.status == "paused" and job.claim_owner is None
            assert job.attempt == 1 and len(claims) == 1
        if ending == "cancel":
            assert await orchestrator.cancel(job_id)
            assert timer.done()
            release_wait.set()
            await asyncio.sleep(0)
            assert not dispatched.is_set() and len(claims) == 1
        elif ending == "close":
            await orchestrator.close()
            assert timer.done() and not orchestrator._tasks
            orchestrator.resume_search(job_id)
            release_wait.set()
            await asyncio.sleep(0)
            assert not orchestrator._tasks
            assert not dispatched.is_set() and len(claims) == 1
        else:
            with SessionLocal() as session:
                session.scalar(select(WebSearchProposal)).dispatch_after = utcnow() - timedelta(
                    seconds=1
                )
                session.commit()
            release_wait.set()
            await dispatched.wait()
            await timer
            assert len(claims) == 2
            assert claims[0].attempt == 1 and claims[1].attempt == 2
            assert claims[0].token != claims[1].token
