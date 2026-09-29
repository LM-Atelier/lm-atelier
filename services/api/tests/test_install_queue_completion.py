"""Accepted workflow completion joins the installation lane before runtime work."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy.exc import OperationalError
from test_workflow_completion_jobs import _accept, _settled, _state
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import source_runtime as source_runtime

from local_lm import workflow_source_completion
from local_lm.db import SessionLocal
from local_lm.models import Job, WorkflowInstallOffer
from local_lm.queue_lane_policy import (
    Action,
    LanePolicy,
    change_lane_policy,
    read_lane_policy,
    recover_queue_lanes,
)
from local_lm.schemas import QueueControlCommand


def control(action: Action, revision: int) -> LanePolicy:
    with SessionLocal() as session:
        return change_lane_policy(
            session,
            "install",
            action,
            QueueControlCommand(expected_revision=revision, idempotency_key=f"command-{revision}"),
        )


async def test_source_completion_waits_for_installation_resume(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = app.state.services.downloads
    assert control("pause_after_current", 0).dispatch_state == "paused"
    with monkeypatch.context() as patch:
        patch.setattr(manager, "start_workflow_installation", lambda _id: None)
        _plan_id, offer_id, job_id = await _accept(client)
    manager.start_workflow_installation(offer_id)
    try:
        async with asyncio.timeout(10):
            while True:
                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    assert job is not None and job.status == "queued" and job.claim_owner is None
                    if job.phase == "install paused":
                        break
                await asyncio.sleep(0.01)
        assert _state(offer_id)[4] == 0
        policy = control("resume", 1)
        assert policy.dispatch_state == "open"
        await manager.scheduler.queue_control_changed("install")
        await _settled(app, offer_id)
        assert _state(offer_id)[0:2] == ("completed", "complete")
        assert _state(offer_id)[4] == 1
    finally:
        await manager.close()


async def test_restart_clears_a_claim_retained_by_a_queued_source_retry(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = app.state.services.downloads
    with monkeypatch.context() as patch:
        patch.setattr(manager, "start_workflow_installation", lambda _id: None)
        _plan_id, offer_id, job_id = await _accept(client)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        job.claim_owner = "departed-worker"
        job.attempt = 1
        job.queue_group = "primary"
        session.commit()
    assert control("pause_after_current", 0).dispatch_state == "draining"
    try:
        manager.recover_interrupted()
        with SessionLocal() as session:
            recover_queue_lanes(session)
            session.commit()
            job = session.get(Job, job_id)
            assert job is not None and job.status == "queued" and job.claim_owner is None
            policy = read_lane_policy(session, "install")
            assert policy.dispatch_state == "paused" and policy.running_jobs == 0
        control("resume", policy.revision)
        await manager.scheduler.queue_control_changed("install")
        await asyncio.wait_for(manager._offer_recovery_task, timeout=10)
        assert _state(offer_id)[0:2] == ("completed", "complete")
        assert _state(offer_id)[4] == 2
    finally:
        await manager.close()


@pytest.mark.parametrize("retry", [False, True])
async def test_source_completion_keeps_its_claim_until_old_work_releases(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, retry: bool
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    manager = app.state.services.downloads
    calls = 0

    async def validate(_graph: dict[str, Any]) -> list[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        return []

    with monkeypatch.context() as patch:
        patch.setattr(manager, "start_workflow_installation", lambda _id: None)
        _plan_id, offer_id, job_id = await _accept(client)
    monkeypatch.setattr(app.state.services.engines.media, "validate_workflow", validate)
    manager.start_workflow_installation(offer_id)
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        task = manager._offer_tasks[offer_id]
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.claim_owner is not None and job.attempt == 1
            token = job.claim_owner
        if retry:
            assert (await client.post(f"/api/jobs/{job_id}/cancel")).status_code == 200
            assert (await client.post(f"/api/jobs/{job_id}/retry")).status_code == 200
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                assert job is not None and job.status == "queued" and job.claim_owner == token
                assert job.attempt == 1
        policy = control("pause_after_current", 0)
        assert policy.dispatch_state == "draining" and policy.running_jobs == 1
        release.set()
        await asyncio.wait_for(asyncio.shield(task), timeout=10)
        with SessionLocal() as session:
            policy = read_lane_policy(session, "install")
            assert policy.dispatch_state == "paused" and policy.running_jobs == 0
        if retry:
            assert _state(offer_id)[0:2] == ("queued", "queued")
            with SessionLocal() as session:
                offer = session.get(WorkflowInstallOffer, offer_id)
                assert offer is not None and offer.completion_error_code is None
            control("resume", policy.revision)
            await manager.scheduler.queue_control_changed("install")
            await _settled(app, offer_id)
            assert calls == 2 and _state(offer_id)[4] == 2
        else:
            assert _state(offer_id)[4] == 1
        assert _state(offer_id)[0:2] == ("completed", "complete")
    finally:
        release.set()
        await manager.close()


@pytest.mark.parametrize("replacement", ["claim", "retry", "none"])
async def test_failed_source_preflight_cannot_mark_a_newer_execution_for_attention(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    manager = app.state.services.downloads
    entered, release = threading.Event(), threading.Event()
    with monkeypatch.context() as patch:
        patch.setattr(manager, "start_workflow_installation", lambda _id: None)
        _plan_id, offer_id, job_id = await _accept(client)

    def fail_preflight(_offer_id: str) -> str | None:
        entered.set()
        assert release.wait(timeout=10)
        raise OperationalError("neutral preflight", {}, RuntimeError("database unavailable"))

    monkeypatch.setattr(workflow_source_completion, "prepare_source_completion", fail_preflight)
    publish = AsyncMock(wraps=manager.events.publish)
    monkeypatch.setattr(manager.events, "publish", publish)
    task = asyncio.create_task(manager.reconcile_workflow_install_offers(only_offer_id=offer_id))
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        if replacement == "claim":
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                assert job is not None
                job.status = "running"
                job.claim_owner = "replacement-claim"
                job.attempt = 1
                session.commit()
        elif replacement == "retry":
            with monkeypatch.context() as patch:
                patch.setattr(manager, "start_workflow_installation", lambda _id: None)
                assert (await client.post(f"/api/jobs/{job_id}/cancel")).status_code == 200
                assert (await client.post(f"/api/jobs/{job_id}/retry")).status_code == 200
        release.set()
        await asyncio.wait_for(task, timeout=10)
        with SessionLocal() as session:
            offer = session.get(WorkflowInstallOffer, offer_id)
            job = session.get(Job, job_id)
            assert offer is not None and job is not None
            assert offer.completion_error_code == (
                "workflow-completion-unavailable" if replacement == "none" else None
            )
            assert job.status == ("running" if replacement == "claim" else "queued")
            assert job.attempt == (1 if replacement == "claim" else 0)
        attention = [
            call for call in publish.await_args_list if call.args[0] == "workflow.install.attention"
        ]
        assert len(attention) == (1 if replacement == "none" else 0)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await manager.close()


@pytest.mark.parametrize("boundary", ["before", "after"])
@pytest.mark.parametrize(
    "ending", ["cancel", "shutdown", "complete", "cancel-error", "shutdown-error"]
)
async def test_source_preflight_finishes_before_its_dispatcher_exits(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    ending: str,
) -> None:
    manager = app.state.services.downloads
    with monkeypatch.context() as patch:
        patch.setattr(manager, "start_workflow_installation", lambda _id: None)
        _plan_id, offer_id, job_id = await _accept(client)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = workflow_source_completion.prepare_source_completion

    def prepare(identifier: str) -> str | None:
        try:
            if boundary == "before":
                entered.set()
                assert release.wait(30), "Source preflight was not released"
            result = original(identifier)
            if boundary == "after":
                entered.set()
                assert release.wait(30), "Source preflight was not released"
            if ending.endswith("-error"):
                raise OperationalError(
                    "neutral preflight", {}, RuntimeError("database unavailable")
                )
            return result
        finally:
            finished.set()

    monkeypatch.setattr(workflow_source_completion, "prepare_source_completion", prepare)
    manager.start_workflow_installation(offer_id)
    task = manager._offer_tasks[offer_id]
    closing: asyncio.Task[None] | None = None
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                assert not task.done(), "Installation ended before preflight"
                await asyncio.sleep(0.01)
        if ending.startswith("cancel"):
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        elif ending.startswith("shutdown"):
            closing = asyncio.create_task(manager.close())
        if ending != "complete":
            for _ in range(10):
                await asyncio.sleep(0)
            assert not task.done(), "Dispatcher exited while its preflight thread was live"
            assert closing is None or not closing.done()
        assert not finished.is_set()
        release.set()
        outcomes = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=30)
        if closing is not None:
            await asyncio.wait_for(closing, timeout=10)
        assert finished.is_set()
        if ending == "complete":
            assert list(outcomes) == [None]
            assert _state(offer_id)[0:2] == ("completed", "complete")
            assert _state(offer_id)[4] == 1
        else:
            assert isinstance(outcomes[0], asyncio.CancelledError)
            assert _state(offer_id)[0:2] == ("queued", "queued")
            assert _state(offer_id)[4] == 0
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                assert job is not None and job.claim_owner is None and job.result_json == {}
                offer = session.get(WorkflowInstallOffer, offer_id)
                assert offer is not None and offer.completion_error_code is None
    finally:
        release.set()
        await asyncio.gather(task, *([closing] if closing else []), return_exceptions=True)
        async with asyncio.timeout(30):
            while not finished.is_set():
                await asyncio.sleep(0.01)
        await manager.close()


@pytest.mark.parametrize("entry", ["direct", "scheduled"])
async def test_duplicate_source_dispatchers_claim_and_complete_only_once(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    manager = app.state.services.downloads
    with monkeypatch.context() as patch:
        patch.setattr(manager, "start_workflow_installation", lambda _id: None)
        _plan_id, offer_id, job_id = await _accept(client)
    entered, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    calls = 0
    expected = 2 if entry == "direct" else 1
    original = workflow_source_completion.prepare_source_completion

    def prepare(identifier: str) -> str | None:
        nonlocal calls
        result = original(identifier)
        with lock:
            calls += 1
            if calls == expected:
                entered.set()
        assert release.wait(30), "Concurrent source preflight was not released"
        return result

    monkeypatch.setattr(workflow_source_completion, "prepare_source_completion", prepare)
    dispatch = AsyncMock(wraps=manager._dispatch_workflow_completion)
    monkeypatch.setattr(manager, "_dispatch_workflow_completion", dispatch)
    if entry == "direct":
        tasks = [
            asyncio.create_task(manager.reconcile_workflow_install_offers(only_offer_id=offer_id))
            for _ in range(2)
        ]
    else:
        manager.start_workflow_installation(offer_id)
        tasks = [manager._offer_tasks[offer_id]]
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                assert all(not task.done() for task in tasks)
                await asyncio.sleep(0.01)
        if entry == "scheduled":
            for _ in range(3):
                manager.start_workflow_installation(offer_id)
                assert manager._offer_tasks[offer_id] is tasks[0]
        release.set()
        outcomes = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=30
        )
        assert all(
            outcome is None or isinstance(outcome, asyncio.CancelledError) for outcome in outcomes
        )
        await _settled(app, offer_id)
        assert dispatch.await_count == 1
        assert _state(offer_id)[0:3] == ("completed", "complete", job_id)
        assert _state(offer_id)[4] == 1
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            offer = session.get(WorkflowInstallOffer, offer_id)
            assert job is not None and job.claim_owner is None
            assert offer is not None and offer.completion_error_code is None
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await manager.close()
