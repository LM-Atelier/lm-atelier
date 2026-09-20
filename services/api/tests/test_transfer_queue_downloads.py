"""Download controllers retain acceptance and recovery while a lane is paused."""

from __future__ import annotations

import asyncio
from typing import Never

import pytest
from fastapi import FastAPI

from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.models import InstallPlan, Job
from local_lm.queue_lane_policy import Action, LanePolicy, change_lane_policy, read_lane_policy
from local_lm.schemas import DownloadRequest, QueueControlCommand


def control(action: Action, revision: int) -> LanePolicy:
    with SessionLocal() as session:
        return change_lane_policy(
            session,
            "transfer",
            action,
            QueueControlCommand(expected_revision=revision, idempotency_key=f"command-{revision}"),
        )


def read() -> LanePolicy:
    with SessionLocal() as session:
        return read_lane_policy(session, "transfer")


def create(manager: DownloadManager, name: str) -> str:
    with SessionLocal() as session:
        job = manager.create(
            session,
            DownloadRequest(
                remote_id="example/" + name,
                revision="a" * 40,
                role="chat",
                engine="llama.cpp",
                allow_patterns=["model.gguf"],
            ),
        )
        return job.id


async def wait_queued(*job_ids: str) -> None:
    async with asyncio.timeout(10):
        while True:
            with SessionLocal() as session:
                jobs = [session.get(Job, job_id) for job_id in job_ids]
                assert all(
                    job is not None and job.status == "queued" and job.claim_owner is None
                    for job in jobs
                )
                if all(job is not None and job.phase == "transfer paused" for job in jobs):
                    return
            await asyncio.sleep(0.01)


async def test_new_and_individually_resumed_downloads_wait_for_lane_resume(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager: DownloadManager = app.state.services.downloads
    entered: list[str] = []

    async def sources(request: DownloadRequest, _plan: InstallPlan | None) -> Never:
        entered.append(request.remote_id)
        raise OSError("Constructed transfer failure")

    monkeypatch.setattr(manager, "_download_sources", sources)
    assert control("pause_after_current", 0).dispatch_state == "paused"
    try:
        accepted = create(manager, "accepted")
        resumed = create(manager, "resumed")
        manual = create(manager, "manual")
        assert await manager.pause(resumed)
        assert await manager.pause(manual)
        assert manager.resume(resumed)
        await wait_queued(accepted, resumed)
        assert entered == []
        with SessionLocal() as session:
            job = session.get(Job, accepted)
            assert job is not None
            original = dict(job.payload_json)
        tasks = [manager._tasks[accepted], manager._tasks[resumed]]
        assert control("resume", 1).dispatch_state == "open"
        await manager.scheduler.queue_control_changed("transfer")
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
        assert sorted(entered) == ["example/accepted", "example/resumed"]
        with SessionLocal() as session:
            job = session.get(Job, accepted)
            paused = session.get(Job, manual)
            assert job is not None and job.payload_json == original
            assert paused is not None and paused.status == "paused"
    finally:
        await manager.close()


@pytest.mark.parametrize("ending", ["failure", "cancel", "shutdown"])
async def test_download_controller_releases_the_last_draining_claim(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    manager: DownloadManager = app.state.services.downloads
    entered = asyncio.Event()
    release = asyncio.Event()

    async def sources(_request: DownloadRequest, _plan: InstallPlan | None) -> Never:
        entered.set()
        await release.wait()
        raise OSError("Constructed transfer failure")

    monkeypatch.setattr(manager, "_download_sources", sources)
    try:
        job_id = create(manager, "active")
        task = manager._tasks[job_id]
        await asyncio.wait_for(entered.wait(), timeout=10)
        policy = control("pause_after_current", 0)
        assert policy.dispatch_state == "draining" and policy.running_jobs == 1
        if ending == "failure":
            release.set()
            await asyncio.wait_for(task, timeout=10)
            expected = "failed"
        elif ending == "cancel":
            assert await manager.cancel(job_id)
            expected = "cancelled"
        else:
            await manager.close()
            expected = "interrupted"
        policy = read()
        assert policy.dispatch_state == "paused" and policy.running_jobs == 0
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.status == expected and job.claim_owner is None
    finally:
        await manager.close()


async def test_application_restart_recovers_downloads_under_the_durable_pause(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager: DownloadManager = app.state.services.downloads
    entered: list[str] = []

    async def sources(request: DownloadRequest, _plan: InstallPlan | None) -> Never:
        entered.append(request.remote_id)
        raise OSError("Constructed transfer failure")

    monkeypatch.setattr(manager, "_download_sources", sources)
    request = DownloadRequest(
        remote_id="example/recovered",
        revision="a" * 40,
        role="chat",
        engine="llama.cpp",
        allow_patterns=["model.gguf"],
    )
    with SessionLocal() as session:
        session.add(
            Job(
                id="recovered-download",
                kind="download",
                status="running",
                claim_owner="departed-process",
                queue_group="network",
                queue_resource="network_transfer",
                payload_json=request.model_dump(mode="json"),
            )
        )
        session.commit()
    assert control("pause_after_current", 0).dispatch_state == "draining"
    async with app.router.lifespan_context(app):
        await wait_queued("recovered-download")
        assert entered == []
        policy = read()
        assert policy.dispatch_state == "paused" and policy.running_jobs == 0
        task = manager._tasks["recovered-download"]
        control("resume", policy.revision)
        await manager.scheduler.queue_control_changed("transfer")
        await asyncio.wait_for(task, timeout=10)
        assert entered == ["example/recovered"]
