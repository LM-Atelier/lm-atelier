"""Cancellation of a displaced execution preserves the durable attempt state."""

import asyncio

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from test_claim_loss_execution import _running_image

from local_lm.db import SessionLocal
from local_lm.models import Job, Run


@pytest.mark.parametrize("replacement", [False, True])
async def test_cancelling_a_displaced_execution_keeps_the_current_attempt_rows(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    replacement: bool,
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
            assert job is not None and job.claim_owner is not None and run is not None
            job.claim_owner = "garden-current-attempt" if replacement else None
            job.claim_expires_at = None
            if replacement:
                job.attempt += 1
            expected = job.claim_owner, job.attempt, job.status, run.status
            session.commit()

        await asyncio.wait_for(asyncio.shield(heartbeat), timeout=PATIENCE_SECONDS)
        execution.cancel()
        outcome = await asyncio.wait_for(
            asyncio.gather(execution, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        assert outcome[0] is None or isinstance(outcome[0], asyncio.CancelledError)
        assert exited.is_set()
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            assert (job.claim_owner, job.attempt, job.status, run.status) == expected


async def test_ordinary_cancellation_still_finishes_the_owned_job_and_run(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _running_image(client, app, monkeypatch) as (
        job_id,
        run_id,
        execution,
        _heartbeat,
        exited,
    ):
        execution.cancel()
        outcome = await asyncio.wait_for(
            asyncio.gather(execution, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        assert isinstance(outcome[0], asyncio.CancelledError) and exited.is_set()
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            assert job.status == "cancelled" and run.status == "cancelled"
            assert job.claim_owner is None


async def test_a_cleanup_error_after_claim_loss_keeps_the_current_rows(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = await client.put(
        "/api/settings/generation-retries", json={"max_retries": 0, "expected_revision": 0}
    )
    assert policy.status_code == 200
    async with _running_image(client, app, monkeypatch, cleanup_error=True) as (
        job_id,
        run_id,
        execution,
        heartbeat,
        exited,
    ):
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None and job.claim_owner is not None
            job.claim_owner = None
            job.claim_expires_at = None
            expected = job.claim_owner, job.attempt, job.status, run.status
            session.commit()
        await asyncio.wait_for(asyncio.shield(heartbeat), timeout=PATIENCE_SECONDS)
        if execution.cancelling() == 0:
            execution.cancel()
        outcome = await asyncio.wait_for(
            asyncio.gather(execution, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
        assert outcome[0] is None or isinstance(outcome[0], asyncio.CancelledError)
        assert exited.is_set()
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            assert job is not None and run is not None
            assert (job.claim_owner, job.attempt, job.status, run.status) == expected
