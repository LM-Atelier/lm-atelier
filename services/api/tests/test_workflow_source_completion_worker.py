from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy import select
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import (
    _approve,
    _plan,
    _state,
)
from test_workflow_source_completion import (
    source_runtime as source_runtime,
)

from local_lm import models, workflow_source_completion
from local_lm.db import SessionLocal
from local_lm.workflow_completion_jobs import WorkflowCompletionJobError
from local_lm.workflow_offer_completion import WorkflowOfferCompletionError


@pytest.mark.parametrize("moment", ["before-transaction", "after-commit"])
async def test_final_completion_worker_retains_its_outcome_and_lease_through_cancellation(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
    moment: str,
) -> None:
    services = app.state.services
    monkeypatch.setattr(services.downloads, "start_workflow_installation", lambda _offer_id: None)
    offer_id = await _approve(client, app, await _plan(client))
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    acquired = asyncio.Event()
    original = workflow_source_completion._finish
    main_thread = threading.get_ident()

    def hold() -> None:
        assert threading.get_ident() != main_thread
        entered.set()
        if not release.wait(30):
            raise AssertionError("Final completion was not released")

    def finish(*args: Any, **kwargs: Any) -> str:
        try:
            if moment == "before-transaction":
                hold()
            result = original(*args, **kwargs)
            if moment == "after-commit":
                hold()
            return result
        finally:
            finished.set()

    monkeypatch.setattr(workflow_source_completion, "_finish", finish)

    async def complete() -> str | None:
        with SessionLocal() as session:
            offer = session.get(models.WorkflowInstallOffer, offer_id)
            assert offer is not None and offer.completion_job_id is not None
            completion_job_id = offer.completion_job_id
        async with services.scheduler.job_lease(
            completion_job_id, resource="media_compute", group="primary"
        ) as claim:
            return await workflow_source_completion.complete_workflow_source(
                services.settings, services.processes, services.engines.media, offer_id, claim=claim
            )

    async def take_lease() -> None:
        async with services.scheduler.lease("primary"):
            acquired.set()

    async def waiting() -> bool:
        return entered.is_set()

    async def settled() -> bool:
        return finished.is_set()

    completion = asyncio.create_task(complete())
    waiter: asyncio.Task[None] | None = None
    try:
        await wait_until(waiting, bool, what="final source completion")
        assert _state(offer_id)[0] == ("completed" if moment == "after-commit" else "queued")
        waiter = asyncio.create_task(take_lease())
        completion.cancel()
        await asyncio.sleep(0)
        completion.cancel()
        await asyncio.sleep(0)
        assert not completion.done() and not acquired.is_set()
        release.set()
        activation_id = await completion
        assert activation_id is not None and _state(offer_id)[0] == "completed"
        await waiter
        with SessionLocal() as session:
            offer = session.get(models.WorkflowInstallOffer, offer_id)
            assert offer is not None
            active = list(session.scalars(select(models.WorkflowActivation)))
            assert len(active) == 1 and active[0].id == activation_id and active[0].is_active
        assert acquired.is_set()
    finally:
        release.set()
        await asyncio.gather(
            completion, *([waiter] if waiter is not None else []), return_exceptions=True
        )
        if entered.is_set():
            await wait_until(settled, bool, what="final completion worker cleanup")


@pytest.mark.parametrize("replacement", ["token", "released", "attempt", "none"])
async def test_final_source_activation_requires_its_original_claim(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
    replacement: str,
) -> None:
    services = app.state.services
    monkeypatch.setattr(services.downloads, "start_workflow_installation", lambda _offer_id: None)
    offer_id = await _approve(client, app, await _plan(client))
    with SessionLocal() as session:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert offer is not None and offer.completion_job_id is not None
        job_id, draft_id = offer.completion_job_id, offer.workflow_revision_id
    entered, release = threading.Event(), threading.Event()
    original = workflow_source_completion._finish

    def finish(*args: Any, **kwargs: Any) -> str:
        entered.set()
        assert release.wait(30), "Final activation was not released"
        return original(*args, **kwargs)

    monkeypatch.setattr(workflow_source_completion, "_finish", finish)

    async def complete() -> str | None:
        async with services.scheduler.job_lease(
            job_id, resource="media_compute", group="primary"
        ) as claim:
            return await workflow_source_completion.complete_workflow_source(
                services.settings, services.processes, services.engines.media, offer_id, claim=claim
            )

    completion = asyncio.create_task(complete())
    try:
        async with asyncio.timeout(30):
            while not entered.is_set():
                assert not completion.done(), "Completion stopped before final activation"
                await asyncio.sleep(0.01)
        with SessionLocal() as session:
            job = session.get(models.Job, job_id)
            assert job is not None and job.status == "running" and job.claim_owner is not None
            if replacement == "token":
                job.claim_owner = "replacement-claim"
            elif replacement == "released":
                job.claim_owner = None
            elif replacement == "attempt":
                job.attempt += 1
            result_before = dict(job.result_json)
            session.commit()
        release.set()
        outcome = (await asyncio.gather(completion, return_exceptions=True))[0]
        if replacement == "none":
            assert isinstance(outcome, str)
        else:
            assert isinstance(outcome, (WorkflowOfferCompletionError, WorkflowCompletionJobError))
        with SessionLocal() as session:
            offer = session.get(models.WorkflowInstallOffer, offer_id)
            job = session.get(models.Job, job_id)
            assert offer is not None and job is not None
            active = list(session.scalars(select(models.WorkflowActivation)))
            if replacement == "none":
                assert offer.status == "completed" and job.status == "complete"
                assert len(active) == 1 and active[0].id == outcome
            else:
                assert offer.status == "queued" and offer.workflow_revision_id == draft_id
                assert job.status == "running" and job.result_json == result_before
                assert not active
    finally:
        release.set()
        await asyncio.gather(completion, return_exceptions=True)


@pytest.mark.parametrize("replacement", ["token", "released", "none"])
@pytest.mark.parametrize(
    "boundary", ["runtime-record", "extension-start", "extension-review", "restore-warning"]
)
async def test_source_runtime_writes_retain_the_original_claim(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
    replacement: str,
    boundary: str,
) -> None:
    services = app.state.services
    monkeypatch.setattr(services.downloads, "start_workflow_installation", lambda _offer_id: None)
    offer_id = await _approve(client, app, await _plan(client))
    with SessionLocal() as session:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert offer is not None and offer.completion_job_id is not None
        job_id, draft_id = offer.completion_job_id, offer.workflow_revision_id
    replaced = False

    def change_claim() -> None:
        nonlocal replaced
        assert not replaced
        replaced = True
        with SessionLocal() as session:
            job = session.get(models.Job, job_id)
            assert job is not None and job.claim_owner is not None
            assert job.status == ("complete" if boundary == "restore-warning" else "running")
            if replacement == "token":
                job.claim_owner = "replacement-claim"
            elif replacement == "released":
                job.claim_owner = None
            session.commit()

    if boundary == "runtime-record":
        provisioner = services.processes.runtimes
        original_provision = provisioner.provision

        async def provision(*args: Any, **kwargs: Any) -> Any:
            result = await original_provision(*args, **kwargs)
            change_claim()
            return result

        monkeypatch.setattr(provisioner, "provision", provision)
    else:
        original_start = services.processes.start_media

        async def start(*args: Any, **kwargs: Any) -> Any:
            scoped = kwargs.get("activation_scope") is not None
            if boundary == "restore-warning" and not scoped:
                change_claim()
                raise OSError("Neutral worker restoration failure")
            result = await original_start(*args, **kwargs)
            if scoped and boundary != "restore-warning":
                change_claim()
                if boundary == "extension-review":
                    raise WorkflowOfferCompletionError("workflow-extension-review-required")
            return result

        monkeypatch.setattr(services.processes, "start_media", start)

    async def complete() -> str | None:
        async with services.scheduler.job_lease(
            job_id, resource="media_compute", group="primary"
        ) as claim:
            return await workflow_source_completion.complete_workflow_source(
                services.settings, services.processes, services.engines.media, offer_id, claim=claim
            )

    outcome = (await asyncio.gather(complete(), return_exceptions=True))[0]
    assert replaced
    if replacement != "none":
        assert isinstance(outcome, (WorkflowOfferCompletionError, WorkflowCompletionJobError))
    elif boundary == "extension-review":
        assert outcome is None
    else:
        assert isinstance(outcome, str)
    with SessionLocal() as session:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        job = session.get(models.Job, job_id)
        assert offer is not None and job is not None
        active = list(session.scalars(select(models.WorkflowActivation)))
        committed = boundary == "restore-warning" or (
            replacement == "none" and boundary != "extension-review"
        )
        assert offer.status == ("completed" if committed else "queued")
        assert len(active) == int(committed)
        if not committed:
            assert offer.workflow_revision_id == draft_id
        if replacement != "none":
            assert offer.completion_error_code is None
            assert job.status == ("complete" if committed else "running")
            if boundary == "runtime-record":
                assert job.result_json == {}
        elif boundary == "extension-review":
            assert job.status == "paused"
            assert offer.completion_error_code == "workflow-extension-review-required"
        elif boundary == "restore-warning":
            assert offer.completion_error_code == "workflow-media-restore-failed"
