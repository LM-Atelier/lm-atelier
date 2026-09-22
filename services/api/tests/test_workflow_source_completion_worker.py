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
        async with services.scheduler.lease("primary"):
            return await workflow_source_completion.complete_workflow_source(
                services.settings, services.processes, services.engines.media, offer_id
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
