"""Delayed installation cleanup cannot deactivate a replacement execution's packages."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_workflow_offer_packages import _accepted, _prepare, _setup

from local_lm import workflow_source_completion, workflow_source_extensions
from local_lm.comfy_registry_lifecycle import ComfyRegistryPreparation
from local_lm.db import SessionLocal
from local_lm.models import ComfyRegistryInstall, Job, WorkflowInstallOffer
from local_lm.workflow_completion_jobs import stage_workflow_completion_job
from local_lm.workflow_offer_packages import record_workflow_offer_package


@pytest.mark.parametrize("replacement", ["token", "attempt", "released", "none", "queued-retry"])
@pytest.mark.parametrize("boundary", ["quarantine", "release"])
async def test_delayed_quarantine_keeps_replacement_packages_active(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replacement: str,
    boundary: str,
) -> None:
    inputs, context, offer_id, saved = await _setup(
        tmp_path,
        source_payload={
            "name": "Neutral extension cleanup",
            "operation": "text_to_image",
            "ui_graph": {"version": 0.4, "nodes": [], "links": []},
            "dependencies": {"version": 1, "slots": []},
            "selections": [],
        },
    )
    with SessionLocal() as session:
        item = _accepted(session, offer_id, saved)
        item.job.status = "running"
        package_job_id = item.job.id
        session.commit()

    def record(session: Any, preparation: ComfyRegistryPreparation) -> None:
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        record_workflow_offer_package(
            session,
            offer,
            saved,
            package_id=inputs["selected"].package_id,
            job_id=package_job_id,
            preparation=preparation,
        )

    preparation = await _prepare(inputs, context, saved, record)
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        job = stage_workflow_completion_job(session, offer)
        job_id = job.id
        install = session.get(ComfyRegistryInstall, preparation.install_id)
        assert install is not None
        install.active = True
        install.trusted = True
        install.review_json = {**install.review_json, "activation_batch_v1": {"state": "starting"}}
        session.commit()

    async def refuse(*_args: object, **_kwargs: object) -> None:
        raise ValueError("Neutral completion refusal")

    entered, release = threading.Event(), threading.Event()
    original = workflow_source_extensions.quarantine_workflow_source_extensions

    def delayed(*args: Any, **kwargs: Any) -> None:
        entered.set()
        assert release.wait(timeout=10)
        original(*args, **kwargs)

    manager = app.state.services.downloads
    monkeypatch.setattr(workflow_source_completion, "complete_workflow_source", refuse)
    if boundary == "quarantine":
        monkeypatch.setattr(
            workflow_source_extensions, "quarantine_workflow_source_extensions", delayed
        )
    else:
        original_release = manager.scheduler._release_job

        async def delayed_release(job_id: str, token: str, group: str) -> None:
            entered.set()
            async with asyncio.timeout(10):
                while not release.is_set():
                    await asyncio.sleep(0.01)
            await original_release(job_id, token, group)

        monkeypatch.setattr(manager.scheduler, "_release_job", delayed_release)
    publish = AsyncMock(wraps=manager.events.publish)
    monkeypatch.setattr(manager.events, "publish", publish)
    task = asyncio.create_task(manager.reconcile_workflow_install_offers(only_offer_id=offer_id))
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                assert not task.done(), "Completion returned before quarantine"
                await asyncio.sleep(0.01)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            offer = session.get(WorkflowInstallOffer, offer_id)
            assert job is not None and offer is not None
            assert job.status == "failed" and job.claim_owner is not None and job.attempt == 1
            if replacement == "token":
                job.status = "running"
                job.claim_owner = "replacement-claim"
            elif replacement == "attempt":
                job.status = "running"
                job.attempt += 1
            elif replacement == "released":
                job.claim_owner = None
            elif replacement == "queued-retry":
                job.status = "queued"
            if replacement != "none":
                offer.completion_error_code = None
            session.commit()
        release.set()
        await asyncio.wait_for(task, timeout=10)
        with SessionLocal() as session:
            install = session.get(ComfyRegistryInstall, preparation.install_id)
            assert install is not None and install.trusted
            assert install.active == (
                boundary == "quarantine" and replacement in {"token", "attempt", "released"}
            )
        attention = [
            call for call in publish.await_args_list if call.args[0] == "workflow.install.attention"
        ]
        assert len(attention) == (1 if replacement == "none" else 0)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await manager.close()
