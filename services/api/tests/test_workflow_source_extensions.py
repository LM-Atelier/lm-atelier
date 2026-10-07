"""Accepted preparations resume their exact jobs without rediscovering or running code."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_offer_packages import _accepted, _setup

from local_lm.comfy_registry_downloads import ComfyRegistryArchiveDownloader
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.db import SessionLocal
from local_lm.models import ComfyRegistryInstall, Job, WorkflowInstallOffer

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "mode",
    [
        "ordinary",
        "resume-running",
        "paused",
        "cancelled",
        "failed",
        "worker-running",
        "bad-archive",
        "cancel",
        "invalidate",
    ],
)
async def test_accepted_preparation_uses_durable_jobs_and_never_activates_packages(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from local_lm.workflow_source_extensions import (
        ExtensionPreparationServices,
        prepare_workflow_source_extensions,
    )

    inputs, context, offer_id, saved = await _setup(tmp_path)
    with SessionLocal() as session:
        item = _accepted(session, offer_id, saved)
        job_id = item.job.id
        if mode in {"paused", "cancelled", "failed"}:
            item.job.status = mode
        elif mode == "resume-running":
            item.job.status = "running"
            item.job.attempt = 1
        session.commit()
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if mode == "invalidate":
            with SessionLocal() as session:
                offer = session.get(WorkflowInstallOffer, offer_id)
                assert offer is not None
                offer.status = "invalidated"
                session.commit()
        return httpx.Response(
            200, content=b"invalid" if mode == "bad-archive" else inputs["state"]["content"]
        )

    archive = ComfyRegistryArchiveDownloader(transport=httpx.MockTransport(respond))
    wheels = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    )
    entered = asyncio.Event()
    if mode == "cancel":

        async def hold(*_args: object) -> Any:
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(inputs["registry_client"], "resolve", hold)
    services = ExtensionPreparationServices(
        inputs["interpreter_probe"],
        inputs["registry_client"],
        inputs["project_client"],
        inputs["metadata_client"],
        archive,
        wheels,
    )
    try:

        async def run() -> Any:
            return await prepare_workflow_source_extensions(
                SessionLocal,
                offer_id,
                context=context,
                media_worker_stopped=mode != "worker-running",
                services=services,
            )

        if mode in {"ordinary", "resume-running"}:
            result = await run()
            assert len(result.preparations) == len(result.execution_plans) == 1
            count = calls
            assert count == 1

            async def forbidden(*_args: object) -> Any:
                raise AssertionError("A completed preparation was restarted")

            monkeypatch.setattr(inputs["registry_client"], "resolve", forbidden)
            assert await run() == result and calls == count
            with SessionLocal() as session:
                item = _accepted(session, offer_id, saved)
                assert item.preparation == result.preparations[0]
                assert item.job.attempt == (2 if mode == "resume-running" else 1)
                rows = list(session.scalars(select(ComfyRegistryInstall)))
                assert len(rows) == 1 and not rows[0].trusted and not rows[0].active
        else:
            if mode == "cancel":
                task = asyncio.create_task(run())
                await asyncio.wait_for(entered.wait(), 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(ValueError):
                    await run()
            if mode == "invalidate":
                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    assert job is not None and job.status == "failed" and job.result_json == {}
                    assert list(session.scalars(select(ComfyRegistryInstall))) == []
                assert list(context.custom_node_root.iterdir()) == []
                return
            with SessionLocal() as session:
                item = _accepted(session, offer_id, saved)
                assert item.preparation is None
                expected = (
                    "queued"
                    if mode == "worker-running"
                    else "failed"
                    if mode == "bad-archive"
                    else "cancelled"
                    if mode == "cancel"
                    else mode
                )
                assert item.job.status == expected
                assert list(session.scalars(select(ComfyRegistryInstall))) == []
            if mode != "bad-archive":
                assert calls == 0
    finally:
        await archive.close()
        await wheels.close()
