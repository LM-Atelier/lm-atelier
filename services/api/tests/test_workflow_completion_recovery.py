"""Changed accepted state must not prevent unrelated startup recovery."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_workflow_completion_jobs import _accept
from test_workflow_offer_packages import _accepted, _prepare, _setup
from test_workflow_package_import_endpoint import _ui_graph
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import source_runtime as source_runtime

from local_lm import models
from local_lm.comfy_registry_lifecycle import ComfyRegistryPreparation
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.main import create_app
from local_lm.workflow_completion_jobs import stage_workflow_completion_job
from local_lm.workflow_offer_packages import record_workflow_offer_package

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("status", ["cancelled", "interrupted"])
@pytest.mark.parametrize("change", ["revision", "deleted-package", "wheel-environment"])
async def test_startup_reports_changed_installation_and_keeps_stopped_work_inert(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
    status: str,
    change: str,
) -> None:
    manager = app.state.services.downloads
    monkeypatch.setattr(manager, "start_workflow_installation", lambda _id: None)
    if change == "revision":
        _plan_id, offer_id, job_id = await _accept(client)
    else:
        inputs, context, offer_id, saved = await _setup(
            tmp_path,
            available_nodes=frozenset(source_runtime),
            source_payload={
                "name": "Neutral extension recovery",
                "operation": "text_to_image",
                "ui_graph": _ui_graph(),
                "dependencies": {"version": 1, "slots": []},
                "selections": [],
            },
        )
        with SessionLocal() as session:
            offer = session.get(models.WorkflowInstallOffer, offer_id)
            assert offer is not None
            job_id = stage_workflow_completion_job(session, offer).id
            child = _accepted(session, offer_id, saved)
            child.job.status = "running"
            preparation_job_id = child.job.id
            session.commit()

        def record(session: Session, preparation: ComfyRegistryPreparation) -> None:
            offer = session.get(models.WorkflowInstallOffer, offer_id)
            assert offer is not None
            record_workflow_offer_package(
                session,
                offer,
                saved,
                package_id=inputs["selected"].package_id,
                job_id=preparation_job_id,
                preparation=preparation,
            )

        prepared = await _prepare(inputs, context, saved, record)

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    with SessionLocal() as session:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert offer is not None
        revision = session.get(models.WorkflowRevision, offer.workflow_revision_id)
        assert revision is not None
        workflow_id = revision.workflow_id
        if change != "revision":
            package = session.get(models.ComfyRegistryInstall, prepared.install_id)
            assert package is not None and not package.active and not package.trusted
            if change == "deleted-package":
                session.delete(package)
            else:
                package.wheel_environment_sha256 = "f" * 64
        job = session.get(models.Job, job_id)
        assert job is not None
        job.status = status
        session.commit()
    if change == "revision":
        changed = await client.post(
            f"/api/workflows/{workflow_id}/revisions",
            json={"api_graph": {}, "ui_graph": _ui_graph()},
        )
        assert changed.status_code == 201, changed.text

    restarted = create_app(
        settings.model_copy(update={"chat_engine": "mock", "media_engine": "mock"})
    )
    async with restarted.router.lifespan_context(restarted):
        await asyncio.wait_for(restarted.state.retention_sweep, timeout=30)
        recovery = restarted.state.services.downloads._offer_recovery_task
        assert recovery is not None
        await asyncio.wait_for(recovery, timeout=30)
        async with AsyncClient(
            transport=ASGITransport(app=restarted), base_url="http://testserver"
        ) as reader:
            ready = await reader.get("/api/ready")
            assert ready.status_code == 200, ready.text
        with SessionLocal() as session:
            offer = session.get(models.WorkflowInstallOffer, offer_id)
            job = session.get(models.Job, job_id)
            assert offer is not None and job is not None
            assert offer.completion_error_code == "workflow-completion-unavailable"
            assert job.status == ("cancelled" if status == "cancelled" else "failed")
            assert list(session.scalars(select(models.WorkflowActivation))) == []
            assert not restarted.state.services.downloads._offer_tasks
