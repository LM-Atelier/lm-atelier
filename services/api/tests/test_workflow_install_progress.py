"""Progress is a read-only account of durable installation state."""

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_workflow_offer_download_acceptance import _created_offer
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.models import Job, WorkflowInstallOffer

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    ("job_state", "phase", "count_field"),
    [
        ("queued", "downloading", "pending_downloads"),
        ("running", "downloading", "pending_downloads"),
        ("paused", "paused", "paused_downloads"),
        ("complete", "verifying", "completed_downloads"),
        ("failed", "needs_attention", "failed_downloads"),
        ("cancelled", "needs_attention", "cancelled_downloads"),
        ("missing", "needs_attention", "unavailable_downloads"),
    ],
)
async def test_progress_reports_accepted_job_state_without_starting_work(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    job_state: str,
    phase: str,
    count_field: str,
) -> None:
    offer_id = await _created_offer(app, client, monkeypatch)
    url = f"/api/workflow-install-offers/{offer_id}/progress"
    ready = await client.get(url)
    assert ready.status_code == 200 and ready.json()["phase"] == "ready"
    assert ready.json()["unavailable_downloads"] == 0
    starts: list[str] = []
    monkeypatch.setattr(app.state.services.downloads, "start", starts.append)
    accepted = await client.post(f"/api/workflow-install-offers/{offer_id}/install")
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()[0]["id"]
    assert starts == [job_id]
    starts.clear()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        if job_state == "missing":
            session.delete(job)
        else:
            job.status = job_state
        session.commit()
    result = await client.get(url)
    assert result.status_code == 200, result.text
    progress = result.json()
    assert progress["status"] == "queued" and progress["phase"] == phase
    assert progress["total_downloads"] == 1 and progress[count_field] == 1
    assert (
        sum(
            progress[field]
            for field in (
                "pending_downloads",
                "completed_downloads",
                "failed_downloads",
                "cancelled_downloads",
                "paused_downloads",
                "unavailable_downloads",
            )
        )
        == 1
    )
    assert starts == []
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None and offer.status == "queued" and offer.completed_at is None


async def test_unknown_attention_code_is_not_exposed(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    offer_id = await _created_offer(app, client, monkeypatch)
    monkeypatch.setattr(app.state.services.downloads, "start", lambda _job_id: None)
    accepted = await client.post(f"/api/workflow-install-offers/{offer_id}/install")
    assert accepted.status_code == 202
    marker = "neutral-unrecognized-error-marker"
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        offer.completion_error_code = marker
        session.commit()
    response = await client.get(f"/api/workflow-install-offers/{offer_id}/progress")
    assert response.status_code == 200
    assert response.json()["attention_code"] == "workflow-completion-unavailable"
    assert marker not in response.text
    missing = await client.get("/api/workflow-install-offers/unknown/progress")
    assert missing.status_code == 404
    assert missing.json()["code"] == "workflow-install-offer-not-found"
