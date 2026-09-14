"""Download approval must use the same durable workflow review as activation."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_workflow_install_offer_api import _offer_payload
from test_workflow_offer_download_acceptance import _created_offer, _links
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.models import Job, WorkflowInstallOffer, WorkflowRevision
from local_lm.workflow_install_offers import current_reviewed_workflow_install_offer

pytestmark = pytest.mark.asyncio


def _identity(offer_id: str) -> tuple[str, str]:
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        revision = session.get(WorkflowRevision, offer.workflow_revision_id)
        assert revision is not None
        return revision.workflow_id, revision.id


def _change_reviewed_metadata(revision_id: str, field: str) -> None:
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None and revision.trusted
        if field == "engine_version":
            revision.engine_version = "changed-runtime-version"
        else:
            revision.capabilities_json = ["seed"]
        session.commit()


@pytest.mark.parametrize("field", ["engine_version", "capabilities"])
@pytest.mark.parametrize("action", ["create", "project", "install"])
async def test_stale_workflow_review_cannot_authorize_a_download_offer(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, field: str, action: str
) -> None:
    offer_id = await _created_offer(app, client, monkeypatch)
    workflow, revision = _identity(offer_id)
    started: list[str] = []
    monkeypatch.setattr(DownloadManager, "start", lambda _manager, job_id: started.append(job_id))
    _change_reviewed_metadata(revision, field)
    review = await client.get(f"/api/workflows/{workflow}/revisions/{revision}/review")
    assert review.status_code == 200 and not review.json()["trusted"], review.text

    if action == "project":
        with SessionLocal() as session:
            assert (
                current_reviewed_workflow_install_offer(
                    session, workflow_id=workflow, revision_id=revision
                )
                is None
            )
    else:
        response = (
            await client.post(
                f"/api/workflows/{workflow}/revisions/{revision}/install-offers",
                json=_offer_payload("accepted-plan"),
            )
            if action == "create"
            else await client.post(f"/api/workflow-install-offers/{offer_id}/install")
        )
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "workflow-revision-needs-attention"
    assert started == [] and _links(offer_id) == []
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Job)) == 0
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        assert offer.status == ("invalidated" if action == "install" else "ready")


@pytest.mark.parametrize("change", ["none", "compatible_runtime", "fresh_review"])
async def test_current_review_keeps_the_exact_asset_offer_actionable(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    offer_id = await _created_offer(app, client, monkeypatch)
    workflow, revision = _identity(offer_id)
    review_url = f"/api/workflows/{workflow}/revisions/{revision}/review"
    if change == "compatible_runtime":
        original_info = app.state.services.engines.media.object_info

        async def object_info() -> dict[str, Any]:
            info: dict[str, Any] = await original_info()
            info["EmptyLatentImage"]["input"]["optional"] = {
                "new_optional_setting": ["INT", {"default": 0}]
            }
            return info

        monkeypatch.setattr(app.state.services.engines.media, "object_info", object_info)
    elif change == "fresh_review":
        _change_reviewed_metadata(revision, "engine_version")
        preview = await client.get(review_url)
        assert preview.status_code == 200 and not preview.json()["trusted"]
        approved = await client.post(
            review_url,
            json={"action": "approve", "subject_sha256": preview.json()["subject_sha256"]},
        )
        assert approved.status_code == 200 and approved.json()["trusted"], approved.text

    review = await client.get(review_url)
    assert review.status_code == 200 and review.json()["trusted"], review.text
    with SessionLocal() as session:
        offer = current_reviewed_workflow_install_offer(
            session, workflow_id=workflow, revision_id=revision
        )
        assert offer is not None and offer.id == offer_id
    started: list[str] = []
    monkeypatch.setattr(DownloadManager, "start", lambda _manager, job_id: started.append(job_id))
    response = await client.post(f"/api/workflow-install-offers/{offer_id}/install")
    assert response.status_code == 202, response.text
    assert started == [response.json()[0]["id"]]
    assert len(_links(offer_id)) == 1
