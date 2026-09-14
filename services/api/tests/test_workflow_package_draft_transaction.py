"""Draft persistence shares the caller's commit and rollback boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_workflow_package_import_endpoint import _ui_graph

from local_lm import workflow_package_drafts as drafts
from local_lm.db import SessionLocal
from local_lm.models import Job, WorkflowDefinition, WorkflowRevision
from local_lm.schemas import WorkflowPackageDraftRequest

pytestmark = pytest.mark.asyncio

DraftWriter = Callable[
    [Session, WorkflowPackageDraftRequest], tuple[WorkflowDefinition, WorkflowRevision]
]


def _writer() -> DraftWriter:
    writer = getattr(drafts, "stage_workflow_package_draft", None)
    assert callable(writer), "Package drafts need a caller-owned transaction."
    return cast(DraftWriter, writer)


def _payload() -> WorkflowPackageDraftRequest:
    return WorkflowPackageDraftRequest.model_validate(
        {"name": "Neutral source", "operation": "text_to_image", "ui_graph": _ui_graph()}
    )


async def test_draft_and_related_work_become_visible_only_after_the_callers_commit(
    client: AsyncClient,
) -> None:
    writer = _writer()
    with SessionLocal() as session:
        job = Job(kind="download", payload_json={"label": "Neutral related work"})
        session.add(job)
        definition, revision = writer(session, _payload())
        session.flush()
        workflow_id, revision_id, job_id = definition.id, revision.id, job.id
        assert definition.current_revision_id == revision_id
        assert not revision.trusted and revision.api_graph_json == {}
        assert revision.dependency_contract_sha256 is None
        assert revision.artifact_sha256 is not None
        with SessionLocal() as observer:
            assert observer.get(WorkflowDefinition, workflow_id) is None
            assert observer.get(WorkflowRevision, revision_id) is None
            assert observer.get(Job, job_id) is None
        session.commit()
    with SessionLocal() as observer:
        assert observer.get(WorkflowDefinition, workflow_id) is not None
        assert observer.get(WorkflowRevision, revision_id) is not None
        assert observer.get(Job, job_id) is not None


@pytest.mark.parametrize("failure", ["contract", "after-stage", "after-flush"])
async def test_interrupted_draft_staging_rolls_back_the_whole_transaction(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    writer = _writer()
    with SessionLocal() as session:
        existing = Job(kind="download", payload_json={"label": "Previously committed"})
        session.add(existing)
        session.commit()
        existing_id = existing.id
        before = {
            model.__tablename__: set(session.scalars(select(model.id)))
            for model in (WorkflowDefinition, WorkflowRevision, Job)
        }
    if failure == "contract":

        def interrupt(_session: Session, _revision: WorkflowRevision) -> None:
            raise RuntimeError("Neutral interrupted transaction")

        monkeypatch.setattr(drafts, "persist_dependency_contract", interrupt)
    with (
        pytest.raises(RuntimeError, match="Neutral interrupted transaction"),
        SessionLocal() as session,
    ):
        session.add(Job(kind="download", payload_json={"label": "Uncommitted"}))
        writer(session, _payload())
        if failure == "after-flush":
            session.flush()
        raise RuntimeError("Neutral interrupted transaction")
    with SessionLocal() as observer:
        assert observer.get(Job, existing_id) is not None
        assert {
            model.__tablename__: set(observer.scalars(select(model.id)))
            for model in (WorkflowDefinition, WorkflowRevision, Job)
        } == before


async def test_reopening_a_draft_does_not_commit_metadata_or_related_work(
    client: AsyncClient,
) -> None:
    writer = _writer()
    response = await client.post("/api/workflows/packages/drafts", json=_payload().model_dump())
    assert response.status_code == 201, response.text
    workflow_id = response.json()["id"]
    updated = _payload().model_copy(update={"name": "Uncommitted rename"})
    with SessionLocal() as session:
        job = Job(kind="download", payload_json={"label": "Uncommitted related work"})
        session.add(job)
        definition, _revision = writer(session, updated)
        session.flush()
        job_id = job.id
        assert definition.name == "Uncommitted rename"
        with SessionLocal() as observer:
            saved = observer.get(WorkflowDefinition, workflow_id)
            assert saved is not None and saved.name == "Neutral source"
            assert observer.get(Job, job_id) is None
        session.rollback()
    restored = await client.get(f"/api/workflows/{workflow_id}")
    assert restored.status_code == 404, restored.text
    with SessionLocal() as observer:
        saved = observer.get(WorkflowDefinition, workflow_id)
        assert saved is not None and saved.name == "Neutral source"
        assert observer.get(Job, job_id) is None


async def test_unicode_source_keeps_the_same_draft_identity_across_entry_points(
    client: AsyncClient,
) -> None:
    payload = _payload()
    payload.ui_graph["nodes"][0]["widgets_values"][0] = "caf\u00e9"
    response = await client.post("/api/workflows/packages/drafts", json=payload.model_dump())
    assert response.status_code == 201, response.text
    assert response.json()["id"] == "wfpkgdraft_4bb4d299c1d0bbca7023e417"
    assert response.json()["current_revision_id"] == "wfpkgdrev_4bb4d299c1d0bbca7023e417"
    writer = _writer()
    with SessionLocal() as session:
        definition, revision = writer(session, payload)
        assert definition.id == response.json()["id"]
        assert revision.id == response.json()["current_revision_id"]
        assert revision.ui_graph_json == payload.ui_graph
        assert not revision.trusted and revision.api_graph_json == {}
