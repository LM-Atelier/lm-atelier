"""Created contracts can reach an activation and an admitted turn without seeded authority."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_workflow_revision_review import _GRAPH
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.models import Job, Run, WorkflowActivation, WorkflowRevision

pytestmark = pytest.mark.asyncio


async def test_created_reviewed_contract_can_activate_and_admit_a_turn(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The adapter fixture supplies neutral node metadata; this measures API
    # admission and saved identity, not a live media generation.
    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    profile = await client.post(
        "/api/profiles",
        json={
            "name": "Neutral admission profile",
            "role": "image",
            "engine": "comfyui",
        },
    )
    assert profile.status_code == 201, profile.text
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral contract integration",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _GRAPH,
            "dependencies": {"version": 1, "slots": []},
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()["id"]
    revision = created.json()["current_revision_id"]
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None and row.dependency_contract_sha256 is not None
        assert not row.trusted
        assert session.scalar(select(func.count()).select_from(WorkflowActivation)) == 0
    review_url = f"/api/workflows/{workflow}/revisions/{revision}/review"
    review = await client.get(review_url)
    assert review.status_code == 200, review.text
    approved = await client.post(
        review_url,
        json={
            "action": "approve",
            "subject_sha256": review.json()["subject_sha256"],
        },
    )
    assert approved.status_code == 200 and approved.json()["trusted"], approved.text
    activation_url = f"/api/workflows/{workflow}/revisions/{revision}/activation"
    subject = await client.get(activation_url)
    assert subject.status_code == 200, subject.text
    chat = await client.post("/api/chats", json={"title": "Neutral activation integration"})
    assert chat.status_code == 201, chat.text
    turn_url = f"/api/chats/{chat.json()['id']}/turns"
    payload = {
        "text": "A blue landscape",
        "mode": "image",
        "profile_id": profile.json()["id"],
        "workflow_revision_id": revision,
        "idempotency_key": "activation-integration-turn",
    }
    async with app.state.services.scheduler.lease("primary"):
        refused = await client.post(turn_url, json=payload)
        assert refused.status_code == 422, refused.text
        with SessionLocal() as session:
            assert session.scalar(select(func.count()).select_from(Run)) == 0
            assert session.scalar(select(func.count()).select_from(Job)) == 0
        activated = await client.post(
            activation_url,
            json={
                "workflow_artifact_sha256": subject.json()["workflow_artifact_sha256"],
                "dependency_contract_sha256": subject.json()["dependency_contract_sha256"],
                "selections": [],
            },
        )
        assert activated.status_code == 200, activated.text
        accepted = await client.post(turn_url, json=payload)
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["run"]["workflow_revision_id"] == revision
        assert accepted.json()["run"]["profile_id"] == profile.json()["id"]
        with SessionLocal() as session:
            accepted_run = session.get(Run, accepted.json()["run"]["id"])
            assert accepted_run is not None
            assert session.scalar(select(func.count()).select_from(WorkflowActivation)) == 1
            assert (
                accepted_run.provenance_json["workflow"]["activation"]["id"]
                == activated.json()["id"]
            )
