"""Regeneration keeps the accepted canvas, source pixels and execution choices."""

from __future__ import annotations

from collections.abc import AsyncIterator
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select
from test_source_fit_acceptance import prepared_turn
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.accepted_turn_context import accepted_context
from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Chat, Job, Message, ModelProfile, Run
from local_lm.output_origin import stated_origin
from local_lm.scheduler import JobClaim
from local_lm.source_fit_image import replay_source_fit_image


async def completed_source(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    orchestrator = app.state.services.orchestrator
    with Image.new("RGB", (4, 5), "blue") as image:
        content = BytesIO()
        image.save(content, format="PNG")

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        assert request.input_contents is not None
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=content.getvalue(),
                    kind="image",
                    media_type="image/png",
                    name="canvas.png",
                    origin=stated_origin("save", "output", "images"),
                ),
            ],
        )

    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)
    monkeypatch.setattr(orchestrator, "_ensure_media_worker", AsyncMock())
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                **request,
                "source_fit": {"mode": "extend", "width": 4, "height": 5},
            },
        )
        assert response.status_code == 202, response.text
        original: dict[str, Any] = response.json()
        with SessionLocal() as session:
            job = session.scalar(select(Job).where(Job.run_id == original["run"]["id"]))
            assert job is not None
            claim = JobClaim(token="source-fit-regeneration", attempt=1)
            job.status = "running"
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id = job.id
            session.commit()
        await orchestrator._execute_media(job_id, original["run"]["id"], claim)
        with SessionLocal() as session:
            assistant = session.get(Message, original["assistant_message"]["id"])
            assert assistant is not None and assistant.status == "complete"
    return original


@pytest.mark.parametrize(
    "choice",
    [
        "inherit",
        "original_changed",
        "prepared_changed",
        "later_defaults",
        "explicit_steps",
    ],
)
async def test_regeneration_keeps_the_accepted_source_canvas(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
) -> None:
    original = await completed_source(app, client, monkeypatch)
    orchestrator = app.state.services.orchestrator
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            source = session.get(Run, original["run"]["id"])
            assert source is not None
            frozen = accepted_context(session, source)
            assert frozen is not None and frozen.source_fit is not None
            expected_bytes = replay_source_fit_image(
                session,
                orchestrator.artifacts,
                frozen.source_fit.image,
                selected_source_id=frozen.input_artifact_ids[0],
            ).content
            if choice in {"original_changed", "prepared_changed"}:
                image = frozen.source_fit.image
                artifact = session.get(
                    Artifact,
                    (
                        image.source_artifact_id
                        if choice == "original_changed"
                        else image.prepared_artifact_id
                    ),
                )
                assert artifact is not None
                orchestrator.artifacts.resolve(artifact).write_bytes(b"changed neutral fixture")
            if choice == "later_defaults":
                profile = session.get(ModelProfile, frozen.profile_id)
                chat = session.get(Chat, frozen.chat_id)
                assert profile is not None and chat is not None
                profile.request_settings_json = {"steps": 7}
                chat.generation_settings_json = {"image": {"steps": 9}}
                session.commit()
        payload = {
            "idempotency_key": "source-canvas-regeneration",
            "settings": {"steps": 3} if choice == "explicit_steps" else {},
        }
        url = f"/api/messages/{original['assistant_message']['id']}/regenerate"
        response = await client.post(url, json=payload)
        if choice == "prepared_changed":
            assert response.status_code == 422, response.text
            return
        assert response.status_code == 202, response.text
        replay = await client.post(url, json=payload)
        assert replay.status_code == 202, replay.text
        assert replay.json()["run"]["id"] == response.json()["run"]["id"]
        with SessionLocal() as session:
            run = session.get(Run, response.json()["run"]["id"])
            assert run is not None
            snapshot = accepted_context(session, run)
            assert snapshot is not None and snapshot.source_fit == frozen.source_fit
            assert snapshot.profile == frozen.profile
            assert snapshot.workflow == frozen.workflow
            assert snapshot.settings["steps"] == (3 if choice == "explicit_steps" else 2)
            assert snapshot.input_artifact_ids == frozen.input_artifact_ids
            assert snapshot.source_fit is not None
            assert (
                replay_source_fit_image(
                    session,
                    orchestrator.artifacts,
                    snapshot.source_fit.image,
                    selected_source_id=frozen.input_artifact_ids[0],
                ).content
                == expected_bytes
            )
