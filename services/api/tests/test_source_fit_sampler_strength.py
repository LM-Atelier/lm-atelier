"""An extension samples at full strength even where the workflow maps its strength."""

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
from local_lm.models import Job, Message, Run, WorkflowRevision
from local_lm.output_origin import stated_origin
from local_lm.scheduler import JobClaim


async def test_an_extension_runs_a_mapped_sampler_strength_at_full(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch, runtime_sampler=True)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, request["workflow_revision_id"])
        assert revision is not None
        # Saving the workflow turned the sampler's fixed full strength into a setting.
        assert revision.api_graph_json["sample"]["inputs"]["denoise"] == "${denoise}"
    orchestrator = app.state.services.orchestrator
    sent: list[dict[str, Any]] = []
    with Image.new("RGB", (4, 5), "blue") as image:
        content = BytesIO()
        image.save(content, format="PNG")

    async def generate(media: MediaRequest) -> AsyncIterator[MediaEvent]:
        sent.append(dict(media.parameters))
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
                "settings": {"steps": 2, "denoise": 0.4},
                "source_fit": {"mode": "extend", "width": 4, "height": 5},
            },
        )
        assert response.status_code == 202, response.text
        accepted = response.json()
        with SessionLocal() as session:
            run = session.get(Run, accepted["run"]["id"])
            assert run is not None
            snapshot = accepted_context(session, run)
            assert snapshot is not None and snapshot.source_fit is not None
            # Recorded as it runs, with no automatic strength for a verifier to raise.
            assert snapshot.settings["denoise"] == 1
            assert snapshot.image_edit_strength is None
            job = session.scalar(select(Job).where(Job.run_id == run.id))
            assert job is not None
            claim = JobClaim(token="source-fit-strength", attempt=1)
            job.status = "running"
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id = job.id
            session.commit()
        await orchestrator._execute_media(job_id, accepted["run"]["id"], claim)
    with SessionLocal() as session:
        assistant = session.get(Message, accepted["assistant_message"]["id"])
        assert assistant is not None and assistant.status == "complete"
    assert [parameters["denoise"] for parameters in sent] == [1]
