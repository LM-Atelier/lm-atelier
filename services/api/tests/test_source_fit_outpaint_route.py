"""An extension runs on the outpainting shape real templates use, source intact."""

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

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.models import Job, Run, WorkflowRevision
from local_lm.output_origin import stated_origin
from local_lm.scheduler import JobClaim


async def test_the_official_outpaint_shape_extends_and_keeps_every_source_pixel(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_id, request = await prepared_turn(
        app, client, monkeypatch, conditioning=True, runtime_sampler=True
    )
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, request["workflow_revision_id"])
        assert revision is not None
        stored = revision.api_graph_json
    # Saved as a real runtime would leave it: whole-pixel pad sides, a feathered
    # mask, conditioning that carries the source, and a mapped strength.
    assert stored["pad"]["inputs"]["feathering"] == 24
    assert stored["sample"]["inputs"]["latent_image"] == ["condition", 2]
    assert stored["sample"]["inputs"]["denoise"] == "${denoise}"
    assert "composite" not in stored

    orchestrator = app.state.services.orchestrator
    sent: list[MediaRequest] = []
    with Image.new("RGB", (4, 5), (51, 102, 204)) as image:
        # The workflow painted over the whole canvas, source included.
        content = BytesIO()
        image.save(content, format="PNG")

    async def generate(media: MediaRequest) -> AsyncIterator[MediaEvent]:
        sent.append(media)
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
            json={**request, "source_fit": {"mode": "extend", "width": 4, "height": 5}},
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run"]["id"]
        with SessionLocal() as session:
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            claim = JobClaim(token="source-fit-outpaint-route", attempt=1)
            job.status = "running"
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id = job.id
            session.commit()
        await orchestrator._execute_media(job_id, run_id, claim)

    assert len(sent) == 1
    dispatched: dict[str, Any] = sent[0].workflow
    # The accepted margins centre the 2x3 source on 4x5, in the one padding step.
    sides = tuple(dispatched["pad"]["inputs"][side] for side in ("left", "top", "right", "bottom"))
    assert sides == (1, 1, 1, 1)
    assert dispatched["pad"]["inputs"]["feathering"] == 24
    assert dispatched["sample"]["inputs"]["denoise"] == 1
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        output = run.provenance_json["outputs"][0]
    assert output["source_restore"]["left"] == 1
    assert output["source_restore"]["top"] == 1
    assert output["source_fit_agreement"] == {"v": 1, "state": "preserved"}
