"""Compare source-fit canvas size only with its attributed completed output."""

from __future__ import annotations

from collections.abc import AsyncIterator
from io import BytesIO
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
from local_lm.models import Job, MessagePart, Run
from local_lm.output_origin import stated_origin
from local_lm.scheduler import JobClaim


@pytest.mark.parametrize("choice", ["match", "wrong_size", "other_output", "preview", "unknown"])
async def test_completed_source_canvas_is_judged_only_for_its_bound_output(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    orchestrator = app.state.services.orchestrator
    dimensions = (6, 5) if choice == "wrong_size" else (4, 5)
    with Image.new("RGB", dimensions, "blue") as image:
        content = BytesIO()
        image.save(content, format="PNG")
    origin = (
        None
        if choice == "unknown"
        else stated_origin(
            "another-save" if choice == "other_output" else "save",
            "temp" if choice == "preview" else "output",
            "images",
        )
    )

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
                    origin=origin,
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
        run_id = response.json()["run"]["id"]
        with SessionLocal() as session:
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            claim = JobClaim(token="source-fit-output-size", attempt=1)
            job.status = "running"
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id = job.id
            session.commit()
        await orchestrator._execute_media(job_id, run_id, claim)

    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        recorded = run.provenance_json["outputs"][0]["output_size_agreement"]
        part = session.scalar(
            select(MessagePart).where(
                MessagePart.message_id == run.assistant_message_id,
                MessagePart.type == "image",
            )
        )
        assert part is not None
        assert part.metadata_json["output_size_agreement"] == recorded
        if choice in {"match", "wrong_size"}:
            assert recorded["state"] == ("agreed" if choice == "match" else "disagreed")
            assert (recorded["requested_width"], recorded["requested_height"]) == (4, 5)
            assert (recorded["raster_width"], recorded["raster_height"]) == dimensions
        else:
            assert recorded == {
                "v": 1,
                "state": "not_assessed",
                "reason": {
                    "other_output": "binding_unconfirmed",
                    "preview": "throwaway",
                    "unknown": "origin_unknown",
                }[choice],
            }
