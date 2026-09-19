"""Dragging the canvas edge pads the picture by that much in the graph that runs."""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select

from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.models import Job, Run, WorkflowDefinition, WorkflowRevision
from local_lm.outpaint_workflows import OUTPAINT_SCHEMA_KIND, OUTPAINT_SETTING_KEY
from local_lm.scheduler import JobClaim
from local_lm.schemas import TurnRequest

#: EXIF orientation 6 stores a picture rotated; every viewer and ComfyUI's
#: LoadImage show it turned a quarter, with width and height swapped.
ROTATED = 6


def _png(width: int, height: int, *, orientation: int | None = None) -> bytes:
    image = Image.new("RGB", (width, height), (90, 120, 150))
    buffer = io.BytesIO()
    if orientation is None:
        image.save(buffer, "PNG")
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation
        image.save(buffer, "PNG", exif=exif.tobytes())
    return buffer.getvalue()


def _outpaint_graph(*, pads: int = 1) -> dict[str, Any]:
    graph: dict[str, Any] = {
        "load": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "pad": {
            "class_type": "ImagePadForOutpaint",
            "inputs": {
                "image": ["load", 0],
                "left": 0,
                "top": 0,
                "right": 0,
                "bottom": 0,
                "feathering": 40,
            },
        },
        "encode": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {
                "pixels": ["pad", 0],
                "mask": ["pad", 1],
                "vae": ["vae", 0],
                "grow_mask_by": 8,
            },
        },
        "sample": {
            "class_type": "KSampler",
            "inputs": {"latent_image": ["encode", 0], "denoise": 1},
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]},
        },
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0]}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "neutral.safetensors"}},
    }
    if pads == 2:
        graph["pad_again"] = {
            "class_type": "ImagePadForOutpaint",
            "inputs": {
                "image": ["load", 0],
                "left": 8,
                "top": 8,
                "right": 8,
                "bottom": 8,
                "feathering": 0,
            },
        }
    return graph


async def _outpainter(
    client: AsyncClient, graph: dict[str, Any], source: bytes
) -> tuple[str, str, str]:
    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Neutral outpainter", operation="image_to_image")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            trusted=True,
            api_graph_json=graph,
            input_schema_json={
                "type": "object",
                "properties": {
                    OUTPAINT_SETTING_KEY: {
                        "type": "object",
                        "x-lm-atelier-kind": OUTPAINT_SCHEMA_KIND,
                        "default": {"top": 0, "right": 0, "bottom": 0, "left": 0},
                    }
                },
            },
            dependencies_json={},
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        revision_id = revision.id
        session.commit()
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("neutral.png", source, "image/png")}
    )
    assert uploaded.status_code == 201, uploaded.text
    chat = (await client.post("/api/chats", json={"title": "Extend"})).json()
    return revision_id, uploaded.json()["id"], chat["id"]


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        # Left and right are fractions of the width, top and bottom of the height.
        (None, {"left": 0, "top": 75, "right": 200, "bottom": 0}),
        # A rotated source is extended as it is seen, 300 wide and 400 high.
        (ROTATED, {"left": 0, "top": 100, "right": 150, "bottom": 0}),
    ],
)
async def test_the_dragged_margins_pad_the_picture_in_the_dispatched_graph(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    orientation: int | None,
    expected: dict[str, int],
) -> None:
    orchestrator = app.state.services.orchestrator
    graph = _outpaint_graph()
    revision_id, source_id, chat_id = await _outpainter(
        client, graph, _png(400, 300, orientation=orientation)
    )
    seen: list[MediaRequest] = []

    async def generate(request: MediaRequest) -> Any:
        seen.append(request)
        yield MediaEvent(type="cancelled")

    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)
    monkeypatch.setattr(orchestrator, "_ensure_media_worker", AsyncMock())
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat_id,
                TurnRequest(
                    text="Extend the scene to the right and upward.",
                    mode="image",
                    input_artifact_ids=[source_id],
                    workflow_revision_id=revision_id,
                    settings={OUTPAINT_SETTING_KEY: {"right": 0.5, "top": 0.25}},
                ),
                freeze_context=True,
                activate_branch=False,
            )
            run = session.get(Run, accepted.run.id)
            assert run is not None
            job = session.scalar(select(Job).where(Job.run_id == run.id))
            assert job is not None
            claim = JobClaim(token="outpaint-attempt", attempt=1)
            job.status = "running"
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id, run_id = job.id, run.id
            session.commit()
        await orchestrator._execute_media(job_id, run_id, claim)

    assert len(seen) == 1
    dispatched = seen[0]
    assert dispatched.workflow["pad"]["inputs"] == {
        "image": ["load", 0],
        **expected,
        "feathering": 40,
    }
    # The margins are spent on the graph; no ${outpaint_margins} is left to fill.
    assert OUTPAINT_SETTING_KEY not in dispatched.parameters
    # The stored revision is untouched: the rewrite belongs to this run.
    with SessionLocal() as session:
        stored = session.get(WorkflowRevision, revision_id)
        assert stored is not None and stored.api_graph_json == graph


async def test_a_workflow_with_no_single_source_pad_refuses_the_margins(
    client: AsyncClient,
) -> None:
    revision_id, source_id, chat_id = await _outpainter(
        client, _outpaint_graph(pads=2), _png(400, 300)
    )
    response = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={
            "text": "Extend the scene.",
            "mode": "image",
            "input_artifact_ids": [source_id],
            "workflow_revision_id": revision_id,
            "settings": {OUTPAINT_SETTING_KEY: {"right": 0.5}},
        },
    )
    assert response.status_code == 422, response.text
    assert "cannot extend a picture past its edge" in response.text
