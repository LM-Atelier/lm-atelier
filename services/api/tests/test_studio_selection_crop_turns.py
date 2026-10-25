"""A small selection is edited on a crop of its picture only where the edit can be placed back."""

from __future__ import annotations

import io
import random
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image, ImageDraw
from sqlalchemy import select
from test_studio_region_edit_turns import _masked_edit_workflow
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_geometry import masked_edit_graph, object_info

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Job, ModelProfile, Run
from local_lm.output_origin import stated_origin
from local_lm.output_recipe_replay import mark_replay
from local_lm.scheduler import JobClaim

SIZE = (512, 384)
#: A 40 x 30 selection, small enough to be edited on a crop.
SELECTED = (200, 150, 239, 179)
PAINTED = (10, 200, 30)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _grain() -> Image.Image:
    """Fine grain, which enlarging the crop and shrinking it back softens."""
    rng = random.Random(11)
    picture = Image.new("RGB", SIZE)
    picture.putdata(
        [
            (
                max(0, min(255, round(rng.gauss(90, 14)))),
                max(0, min(255, round(rng.gauss(140, 14)))),
                max(0, min(255, round(rng.gauss(70, 14)))),
            )
            for _ in range(SIZE[0] * SIZE[1])
        ]
    )
    return picture


def _selection() -> bytes:
    alpha = Image.new("L", SIZE, 0)
    ImageDraw.Draw(alpha).rectangle(SELECTED, fill=255)
    drawn = Image.new("RGBA", SIZE, (255, 255, 255, 0))
    drawn.putalpha(alpha)
    return _png(drawn)


async def _upload(client: AsyncClient, name: str, content: bytes) -> str:
    response = await client.post("/api/artifacts", files={"file": (name, content, "image/png")})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _reviewed_workflow(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, shape: str
) -> dict[str, Any]:
    """A reviewed ComfyUI edit with its own mask input, and what a turn names to run it."""
    graph = masked_edit_graph(shape)

    async def describe() -> dict[str, Any]:
        return object_info(graph)

    services = app.state.services
    monkeypatch.setattr(services.engines.settings, "media_engine", "comfyui")
    monkeypatch.setattr(services.engines.media, "object_info", describe, raising=False)
    monkeypatch.setattr(services.orchestrator, "_ensure_media_worker", AsyncMock())
    with SessionLocal() as session:
        profile = ModelProfile(name="Masked edit fixture", role="image", engine="comfyui")
        session.add(profile)
        session.flush()
        profile_id = profile.id
        session.commit()
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Masked edit fixture",
            "operation": "image_to_image",
            "engine": "comfyui",
            "api_graph": graph,
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "input_image": {"type": "string"},
                    "mask": {"type": "object", "x-lm-atelier-kind": "mask"},
                    "checkpoint": {"type": "string", "default": "fixture.safetensors"},
                },
            },
        },
    )
    assert created.status_code == 201, created.text
    definition = created.json()
    revision_id = definition["current_revision_id"]
    url = f"/api/workflows/{definition['id']}/revisions/{revision_id}/review"
    preview = await client.get(url)
    assert preview.status_code == 200, preview.text
    reviewed = await client.post(
        url, json={"action": "approve", "subject_sha256": preview.json()["subject_sha256"]}
    )
    assert reviewed.status_code == 200, reviewed.text
    return {"workflow_revision_id": revision_id, "profile_id": profile_id}


def _faithful(request: MediaRequest, scale: int) -> bytes:
    """What a faithful inpaint returns: the picture it was given, painted where its mask selects."""
    if request.input_contents is not None:
        given = Image.open(io.BytesIO(request.input_contents[0])).convert("RGB")
    else:
        given = Image.open(request.input_paths[0]).convert("RGB")
    assert request.mask_content is not None
    selected = Image.open(io.BytesIO(request.mask_content)).getchannel("A")
    assert selected.size == given.size
    painted = Image.composite(
        Image.new("RGB", given.size, PAINTED),
        given,
        selected.point(lambda value: 255 if value >= 128 else 0),
    )
    return _png(painted.resize((given.width * scale, given.height * scale)))


async def _edited(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    picture: Image.Image,
    run_with: dict[str, Any],
    *,
    scale: int = 1,
    replayed: bool = False,
) -> tuple[MediaRequest, Run, Image.Image]:
    """A brushed turn run through to its stored picture, and the request the engine was given."""
    orchestrator = app.state.services.orchestrator
    seen: list[MediaRequest] = []

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        seen.append(request)
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=_faithful(request, scale),
                    kind="image",
                    media_type="image/png",
                    name="edited.png",
                    origin=stated_origin("save", "output", "images"),
                )
            ],
        )

    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)
    source_id = await _upload(client, "source.png", _png(picture))
    mask_id = await _upload(client, "selection.png", _selection())
    chat = (await client.post("/api/chats", json={"title": "Paint"})).json()
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Paint the marked part green.",
                "mode": "image",
                "input_artifact_ids": [source_id],
                "settings": {"mask": {"artifact_id": mask_id, "feather_px": 0, "invert": False}},
                **run_with,
            },
        )
        assert accepted.status_code == 202, accepted.text
        run_id = accepted.json()["run"]["id"]
        with SessionLocal() as session:
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            claim = JobClaim(token="selection-crop-turn", attempt=1)
            job.status = "running"
            job.claim_owner, job.attempt = claim.token, claim.attempt
            job_id = job.id
            if replayed:
                run = session.get(Run, run_id)
                assert run is not None
                # As generating a record again marks its run before it is committed.
                mark_replay(
                    run,
                    {"digest": f"sha256:{'d' * 64}", "output": {"index": 0, "sha256": "e" * 64}},
                )
            session.commit()
        await orchestrator._execute_media(job_id, run_id, claim)
    assert len(seen) == 1
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.status == "complete", run.error
        kept = run.provenance_json["outputs"][0]
        stored = session.get(Artifact, kept["artifact_id"])
        assert stored is not None
        out = Image.open(orchestrator.artifacts.resolve(stored)).convert("RGB")
        out.load()
        session.expunge(run)
    return seen[0], run, out


@pytest.mark.parametrize("shape", ["inpaint", "conditioning"])
async def test_a_small_selection_is_edited_on_a_crop_and_placed_back(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """The workflow draws the selection with all its pixels, and only the selection changes.

    On fine grain, which the crop's resizing softens, the edit keeps the colour
    it was drawn in.
    """
    picture = _grain()
    run_with = await _reviewed_workflow(app, client, monkeypatch, shape)

    request, run, out = await _edited(app, client, monkeypatch, picture, run_with)

    assert request.input_contents is not None
    assert request.mask_content is not None
    given = Image.open(io.BytesIO(request.input_contents[0]))
    handed = Image.open(io.BytesIO(request.mask_content))
    assert given.size == handed.size == (1088, 960)
    placed = run.provenance_json["outputs"][0]["selection_crop"]
    assert placed["box"] == [168, 118, 272, 212]
    assert placed["colour_shift"] == [0, 0, 0]
    assert out.size == SIZE
    for y in range(0, SIZE[1], 3):
        for x in range(0, SIZE[0], 3):
            if not (SELECTED[0] <= x <= SELECTED[2] and SELECTED[1] <= y <= SELECTED[3]):
                assert out.getpixel((x, y)) == picture.getpixel((x, y)), (x, y)
    assert out.getpixel((219, 164)) == PAINTED


@pytest.mark.parametrize(
    "shape,scale",
    [("enlarges", 2), ("sets_its_own_size", 1), ("saves_its_mask", 1)],
    ids=["enlarges", "sets-its-own-size", "saves-another-picture"],
)
async def test_a_workflow_whose_pictures_may_change_size_is_handed_the_whole_picture(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
    scale: int,
) -> None:
    """Its result could not be placed back from a crop, so it is kept as the workflow made it."""
    picture = _grain()
    run_with = await _reviewed_workflow(app, client, monkeypatch, shape)

    request, run, out = await _edited(app, client, monkeypatch, picture, run_with, scale=scale)

    assert request.input_contents is None
    assert request.mask_content is not None
    assert Image.open(io.BytesIO(request.mask_content)).size == SIZE
    assert "selection_crop" not in run.provenance_json["outputs"][0]
    assert out.size == (SIZE[0] * scale, SIZE[1] * scale)


async def test_generating_a_record_again_edits_the_whole_picture(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact replay repeats a record, and a record of a cropped edit is never exact."""
    picture = _grain()
    run_with = await _reviewed_workflow(app, client, monkeypatch, "inpaint")

    request, run, out = await _edited(app, client, monkeypatch, picture, run_with, replayed=True)

    assert request.input_contents is None
    assert "selection_crop" not in run.provenance_json["outputs"][0]
    assert out.size == SIZE


async def test_an_engine_that_reads_files_is_handed_the_whole_picture(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only ComfyUI is handed the crop's own bytes; another engine reads the stored picture."""
    picture = _grain()

    request, run, out = await _edited(
        app,
        client,
        monkeypatch,
        picture,
        {"workflow_revision_id": _masked_edit_workflow()},
    )

    assert request.input_contents is None
    assert "selection_crop" not in run.provenance_json["outputs"][0]
    assert out.size == SIZE
