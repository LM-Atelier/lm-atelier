"""Extend runs on a canvas of whole blocks and keeps the canvas that was asked for."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_geometry import composited_graph, object_info

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Job, ModelProfile, Run
from local_lm.outpaint_workflows import OUTPAINT_SCHEMA_KIND, OUTPAINT_SETTING_KEY
from local_lm.output_origin import stated_origin
from local_lm.scheduler import JobClaim

SIZE = (517, 383)
NEW = (51, 102, 204)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _source() -> Image.Image:
    picture = Image.new("RGB", SIZE)
    picture.putdata(
        [
            ((x * 7) % 256, (y * 11) % 256, (x * y) % 256)
            for y in range(SIZE[1])
            for x in range(SIZE[0])
        ]
    )
    return picture


def _outpainter(shape: str, feathering: object = 0) -> dict[str, Any]:
    graph = composited_graph()
    graph["positive"]["inputs"]["text"] = "${prompt}"
    graph["pad"]["inputs"]["feathering"] = feathering
    if shape == "decoded":
        del graph["composite"]
        graph["save"]["inputs"]["images"] = ["decode", 0]
    elif shape == "enlarges":
        graph["enlarge"] = {
            "class_type": "ImageScaleBy",
            "inputs": {"image": ["composite", 0], "upscale_method": "lanczos", "scale_by": 2.0},
        }
        graph["save"]["inputs"]["images"] = ["enlarge", 0]
    elif shape == "saves_its_mask":
        graph["shown"] = {"class_type": "MaskToImage", "inputs": {"mask": ["pad", 1]}}
        graph["save_mask"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["shown", 0], "filename_prefix": "mask"},
        }
    return graph


async def _extended(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shape: str = "composited",
    source: bytes | None = None,
    keeps_the_asked_size: bool = False,
    feathering: object = 0,
) -> tuple[MediaRequest, dict[str, Any], Image.Image]:
    """An Extend turn on a reviewed ComfyUI outpainter, run through to its stored picture."""
    graph = _outpainter(shape, feathering)
    # A feather mapped to a setting takes that setting's saved value.
    settable = (
        {"feathering": {"type": "integer", "default": 32}}
        if feathering == "$" + "{feathering}"
        else {}
    )

    async def describe() -> dict[str, Any]:
        return object_info(graph)

    services = app.state.services
    monkeypatch.setattr(services.engines.settings, "media_engine", "comfyui")
    monkeypatch.setattr(services.engines.media, "object_info", describe, raising=False)
    monkeypatch.setattr(services.orchestrator, "_ensure_media_worker", AsyncMock())
    with SessionLocal() as session:
        profile = ModelProfile(name="Outpaint fixture", role="image", engine="comfyui")
        session.add(profile)
        session.flush()
        profile_id = profile.id
        session.commit()
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Outpaint fixture",
            "operation": "image_to_image",
            "engine": "comfyui",
            "api_graph": graph,
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "input_image": {"type": "string"},
                    "checkpoint": {"type": "string", "default": "fixture.safetensors"},
                    OUTPAINT_SETTING_KEY: {
                        "type": "object",
                        "x-lm-atelier-kind": OUTPAINT_SCHEMA_KIND,
                        "default": {"top": 0, "right": 0, "bottom": 0, "left": 0},
                    },
                    **settable,
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
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("source.png", source or _png(_source()), "image/png")},
    )
    assert uploaded.status_code == 201, uploaded.text
    chat = (await client.post("/api/chats", json={"title": "Extend"})).json()
    sent: list[MediaRequest] = []

    async def generate(media: MediaRequest) -> AsyncIterator[MediaEvent]:
        sent.append(media)
        pad = media.workflow["pad"]["inputs"]
        handed = (
            Image.open(io.BytesIO(media.input_contents[0])).convert("RGB")
            if media.input_contents
            else _source()
        )
        canvas = Image.new(
            "RGB",
            (
                handed.width + pad["left"] + pad["right"],
                handed.height + pad["top"] + pad["bottom"],
            ),
            NEW,
        )
        canvas.paste(handed, (pad["left"], pad["top"]))
        if keeps_the_asked_size:
            canvas = canvas.resize((SIZE[0] + 129, SIZE[1]))
        elif shape == "enlarges":
            canvas = canvas.resize((canvas.width * 2, canvas.height * 2))
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=_png(canvas),
                    kind="image",
                    media_type="image/png",
                    name="canvas.png",
                    origin=stated_origin("save", "output", "images"),
                )
            ],
        )

    monkeypatch.setattr(services.orchestrator.engines.media, "generate", generate)
    async with services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Continue the pattern past the right edge.",
                "mode": "image",
                "input_artifact_ids": [uploaded.json()["id"]],
                "workflow_revision_id": revision_id,
                "profile_id": profile_id,
                "settings": {OUTPAINT_SETTING_KEY: {"right": 0.25}},
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run"]["id"]
        with SessionLocal() as session:
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            claim = JobClaim(token="outpaint-alignment", attempt=1)
            job.status = "running"
            job.claim_owner, job.attempt = claim.token, claim.attempt
            job_id = job.id
            session.commit()
        await services.orchestrator._execute_media(job_id, run_id, claim)
    assert len(sent) == 1
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.status == "complete", run.error
        output = run.provenance_json["outputs"][0]
        stored = session.get(Artifact, output["artifact_id"])
        assert stored is not None
        picture = Image.open(services.orchestrator.artifacts.resolve(stored)).convert("RGB")
        picture.load()
    return sent[0], output, picture


def _sides(request: MediaRequest) -> dict[str, int]:
    pad = request.workflow["pad"]["inputs"]
    return {side: pad[side] for side in ("top", "right", "bottom", "left")}


async def test_extend_runs_on_whole_blocks_and_keeps_the_canvas_asked_for(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """517 + 129 by 383 is not whole blocks, so the encoder would crop and shift it."""
    request, output, picture = await _extended(app, client, monkeypatch)

    # The extended side grows on its outer edge; the height nobody extended is
    # not padded, so ComfyUI feathers no edge but the right one.
    assert _sides(request) == {"top": 0, "right": 187, "bottom": 0, "left": 0}
    assert request.input_contents is not None
    handed = Image.open(io.BytesIO(request.input_contents[0])).convert("RGB")
    assert handed.size == (517, 384)
    assert handed.crop((0, 0, *SIZE)).tobytes() == _source().tobytes()
    assert output["extension_finish"]["requested"] == {
        "top": 0,
        "right": 129,
        "bottom": 0,
        "left": 0,
    }
    assert picture.size == (646, 383)
    assert picture.crop((0, 0, *SIZE)).tobytes() == _source().tobytes()
    assert picture.getpixel((645, 382)) == NEW


async def test_a_workflow_that_returns_another_size_is_kept_as_made(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, picture = await _extended(app, client, monkeypatch, keeps_the_asked_size=True)

    assert "extension_finish" not in output
    assert picture.size == (646, 383)


@pytest.mark.parametrize("shape", ["enlarges", "saves_its_mask"])
async def test_an_outpainter_that_may_save_another_size_is_padded_as_asked(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """Its pictures could not be cut back, so it runs exactly as asked."""
    request, output, _ = await _extended(app, client, monkeypatch, shape=shape)

    assert _sides(request) == {"top": 0, "right": 129, "bottom": 0, "left": 0}
    assert request.input_contents is None
    assert "extension_finish" not in output


async def test_a_source_comfyui_reads_differently_is_sent_as_stored_and_padded_as_asked(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sixteen_bit = _png(Image.new("I;16", SIZE, 32768))

    request, output, _ = await _extended(app, client, monkeypatch, source=sixteen_bit)

    assert _sides(request) == {"top": 0, "right": 129, "bottom": 0, "left": 0}
    assert request.input_contents is None
    assert "extension_finish" not in output


async def test_a_side_that_would_grow_past_comfyuis_largest_padding_is_padded_as_asked(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ComfyUI refuses a padding past its largest, which the request alone stayed within."""
    import local_lm.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "MAX_PAD_PX", 150)

    request, output, _ = await _extended(app, client, monkeypatch)

    assert _sides(request) == {"top": 0, "right": 129, "bottom": 0, "left": 0}
    assert "extension_finish" not in output


@pytest.mark.parametrize(
    "shape,feathering,colour",
    [
        ("composited", "$" + "{feathering}", None),
        ("composited", 32.0, {"applied": False, "reason": "feather_unknown"}),
        ("decoded", 32, {"applied": False, "reason": "not_blended"}),
    ],
    ids=["feather-from-its-setting", "feather-not-a-whole-number", "decoded-straight-to-its-save"],
)
async def test_only_a_picture_pasted_through_its_padding_mask_has_its_colour_matched(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
    feathering: object,
    colour: dict[str, object] | None,
) -> None:
    """The decode reads back only where the padding's own mask, at a known feather, mixed it in."""
    _, output, picture = await _extended(
        app, client, monkeypatch, shape=shape, feathering=feathering
    )

    recorded = output["extension_finish"]["colour"]
    if colour is None:
        assert recorded["applied"] is True
        assert recorded["feather_px"] == 32
    else:
        assert recorded == colour
    assert picture.crop((0, 0, *SIZE)).tobytes() == _source().tobytes()
