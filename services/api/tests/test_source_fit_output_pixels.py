"""Source preservation is judged at the real completed-output boundary."""

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
from local_lm.models import Artifact, Job, MessagePart, Run
from local_lm.output_origin import stated_origin
from local_lm.scheduler import JobClaim


def output_png(choice: str = "preserved") -> bytes:
    if choice == "malformed":
        return output_png()[:-8]
    dimensions = (5, 5) if choice == "wrong_size" else (4, 5)
    with Image.new("RGB", dimensions, (51, 102, 204)) as image:
        # Explicit expected clockwise source orientation, independent of any
        # bytes returned by the accepted context or the adapter request.
        pixels = [
            (255, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (0, 255, 0),
            (255, 0, 255),
            (0, 0, 255),
        ]
        offset = 0 if choice == "shifted" else 1
        for index, pixel in enumerate(pixels):
            image.putpixel((offset + index % 2, offset + index // 2), pixel)
        if choice == "changed":
            image.putpixel((1, 1), (0, 0, 0))
        content = BytesIO()
        image.save(content, format="PNG")
    return content.getvalue()


async def complete(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    chat_id: str,
    request: dict[str, Any],
    content: bytes,
    *,
    choice: str = "preserved",
    with_fit: bool = True,
    count: int = 1,
) -> str:
    origin = (
        None
        if choice == "unknown"
        else stated_origin(
            "another-save" if choice == "other_output" else "save",
            "temp" if choice == "preview" else "output",
            "images",
        )
    )
    orchestrator = app.state.services.orchestrator

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        if with_fit:
            assert request.input_contents is not None
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=content,
                    kind="image",
                    media_type="image/png",
                    name="canvas.png",
                    origin=origin,
                )
                for _ in range(count)
            ],
        )

    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)
    monkeypatch.setattr(orchestrator, "_ensure_media_worker", AsyncMock())
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                **request,
                **({"source_fit": {"mode": "extend", "width": 4, "height": 5}} if with_fit else {}),
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run"]["id"]
        assert isinstance(run_id, str)
        with SessionLocal() as session:
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            claim = JobClaim(token="source-fit-output-pixels", attempt=1)
            job.status = "running"
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id = job.id
            session.commit()
        await orchestrator._execute_media(job_id, run_id, claim)
    return run_id


def records(run_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        outputs = run.provenance_json["outputs"]
        parts = list(
            session.scalars(
                select(MessagePart)
                .where(
                    MessagePart.message_id == run.assistant_message_id,
                    MessagePart.type == "image",
                )
                .order_by(MessagePart.position)
            )
        )
        return outputs, [part.metadata_json for part in parts]


@pytest.mark.parametrize(
    "choice",
    [
        "preserved",
        "changed",
        "shifted",
        "wrong_size",
        "preview",
        "other_output",
        "unknown",
        "malformed",
    ],
)
async def test_completed_output_compares_the_exact_accepted_source_rectangle(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    run_id = await complete(
        app, client, monkeypatch, chat_id, request, output_png(choice), choice=choice
    )
    outputs, parts = records(run_id)
    assert len(outputs) == len(parts) == 1
    value = outputs[0]["source_fit_agreement"]
    assert parts[0]["source_fit_agreement"] == value
    if choice in {"preserved", "changed", "shifted"}:
        # A source the workflow altered is put back before anything judges it.
        assert value == {"v": 1, "state": "preserved"}
        assert ("source_restore" in outputs[0]) == (choice != "preserved")
    else:
        assert "source_restore" not in outputs[0]
        assert value == {
            "v": 1,
            "state": "not_assessed",
            "reason": {
                "wrong_size": "canvas_mismatch",
                "preview": "throwaway",
                "other_output": "binding_unconfirmed",
                "unknown": "origin_unknown",
                "malformed": "output_unreadable",
            }[choice],
        }
    with SessionLocal() as session:
        artifact = session.get(Artifact, outputs[0]["artifact_id"])
        assert artifact is not None
        assert "source_fit_agreement" not in artifact.metadata_json


async def test_shared_output_keeps_each_runs_source_judgement(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    content = output_png()
    first = await complete(app, client, monkeypatch, chat_id, request, content)
    second = await complete(app, client, monkeypatch, chat_id, request, content, with_fit=False)
    first_outputs, first_parts = records(first)
    second_outputs, second_parts = records(second)
    # A source already in place keeps the workflow's own bytes, so both runs
    # share one artifact while only the fitted run carries a judgement.
    assert first_outputs[0]["artifact_id"] == second_outputs[0]["artifact_id"]
    assert (
        first_outputs[0]["source_fit_agreement"]
        == first_parts[0]["source_fit_agreement"]
        == {"v": 1, "state": "preserved"}
    )
    assert "source_fit_agreement" not in second_outputs[0]
    assert "source_fit_agreement" not in second_parts[0]


async def test_ordinary_output_does_not_acquire_a_source_fit_judgement(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    run_id = await complete(
        app, client, monkeypatch, chat_id, request, output_png(), with_fit=False
    )
    outputs, parts = records(run_id)
    assert "source_fit_agreement" not in outputs[0]
    assert "source_fit_agreement" not in parts[0]


@pytest.mark.parametrize(
    "pixels,states",
    [
        (25, ["not_assessed", "not_assessed"]),
        (26, ["preserved", "not_assessed"]),
        (46, ["preserved", "preserved"]),
    ],
)
async def test_source_and_every_output_share_one_pixel_budget(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    pixels: int,
    states: list[str],
) -> None:
    from local_lm import source_fit_output

    chat_id, request = await prepared_turn(app, client, monkeypatch)
    # Source has six pixels; each output has twenty. The second output must
    # spend another twenty even when its content-addressed bytes are identical.
    monkeypatch.setattr(source_fit_output, "MAX_VERIFICATION_PIXELS", pixels)
    run_id = await complete(app, client, monkeypatch, chat_id, request, output_png(), count=2)
    outputs, parts = records(run_id)
    assert len(outputs) == len(parts) == 2
    for index, state in enumerate(states):
        value = {"v": 1, "state": state}
        if state == "not_assessed":
            value["reason"] = "over_budget"
        assert (
            outputs[index]["source_fit_agreement"] == parts[index]["source_fit_agreement"] == value
        )


async def test_pixel_verification_refuses_a_shared_byte_budget_exhaustion(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import source_fit_output

    chat_id, request = await prepared_turn(app, client, monkeypatch)
    monkeypatch.setattr(source_fit_output, "MAX_VERIFICATION_BYTES", 1)
    run_id = await complete(app, client, monkeypatch, chat_id, request, output_png())
    outputs, parts = records(run_id)
    assert (
        outputs[0]["source_fit_agreement"]
        == parts[0]["source_fit_agreement"]
        == {"v": 1, "state": "not_assessed", "reason": "over_budget"}
    )


@pytest.mark.parametrize("opaque", [True, False])
async def test_preservation_includes_source_opacity(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    opaque: bool,
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    content = BytesIO()
    with Image.open(BytesIO(output_png())) as source, source.convert("RGBA") as rgba:
        if not opaque:
            rgba.putpixel((1, 1), (255, 255, 0, 0))
        rgba.save(content, format="PNG")
    run_id = await complete(app, client, monkeypatch, chat_id, request, content.getvalue())
    outputs, parts = records(run_id)
    # A source pixel made transparent is put back opaque, as the source was.
    assert (
        outputs[0]["source_fit_agreement"]
        == parts[0]["source_fit_agreement"]
        == {"v": 1, "state": "preserved"}
    )
    assert ("source_restore" in outputs[0]) == (not opaque)


@pytest.mark.parametrize("format", ["JPEG", "PNG"])
async def test_unsupported_output_encoding_is_not_called_preserved(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    format: str,
) -> None:
    from PIL import PngImagePlugin

    chat_id, request = await prepared_turn(app, client, monkeypatch)
    content = BytesIO()
    with Image.open(BytesIO(output_png())) as source:
        if format == "PNG":
            info = PngImagePlugin.PngInfo()
            info.add(b"gAMA", (45455).to_bytes(4, "big"))
            source.save(content, format="PNG", pnginfo=info)
        else:
            source.save(content, format=format)
    run_id = await complete(app, client, monkeypatch, chat_id, request, content.getvalue())
    outputs, parts = records(run_id)
    assert (
        outputs[0]["source_fit_agreement"]
        == parts[0]["source_fit_agreement"]
        == {"v": 1, "state": "not_assessed", "reason": "output_unsupported"}
    )
