"""A running media turn says which step its engine is on, beside its progress and previews."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.models import Job, Run
from local_lm.orchestrator import _reported_step
from local_lm.scheduler import JobClaim


def _png(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "data,reported",
    [
        ({"value": 3.0, "max": 12.0}, {"step": 3, "steps": 12}),
        ({"value": 0, "max": 20}, {"step": 0, "steps": 20}),
        ({"value": 12, "max": 12}, {"step": 12, "steps": 12}),
        ({"value": 13, "max": 12}, {}),
        ({"value": 2.5, "max": 12}, {}),
        ({"value": float("inf"), "max": 12}, {}),
        ({"value": 3, "max": float("nan")}, {}),
        ({"value": -1, "max": 12}, {}),
        ({"value": 0, "max": 0}, {}),
        ({"value": 3, "max": 100_001}, {}),
        ({"value": 10**400, "max": 10**400}, {}),
        ({"value": 3, "max": 1e400}, {}),
        ({"value": True, "max": 12}, {}),
        ({"value": "3", "max": "12"}, {}),
        ({"max": 12}, {}),
        ({}, {}),
    ],
)
def test_only_a_whole_step_inside_its_count_is_reported(
    data: dict[str, Any], reported: dict[str, int]
) -> None:
    assert _reported_step(data) == reported


async def _published_during_a_turn(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    generate: Any = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Every event the app published while one image turn ran through its media path."""
    services = app.state.services
    if generate is not None:
        monkeypatch.setattr(services.orchestrator.engines.media, "generate", generate)
    published: list[tuple[str, dict[str, Any]]] = []
    real_publish = services.orchestrator.events.publish

    async def recording_publish(name: str, entity: object, payload: dict[str, Any]) -> None:
        published.append((name, payload))
        await real_publish(name, entity, payload)

    monkeypatch.setattr(services.orchestrator.events, "publish", recording_publish)
    chat = (await client.post("/api/chats", json={"title": "Steps"})).json()
    async with services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": "A plain grey square.", "mode": "image"},
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run"]["id"]
        with SessionLocal() as session:
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            claim = JobClaim(token="step-events", attempt=1)
            job.status = "running"
            job.claim_owner, job.attempt = claim.token, claim.attempt
            job_id = job.id
            session.commit()
        await services.orchestrator._execute_media(job_id, run_id, claim)
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.status == "complete", run.error
    return published


def _payloads(published: list[tuple[str, dict[str, Any]]], name: str) -> list[dict[str, Any]]:
    return [payload for event, payload in published if event == name]


async def test_progress_and_previews_carry_the_step_the_engine_reported(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def generate(media: MediaRequest) -> AsyncIterator[MediaEvent]:
        yield MediaEvent(type="queued", data={"prompt_id": "p"}, engine_sourced=True)
        yield MediaEvent(
            type="progress",
            progress=0.25,
            phase="sampling",
            data={"prompt_id": "p", "value": 3.0, "max": 12.0},
            engine_sourced=True,
        )
        yield MediaEvent(
            type="preview",
            progress=0.25,
            phase="sampling",
            preview=_png((90, 90, 90)),
            engine_sourced=True,
        )
        yield MediaEvent(
            type="progress",
            progress=0.33,
            phase="sampling",
            data={"prompt_id": "p", "value": 4.0, "max": 12.0},
            engine_sourced=True,
        )
        yield MediaEvent(
            type="preview",
            progress=0.33,
            phase="sampling",
            preview=_png((100, 100, 100)),
            engine_sourced=True,
        )
        # A stage that reports no step of its own.
        yield MediaEvent(
            type="progress",
            progress=0.95,
            phase="decoding",
            data={"indeterminate": True},
            engine_sourced=True,
        )
        yield MediaEvent(
            type="complete",
            progress=1,
            assets=[
                GeneratedAsset(
                    content=_png((120, 120, 120)),
                    kind="image",
                    media_type="image/png",
                    name="square.png",
                )
            ],
            engine_sourced=True,
        )

    published = await _published_during_a_turn(app, client, monkeypatch, generate)

    progress = _payloads(published, "generation.progress")
    sampling = [payload for payload in progress if payload.get("phase") == "sampling"]
    assert [(p["step"], p["steps"]) for p in sampling] == [(3, 12), (4, 12)]
    decoding = [payload for payload in progress if payload.get("phase") == "decoding"]
    assert decoding and all("step" not in p and "steps" not in p for p in decoding)
    previews = _payloads(published, "generation.preview")
    assert [(p["step"], p["steps"]) for p in previews] == [(3, 12), (4, 12)]
    assert all(p["artifact_id"] for p in previews)


async def test_a_preview_before_any_reported_step_says_none(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def generate(media: MediaRequest) -> AsyncIterator[MediaEvent]:
        yield MediaEvent(
            type="preview",
            progress=0.1,
            phase="sampling",
            preview=_png((80, 80, 80)),
            engine_sourced=True,
        )
        yield MediaEvent(
            type="complete",
            progress=1,
            assets=[
                GeneratedAsset(
                    content=_png((120, 120, 120)),
                    kind="image",
                    media_type="image/png",
                    name="square.png",
                )
            ],
            engine_sourced=True,
        )

    published = await _published_during_a_turn(app, client, monkeypatch, generate)

    previews = _payloads(published, "generation.preview")
    assert len(previews) == 1
    assert "step" not in previews[0] and "steps" not in previews[0]


async def test_the_mock_engine_reports_its_sampling_step(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clients can be built against the mock engine and see the same fields."""
    published = await _published_during_a_turn(app, client, monkeypatch)

    sampling = [p for p in _payloads(published, "generation.progress") if p["phase"] == "sampling"]
    assert [(p["step"], p["steps"]) for p in sampling] == [(7, 10)]
    previews = _payloads(published, "generation.preview")
    assert [(p["step"], p["steps"]) for p in previews] == [(7, 10)]
