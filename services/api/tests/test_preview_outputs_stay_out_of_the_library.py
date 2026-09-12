"""A preview node's throwaway is shown, and is not filed as something you made.

Many workflows carry a preview node beside the node that saves the real picture.
The engine marks that file as its own throwaway and does not keep it; the tool
was keeping it, filing it in the Media Library and counting it as one of your
outputs, with nothing able to tell it from the picture you asked for.

It is now shown, not filed, and not counted. These cover the two halves that
changed and pin the half that deliberately did not: the throwaway still reaches
the conversation, because somebody may have added that branch precisely to
watch it.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.models import Artifact, ArtifactLibraryEntry, Job, Run
from local_lm.output_origin import names_a_preview, record_for, stated_origin
from local_lm.scheduler import JobClaim
from local_lm.schemas import TurnRequest


def _png(width: int, height: int) -> bytes:
    pixels = bytes(
        (row * 7 + column * 13) % 256 for row in range(height) for column in range(width * 3)
    )
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), pixels).save(buffer, "PNG")
    return buffer.getvalue()


async def _run_with(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, assets: list[GeneratedAsset]
) -> tuple[str, list[str]]:
    """One real media execution; returns the run id and the library's contents."""
    orchestrator = app.state.services.orchestrator
    chat = (await client.post("/api/chats", json={"title": "Preview outputs"})).json()

    async def generate(request: MediaRequest) -> Any:
        yield MediaEvent(type="complete", assets=assets)

    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)

    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(text="A picture of something neutral.", mode="image"),
                freeze_context=True,
                activate_branch=False,
            )
            run_id = accepted.run.id
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            job.status = "running"
            claim = JobClaim(token="preview-output-attempt", attempt=1)
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id = job.id
            session.commit()

        await orchestrator._execute_media(job_id, run_id, claim)

    with SessionLocal() as session:
        filed = [entry.artifact_id for entry in session.scalars(select(ArtifactLibraryEntry))]
    return run_id, filed


def _saved(content: bytes, name: str) -> GeneratedAsset:
    return GeneratedAsset(
        content=content,
        kind="image",
        media_type="image/png",
        name=name,
        origin=stated_origin("save", "output", "images"),
    )


def _throwaway(content: bytes, name: str) -> GeneratedAsset:
    return GeneratedAsset(
        content=content,
        kind="image",
        media_type="image/png",
        name=name,
        origin=stated_origin("preview", "temp", "images"),
    )


async def test_the_saved_picture_is_filed_and_the_throwaway_is_not(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    kept, thrown = _png(96, 64), _png(32, 128)

    run_id, filed = await _run_with(
        app, client, monkeypatch, [_saved(kept, "kept.png"), _throwaway(thrown, "throwaway.png")]
    )

    with SessionLocal() as session:
        rows = {
            artifact.sha256: artifact.id
            for artifact in session.scalars(select(Artifact))
            if artifact.metadata_json.get("run_id") == run_id
        }
    assert rows[hashlib.sha256(kept).hexdigest()] in filed
    assert rows[hashlib.sha256(thrown).hexdigest()] not in filed


async def test_the_throwaway_still_reaches_the_conversation(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The half that deliberately did NOT change.

    Not filing it is a decision about a curated collection. Hiding it would be a
    different decision, and somebody may have added that branch to watch it.
    """
    kept, thrown = _png(96, 64), _png(32, 128)

    run_id, _ = await _run_with(
        app, client, monkeypatch, [_saved(kept, "kept.png"), _throwaway(thrown, "throwaway.png")]
    )

    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        shown = {entry["sha256"] for entry in run.provenance_json["outputs"]}
    assert hashlib.sha256(kept).hexdigest() in shown
    assert hashlib.sha256(thrown).hexdigest() in shown, "the picture is still produced and recorded"


async def test_a_throwaway_is_not_counted_as_something_you_made(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the count actually decides, observed where the decision is made.

    `output_count` gates whether a run is recorded as evidence that the engine
    can produce output at all (orchestrator.py:5423). So a run whose only files
    were throwaways must not count as that proof. The evidence record itself is
    only written when the engine also reports healthy, which this fixture's mock
    does not, so the value is observed at the call rather than in the record -
    this shows what production computes, not what a later reader would see.
    """
    orchestrator = app.state.services.orchestrator
    counts: list[int] = []
    original = orchestrator._record_successful_media_evidence

    def watched(*args: Any, output_count: int, **kwargs: Any) -> Any:
        counts.append(output_count)
        return original(*args, output_count=output_count, **kwargs)

    monkeypatch.setattr(orchestrator, "_record_successful_media_evidence", watched)

    await _run_with(
        app,
        client,
        monkeypatch,
        [_saved(_png(96, 64), "kept.png"), _throwaway(_png(32, 128), "throwaway.png")],
    )

    assert counts == [1], "two files were produced and exactly one was yours"


async def test_a_run_that_produced_only_throwaways_counts_as_nothing(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case the subtraction exists for.

    A preview branch alone is not the engine demonstrating it can make you a
    picture, and a zero count is what stops it being recorded as such.
    """
    orchestrator = app.state.services.orchestrator
    counts: list[int] = []
    original = orchestrator._record_successful_media_evidence

    def watched(*args: Any, output_count: int, **kwargs: Any) -> Any:
        counts.append(output_count)
        return original(*args, output_count=output_count, **kwargs)

    monkeypatch.setattr(orchestrator, "_record_successful_media_evidence", watched)

    await _run_with(app, client, monkeypatch, [_throwaway(_png(48, 48), "only-a-preview.png")])

    assert counts == [0]


async def test_a_file_we_cannot_attribute_is_never_treated_as_a_throwaway(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absence of evidence must not discard somebody's picture.

    An engine that names nothing leaves us unable to say which node wrote a file.
    Reading that as "probably a preview" would quietly keep a real output out of
    the library, which is the failure this whole area exists to avoid.
    """
    content = _png(64, 64)
    unattributed = GeneratedAsset(
        content=content, kind="image", media_type="image/png", name="unnamed.png"
    )

    run_id, filed = await _run_with(app, client, monkeypatch, [unattributed])

    assert names_a_preview(record_for(None, "mock")) is False
    with SessionLocal() as session:
        rows = [
            artifact.id
            for artifact in session.scalars(select(Artifact))
            if artifact.metadata_json.get("run_id") == run_id
        ]
    assert rows and rows[0] in filed
