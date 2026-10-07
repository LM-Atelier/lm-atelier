"""The size judgement is made where the picture is stored, and from the right facts.

`output_size_agreement.py` decides whether what arrived is the size that was
asked for, and its own tests cover that decision. What these cover is the SEAM:
that the orchestrator actually asks, that it asks with the run's accepted
settings rather than the artifact's own, and that the binding it passes is a
statement about the graph that RAN rather than the one that was stored.

The two halves are separated deliberately. `_geometry_binding_confirmed` is
where a rewritten graph could quietly inherit a promise made about a different
document, so it is driven directly with a real accepted context. The recording
is driven through a real media execution, because a helper that is never called
is the failure this file exists to prevent.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select

from local_lm.accepted_turn_context import AcceptedWorkflow
from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.domain import PartType
from local_lm.model_planner import workflow_artifact_contract
from local_lm.models import Artifact, Job, MessagePart, Run
from local_lm.output_origin import stated_origin
from local_lm.scheduler import JobClaim
from local_lm.schemas import TurnRequest


def _provable_graph() -> dict[str, Any]:
    """The smallest graph the proof accepts: save <- decode <- sampler <- latent."""
    return {
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": "${width}", "height": "${height}", "batch_size": 1},
        },
        "sampler": {"class_type": "KSampler", "inputs": {"latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0]}},
    }


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            # multipleOf is not decoration: without a declared step the proof
            # refuses the revision as unsupported, because it cannot say which
            # sizes are actually offerable.
            "width": {
                "type": "integer",
                "default": 1024,
                "minimum": 128,
                "maximum": 2048,
                "multipleOf": 64,
            },
            "height": {
                "type": "integer",
                "default": 768,
                "minimum": 128,
                "maximum": 2048,
                "multipleOf": 64,
            },
        },
    }


def _accepted_workflow(*, trusted: bool = True) -> AcceptedWorkflow:
    graph, schema = _provable_graph(), _schema()
    return AcceptedWorkflow(
        id="revision-1",
        workflow_id="workflow-1",
        version=1,
        engine="comfyui",
        engine_version=None,
        api_graph_json=graph,
        input_schema_json=schema,
        capabilities_json=[],
        dependencies_json={},
        dependency_contract_sha256=None,
        artifact_sha256=workflow_artifact_contract(
            operation="text_to_image",
            engine="comfyui",
            api_graph=graph,
            input_schema=schema,
            dependencies={},
        ),
        trusted=trusted,
    )


def _context(workflow: AcceptedWorkflow | None) -> Any:
    """A minimal accepted context carrying only what the binding question reads."""

    class _Frozen:
        operation = "text_to_image"
        settings: dict[str, Any] = {"width": 1024, "height": 768}

        def __init__(self, value: AcceptedWorkflow | None) -> None:
            self.workflow = value

    return _Frozen(workflow)


def test_a_run_that_kept_its_proven_graph_is_confirmed(app: FastAPI) -> None:
    orchestrator = app.state.services.orchestrator
    assert (
        orchestrator._geometry_binding_confirmed(_context(_accepted_workflow()), _provable_graph())
        is True
    )


def test_a_rewritten_graph_does_not_inherit_the_promise(app: FastAPI) -> None:
    """The whole reason the executed graph is consulted at all.

    A scaling step inserted after the decode leaves a graph that still reads
    correctly and still produces a picture - of a different size. The proof was
    granted over a document that did not contain it.
    """
    orchestrator = app.state.services.orchestrator
    rewritten = _provable_graph()
    rewritten["scale"] = {
        "class_type": "ImageScaleBy",
        "inputs": {"image": ["decode", 0], "scale_by": 2.0},
    }
    rewritten["save"]["inputs"]["images"] = ["scale", 0]

    assert (
        orchestrator._geometry_binding_confirmed(_context(_accepted_workflow()), rewritten) is False
    )


@pytest.mark.parametrize(
    ("workflow", "executed", "why"),
    [
        (None, _provable_graph(), "no workflow was accepted for this run"),
        (_accepted_workflow(trusted=False), _provable_graph(), "the revision is not trusted"),
        (_accepted_workflow(), {}, "nothing was dispatched"),
    ],
)
def test_anything_short_of_both_answers_is_unconfirmed(
    app: FastAPI, workflow: AcceptedWorkflow | None, executed: dict[str, Any], why: str
) -> None:
    """Unconfirmed is the safe answer, and it must not be reachable by accident."""
    orchestrator = app.state.services.orchestrator
    assert orchestrator._geometry_binding_confirmed(_context(workflow), executed) is False, why


def _png(width: int, height: int) -> bytes:
    pixels = bytes(
        (row * 7 + column * 13) % 256 for row in range(height) for column in range(width * 3)
    )
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), pixels).save(buffer, "PNG")
    return buffer.getvalue()


async def _run_once(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    *,
    title: str,
    content: bytes,
    settings: dict[str, Any] | None = None,
    token: str,
) -> tuple[str, str]:
    """One real media execution; returns its run id and the artifact it stored."""
    orchestrator = app.state.services.orchestrator
    chat = (await client.post("/api/chats", json={"title": title})).json()

    async def generate(request: MediaRequest) -> Any:
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=content,
                    kind="image",
                    media_type="image/png",
                    name="made.png",
                    origin=stated_origin("save", "output", "images"),
                )
            ],
        )

    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)

    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(
                    text="A picture of something neutral.",
                    mode="image",
                    settings=settings or {},
                ),
                freeze_context=True,
                activate_branch=False,
            )
            run_id = accepted.run.id
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            job.status = "running"
            claim = JobClaim(token=token, attempt=1)
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id = job.id
            session.commit()

        await orchestrator._execute_media(job_id, run_id, claim)

    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        part = session.scalar(
            select(MessagePart)
            .where(MessagePart.message_id == run.assistant_message_id)
            .where(MessagePart.type == PartType.IMAGE.value)
        )
        # Found through the run's OWN message rather than by asking which run
        # stored the artifact. A deduped row keeps the first run's stamp, so
        # asking the artifact would give the wrong answer for the second run -
        # which is the very behaviour these tests are about.
        assert part is not None and part.artifact_id, "the run put a picture in its message"
        return run_id, part.artifact_id


def _judgement_for(run_id: str) -> dict[str, Any]:
    """What THIS run recorded, read the way a reader of that run would."""
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        outputs = run.provenance_json["outputs"]
        assert len(outputs) == 1
        recorded = outputs[0]["output_size_agreement"]
    assert isinstance(recorded, dict)
    return recorded


def _shown_beside(run_id: str) -> dict[str, Any]:
    """What THIS run's own message shows beside its picture.

    Reached through the run's assistant message rather than by picking the first
    image part in the database. With two runs storing the same bytes, a lookup
    that ignored the run would read whichever part came first and agree with
    itself no matter which record the production code wrote.
    """
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        part = session.scalar(
            select(MessagePart)
            .where(MessagePart.message_id == run.assistant_message_id)
            .where(MessagePart.type == PartType.IMAGE.value)
        )
        assert part is not None, "the run's own message carries its picture"
        shown = part.metadata_json.get("output_size_agreement")
    assert isinstance(shown, dict)
    return shown


@pytest.mark.parametrize(
    ("first_settings", "second_settings", "first_state", "direction"),
    [
        (
            {"width": 64, "height": 64},
            {"width": 1024, "height": 768},
            "agreed",
            "a later run must not invent a warning for an earlier one",
        ),
        (
            {"width": 1024, "height": 768},
            {"width": 64, "height": 64},
            "disagreed",
            "a later run must not erase an earlier one's warning",
        ),
    ],
)
async def test_one_run_cannot_rewrite_what_another_was_told(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    first_settings: dict[str, Any],
    second_settings: dict[str, Any],
    first_state: str,
    direction: str,
) -> None:
    """The defect this repair exists for, driven through two real executions.

    An artifact is addressed by its contents, so two runs that produce identical
    bytes are ONE stored row. Whether a picture is the size somebody asked for
    is a fact about the ASKING, not about the bytes - so putting it on that
    shared row let whichever run finished last replace what an earlier
    conversation had already been shown.

    MEASUREMENT BOUNDARY, stated rather than hidden. This fixture has no trusted
    revision, so a real run here always answers `binding_unconfirmed` and both
    runs would record the same thing - which would make this test pass without
    testing anything. The binding input is therefore pinned to True so the
    comparison actually happens and the two runs differ. The real helper is
    exercised unpinned by the four tests above, including the case where an
    inserted scaler must refuse it; this pins only the boolean, to isolate
    which record the answer lands in.
    """
    orchestrator = app.state.services.orchestrator
    monkeypatch.setattr(orchestrator, "_geometry_binding_confirmed", lambda *args, **kwargs: True)
    identical = _png(64, 64)

    first, first_artifact = await _run_once(
        app,
        client,
        monkeypatch,
        title="First request",
        content=identical,
        settings=first_settings,
        token="size-first",
    )
    first_before = _judgement_for(first)
    assert first_before["state"] == first_state, "the first run got a real judgement"

    second, second_artifact = await _run_once(
        app,
        client,
        monkeypatch,
        title="Second request",
        content=identical,
        settings=second_settings,
        token="size-second",
    )

    # The premise. Without identical bytes landing on ONE row this test would
    # pass for the wrong reason and prove nothing about sharing at all.
    assert first_artifact == second_artifact, "identical bytes are one stored artifact"
    assert _judgement_for(second)["state"] != first_state, "the runs really do differ"

    assert _judgement_for(first) == first_before, direction
    assert _shown_beside(first) == first_before, direction

    with SessionLocal() as session:
        shared = session.get(Artifact, first_artifact)
        assert shared is not None
    assert "output_size_agreement" not in shared.metadata_json, (
        "a run-specific judgement must not live on a content-addressed row"
    )


async def test_the_seam_records_a_judgement_for_every_stored_picture(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Driven through a real media execution, because an uncalled helper is the risk.

    This fixture's run has no trusted revision, so the honest answer for it is
    `binding_unconfirmed` rather than a comparison - and that IS the assertion.
    Recording a non-assessment is what stops a later reader mistaking "we did not
    look" for "we looked and it was fine", and deleting the production call makes
    the key absent rather than merely wrong.
    """
    orchestrator = app.state.services.orchestrator
    chat = (await client.post("/api/chats", json={"title": "Output size"})).json()
    content = _png(64, 48)

    async def generate(request: MediaRequest) -> Any:
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=content,
                    kind="image",
                    media_type="image/png",
                    name="made.png",
                    origin=stated_origin("save", "output", "images"),
                )
            ],
        )

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
            claim = JobClaim(token="size-agreement-attempt", attempt=1)
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id = job.id
            session.commit()

        await orchestrator._execute_media(job_id, run_id, claim)

    recorded = _judgement_for(run_id)
    assert recorded["state"] == "not_assessed", "the seam asked the question"
    assert recorded["reason"] == "binding_unconfirmed"
    # And the same answer is beside the picture in the conversation.
    assert _shown_beside(run_id) == recorded
