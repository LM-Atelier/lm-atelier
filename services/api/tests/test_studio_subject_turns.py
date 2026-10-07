"""Pictures an edit only reads, and what the report needs before a subject can be replaced."""

from __future__ import annotations

import io
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image, ImageDraw
from sqlalchemy import select

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Job, Run, WorkflowDefinition, WorkflowRevision
from local_lm.scheduler import JobClaim
from local_lm.schemas import TurnRequest

BOX = (16, 12, 40, 30)
REDRAWN = (10, 200, 30)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _source() -> Image.Image:
    picture = Image.new("RGB", (64, 48))
    picture.putdata(
        [((x * 7) % 256, (y * 11) % 256, (x * y) % 256) for y in range(48) for x in range(64)]
    )
    return picture


def _mask() -> bytes:
    alpha = Image.new("L", (64, 48), 0)
    ImageDraw.Draw(alpha).rectangle(BOX, fill=255)
    mask = Image.new("RGBA", (64, 48), (255, 255, 255, 0))
    mask.putalpha(alpha)
    return _png(mask)


def _workflow(name: str, graph: dict[str, Any], properties: dict[str, Any]) -> str:
    with SessionLocal() as session:
        definition = WorkflowDefinition(name=name, operation="image_to_image")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            trusted=True,
            api_graph_json=graph,
            input_schema_json={"type": "object", "properties": properties},
            dependencies_json={},
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        return revision.id


def _two_picture_editor() -> str:
    """An edit workflow that reads a second picture beside the one it edits."""
    return _workflow(
        "Two-picture editor",
        {
            "1": {"class_type": "LoadImage", "inputs": {"image": "${input_image_0}"}},
            "2": {"class_type": "LoadImage", "inputs": {"image": "${input_image_1}"}},
        },
        {},
    )


async def _upload(client: AsyncClient, name: str, content: bytes) -> str:
    response = await client.post("/api/artifacts", files={"file": (name, content, "image/png")})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def test_a_blend_selection_may_name_the_pictures_it_only_reads(
    client: AsyncClient,
) -> None:
    revision_id = _two_picture_editor()
    source_id = await _upload(client, "source.png", _png(_source()))
    reference_id = await _upload(
        client, "reference.png", _png(Image.new("RGB", (40, 40), (1, 2, 3)))
    )
    mask_id = await _upload(client, "selection.png", _mask())
    chat = (await client.post("/api/chats", json={"title": "Replace the subject"})).json()

    async def post(inputs: list[str], mask: dict[str, Any]) -> Any:
        return await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Replace the subject with the one in the second picture",
                "mode": "image",
                "input_artifact_ids": inputs,
                "workflow_revision_id": revision_id,
                "settings": {"mask": mask},
            },
        )

    blend = {"artifact_id": mask_id, "feather_px": 4, "invert": False, "apply": "blend"}
    with_reference = {**blend, "references": 1}

    # Two pictures and no word on which is edited: still ambiguous.
    unnamed = await post([source_id, reference_id], blend)
    assert unnamed.status_code != 202
    assert "only into the one picture being edited" in unnamed.text
    # A reference with nothing left to edit.
    alone = await post([reference_id], with_reference)
    assert alone.status_code != 202
    assert "only into the one picture being edited" in alone.text
    # Only a selection blended back can leave pictures out of it.
    workflow_mask = {key: value for key, value in with_reference.items() if key != "apply"}
    refused = await post([source_id, reference_id], workflow_mask)
    assert refused.status_code != 202
    accepted = await post([source_id, reference_id], with_reference)
    assert accepted.status_code == 202, accepted.text
    with SessionLocal() as session:
        run = session.get(Run, accepted.json()["run"]["id"])
        assert run is not None
        assert run.settings_json["mask"] == with_reference


async def test_the_new_subject_is_placed_back_into_the_first_picture(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = app.state.services.orchestrator
    revision_id = _two_picture_editor()
    source = _source()
    source_id = await _upload(client, "source.png", _png(source))
    reference = Image.new("RGB", (64, 48), (240, 240, 240))
    reference_id = await _upload(client, "reference.png", _png(reference))
    mask_id = await _upload(client, "selection.png", _mask())
    chat = (await client.post("/api/chats", json={"title": "Replace the subject"})).json()
    seen: list[MediaRequest] = []

    async def generate(request: MediaRequest) -> Any:
        seen.append(request)
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=_png(Image.new("RGB", (96, 72), REDRAWN)),
                    kind="image",
                    media_type="image/png",
                    name="redrawn.png",
                    origin={"node_id": "9", "output_type": "output", "collection": "images"},
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
                    text="Replace the subject with the one in the second picture",
                    mode="image",
                    input_artifact_ids=[source_id, reference_id],
                    workflow_revision_id=revision_id,
                    settings={
                        "mask": {
                            "artifact_id": mask_id,
                            "feather_px": 0,
                            "invert": False,
                            "apply": "blend",
                            "references": 1,
                        }
                    },
                ),
                freeze_context=True,
                activate_branch=False,
            )
            job = session.scalar(select(Job).where(Job.run_id == accepted.run.id))
            assert job is not None
            claim = JobClaim(token="subject-attempt", attempt=1)
            job.status = "running"
            job.claim_owner, job.attempt = claim.token, claim.attempt
            job_id, run_id = job.id, accepted.run.id
            session.commit()
        await orchestrator._execute_media(job_id, run_id, claim)

    assert len(seen) == 1
    # The workflow reads both pictures and never the selection.
    assert len(seen[0].input_paths) == 2
    assert "mask" not in seen[0].parameters
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.status == "complete", run.error
        [entry] = run.provenance_json["outputs"]
        picture = session.get(Artifact, entry["artifact_id"])
        assert picture is not None
        region_edit = entry["region_edit"]
        assert region_edit["source_artifact_id"] == source_id
        assert region_edit["selection"]["references"] == 1
        picture_bytes = orchestrator.artifacts.resolve(picture).read_bytes()
    out = Image.open(io.BytesIO(picture_bytes))
    # The picture being edited, never the reference: its size, and its own
    # pixels everywhere the selection does not reach.
    assert out.size == source.size
    for y in range(source.height):
        for x in range(source.width):
            inside = BOX[0] <= x <= BOX[2] and BOX[1] <= y <= BOX[3]
            assert out.getpixel((x, y)) == (REDRAWN if inside else source.getpixel((x, y)))


async def test_the_report_offers_a_subject_once_a_cutout_and_an_editor_are_installed(
    client: AsyncClient,
) -> None:
    async def subject() -> dict[str, Any]:
        tools = (await client.get("/api/studio/capabilities")).json()["tools"]
        return next(tool for tool in tools if tool["kind"] == "subject")

    nothing = await subject()
    assert nothing["available"] is False
    assert "background removal" in nothing["reason"]

    # The editor that removes the old subject is installed here already, and
    # it reads one picture, which is enough: the new subject is placed, not
    # redrawn. A cutout is all that is missing.
    cutout_id = _workflow(
        "Cutout",
        {"1": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}}},
        {"matte": {"type": "boolean", "x-lm-atelier-kind": "matting"}},
    )
    both = await subject()
    assert both["available"] is True
    assert both["reason"] is None
    # Both cutouts run on the workflow Isolate names; the old subject is
    # removed on the studio's own, which the report never names.
    assert both["workflow_revision_id"] == cutout_id
