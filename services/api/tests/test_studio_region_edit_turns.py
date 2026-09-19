"""A turn with a blend selection keeps everything outside the selection as it was."""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image, ImageDraw
from sqlalchemy import select

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import Artifact, Job, Run, WorkflowDefinition, WorkflowRevision
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.scheduler import JobClaim
from local_lm.schemas import TurnRequest
from local_lm.studio_masks import MaskSelection
from local_lm.studio_region_edit import RegionEdit

BOX = (16, 12, 40, 30)
EDITED = (10, 200, 30)


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


def _mask(size: tuple[int, int] = (64, 48)) -> bytes:
    alpha = Image.new("L", size, 0)
    ImageDraw.Draw(alpha).rectangle(BOX, fill=255)
    mask = Image.new("RGBA", size, (255, 255, 255, 0))
    mask.putalpha(alpha)
    return _png(mask)


def _plain_edit_workflow() -> str:
    """An edit workflow that declares no mask input at all."""
    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Plain editor", operation="image_to_image")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            trusted=True,
            api_graph_json={"nodes": [{"inputs": {"image": "${input_image}"}}]},
            input_schema_json={"type": "object", "properties": {}},
            dependencies_json={},
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        return revision.id


async def _upload(client: AsyncClient, name: str, content: bytes) -> str:
    response = await client.post("/api/artifacts", files={"file": (name, content, "image/png")})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def test_a_blend_selection_is_accepted_on_a_workflow_without_a_mask_input(
    client: AsyncClient,
) -> None:
    revision_id = _plain_edit_workflow()
    source_id = await _upload(client, "source.png", _png(_source()))
    other_id = await _upload(client, "other.png", _png(Image.new("RGB", (64, 48), (1, 2, 3))))
    mask_id = await _upload(client, "selection.png", _mask())
    chat = (await client.post("/api/chats", json={"title": "Replace words"})).json()

    async def post(inputs: list[str], mask: dict[str, Any]) -> Any:
        return await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Replace the words",
                "mode": "image",
                "input_artifact_ids": inputs,
                "workflow_revision_id": revision_id,
                "settings": {"mask": mask},
            },
        )

    blend = {"artifact_id": mask_id, "feather_px": 4, "invert": False, "apply": "blend"}
    plain = {key: value for key, value in blend.items() if key != "apply"}

    two_sources = await post([source_id, other_id], blend)
    assert two_sources.status_code != 202
    assert "only into the one picture being edited" in two_sources.text
    without_blend = await post([source_id], plain)
    assert without_blend.status_code != 202
    assert "cannot apply a selection" in without_blend.text
    accepted = await post([source_id], blend)
    assert accepted.status_code == 202, accepted.text
    with SessionLocal() as session:
        run = session.get(Run, accepted.json()["run"]["id"])
        assert run is not None
        assert run.settings_json["mask"] == blend


async def _accepted_blend_run(
    app: FastAPI, client: AsyncClient, mask: bytes
) -> tuple[str, str, JobClaim, Image.Image, str]:
    orchestrator = app.state.services.orchestrator
    revision_id = _plain_edit_workflow()
    source = _source()
    source_id = await _upload(client, "source.png", _png(source))
    mask_id = await _upload(client, "selection.png", mask)
    chat = (await client.post("/api/chats", json={"title": "Replace words"})).json()
    with SessionLocal() as session:
        accepted = await orchestrator.create_turn(
            session,
            chat["id"],
            TurnRequest(
                text="Replace the words",
                mode="image",
                input_artifact_ids=[source_id],
                workflow_revision_id=revision_id,
                settings={
                    "mask": {
                        "artifact_id": mask_id,
                        "feather_px": 0,
                        "invert": False,
                        "apply": "blend",
                    }
                },
            ),
            freeze_context=True,
            activate_branch=False,
        )
        run = session.get(Run, accepted.run.id)
        assert run is not None
        job = session.scalar(select(Job).where(Job.run_id == run.id))
        assert job is not None
        claim = JobClaim(token="region-edit-attempt", attempt=1)
        job.status = "running"
        job.claim_owner, job.attempt = claim.token, claim.attempt
        job_id, run_id = job.id, run.id
        session.commit()
    return job_id, run_id, claim, source, source_id


async def test_the_stored_picture_is_the_source_outside_the_selection(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = app.state.services.orchestrator
    edited = _png(Image.new("RGB", (96, 72), EDITED))
    preview = _png(Image.new("RGB", (32, 24), (90, 90, 90)))
    seen: list[MediaRequest] = []

    async def generate(request: MediaRequest) -> Any:
        seen.append(request)
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=edited,
                    kind="image",
                    media_type="image/png",
                    name="edited.png",
                    origin={"node_id": "9", "output_type": "output", "collection": "images"},
                ),
                GeneratedAsset(
                    content=preview,
                    kind="image",
                    media_type="image/png",
                    name="preview.png",
                    origin={"node_id": "12", "output_type": "temp", "collection": "images"},
                ),
            ],
        )

    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)
    async with app.state.services.scheduler.lease("primary"):
        job_id, run_id, claim, source, source_id = await _accepted_blend_run(app, client, _mask())
        await orchestrator._execute_media(job_id, run_id, claim)

    assert len(seen) == 1
    # The workflow edits the whole picture: it is never handed the selection.
    assert "mask" not in seen[0].parameters
    assert len(seen[0].input_paths) == 1
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.status == "complete", run.error
        outputs = {entry["artifact_id"]: entry for entry in run.provenance_json["outputs"]}
        stored = {artifact_id: session.get(Artifact, artifact_id) for artifact_id in outputs}
        kept = [
            artifact
            for artifact in stored.values()
            if artifact is not None and artifact.size_bytes != len(preview)
        ]
        assert len(kept) == 1
        picture = kept[0]
        picture_bytes = orchestrator.artifacts.resolve(picture).read_bytes()
        assert picture.media_type == "image/png"
        region_edit = outputs[picture.id]["region_edit"]
        assert region_edit["mode"] == "blend"
        assert region_edit["source_artifact_id"] == source_id
        assert region_edit["selection"]["apply"] == "blend"
        assert picture.metadata_json["region_edit"] == region_edit
        throwaway = next(
            artifact
            for artifact in stored.values()
            if artifact is not None and artifact.id != picture.id
        )
        # The workflow's own preview is not the edit the selection was for.
        assert orchestrator.artifacts.resolve(throwaway).read_bytes() == preview
        assert "region_edit" not in outputs[throwaway.id]
        assert "region_edit" not in throwaway.metadata_json
    out = Image.open(io.BytesIO(picture_bytes))
    assert out.size == source.size
    for y in range(source.height):
        for x in range(source.width):
            inside = BOX[0] <= x <= BOX[2] and BOX[1] <= y <= BOX[3]
            assert out.getpixel((x, y)) == (EDITED if inside else source.getpixel((x, y)))


async def test_a_selection_drawn_on_another_shape_refuses_before_the_model_runs(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = app.state.services.orchestrator
    generate = Mock(side_effect=AssertionError("the model ran for a selection that cannot fit"))
    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)
    async with app.state.services.scheduler.lease("primary"):
        job_id, run_id, claim, _source_picture, _source_id = await _accepted_blend_run(
            app, client, _mask((64, 64))
        )
        with pytest.raises(RuntimeError, match="does not match this image"):
            await orchestrator._execute_media(job_id, run_id, claim)
    generate.assert_not_called()


async def test_video_output_passes_through_unblended() -> None:
    video = GeneratedAsset(content=b"neutral-video", kind="video", media_type="video/mp4", name="v")
    edit = RegionEdit(
        selection=MaskSelection(
            artifact_id=f"sha256:{'c' * 64}", feather_px=0, invert=False, blend=True
        ),
        source_artifact_id=f"sha256:{'b' * 64}",
        source=_png(_source()),
        mask=_mask(),
    )

    result = await ConversationOrchestrator._studio_finished_outputs(
        [video], relight=None, region_edit=edit, media_engine="comfyui"
    )

    assert result == [video]


async def test_an_automatic_retry_keeps_the_selection() -> None:
    from types import SimpleNamespace

    from local_lm.domain import PartType
    from local_lm.image_edit_verification import (
        ImageEditRetryDecision,
        ImageEditVerificationJobPayload,
        VerificationReason,
        image_edit_verification_job_id,
    )
    from local_lm.models import Message, MessagePart, ModelProfile

    selection = {
        "artifact_id": f"sha256:{'c' * 64}",
        "feather_px": 4,
        "invert": False,
        "apply": "blend",
    }
    source_user = Message(
        id="message-user",
        chat_id="chat-retry",
        parent_id="message-before",
        role="user",
        status="complete",
        parts=[MessagePart(position=0, type=PartType.TEXT.value, text="Replace the words")],
    )
    source_assistant = Message(
        id="message-assistant",
        chat_id="chat-retry",
        parent_id=source_user.id,
        role="assistant",
        status="complete",
    )
    source_run = SimpleNamespace(
        id="run-source",
        chat_id="chat-retry",
        user_message_id=source_user.id,
        assistant_message_id=source_assistant.id,
        operation=Operation.IMAGE_TO_IMAGE.value,
        workflow_revision_id="workflow-revision",
        profile_id="profile-image",
        settings_json={"steps": 8, "denoise": 0.5, "mask": selection},
        provenance_json={
            "image_edit": {
                "strength": {
                    "mode": "auto",
                    "parameter": "denoise",
                    "value": 0.5,
                    "scope": "localized",
                    "confidence": "high",
                }
            }
        },
    )
    lookups: dict[tuple[Any, str], Any] = {
        (Run, source_run.id): source_run,
        (Job, image_edit_verification_job_id(source_run.id)): SimpleNamespace(status="running"),
        (Message, source_assistant.id): source_assistant,
        (WorkflowRevision, source_run.workflow_revision_id): SimpleNamespace(
            input_schema_json={"type": "object"}
        ),
        (ModelProfile, source_run.profile_id): SimpleNamespace(engine="comfyui"),
    }

    class FakeSession:
        def get(self, model: Any, identity: str) -> Any:
            return lookups.get((model, identity))

        def scalar(self, _statement: Any) -> Any:
            return source_user

        def in_transaction(self) -> bool:
            return True

        def commit(self) -> None:
            return None

    orchestrator = ConversationOrchestrator(
        engines=Mock(spec_set=["settings"], settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(spec_set=["publish"], publish=AsyncMock()),
        scheduler=Mock(spec_set=["publish_job"], publish_job=AsyncMock()),
        processes=Mock(spec_set=["statuses"], statuses=Mock(return_value=[])),
        session_factory=Mock(),
    )
    # What the stored-settings filter returns: the workflow's own fields only.
    orchestrator.request_settings_for_operation = AsyncMock(  # type: ignore[method-assign]
        return_value={"steps": 8, "denoise": 0.5}
    )
    orchestrator.input_artifact_ids_for_run = Mock(  # type: ignore[method-assign]
        return_value=["artifact-original"]
    )
    accepted = SimpleNamespace(
        run=SimpleNamespace(id="run-retry", work_plan_id="plan-retry", provenance_json={})
    )
    orchestrator.create_turn = AsyncMock(return_value=accepted)  # type: ignore[method-assign]

    await orchestrator._create_image_edit_verification_retry(
        FakeSession(),  # type: ignore[arg-type]
        ImageEditVerificationJobPayload(
            chat_id=source_run.chat_id,
            source_run_id=source_run.id,
            source_job_id="job-source",
            source_artifact_id="artifact-original",
            result_artifact_id="artifact-first-result",
            vision_profile_id="profile-vision",
            automatic_strength=True,
            strength_parameter="denoise",
            current_strength=0.5,
            minimum=0.3,
            maximum=0.8,
        ),
        ImageEditRetryDecision(
            retry=True,
            reason=VerificationReason.ELIGIBLE,
            attempt=1,
            parameter="denoise",
            value_before=0.5,
            value_after=0.62,
            minimum=0.3,
            maximum=0.8,
        ),
        claim=JobClaim(token="verification-attempt", attempt=1),
    )

    awaited = orchestrator.create_turn.await_args
    assert awaited is not None
    request = awaited.args[2]
    assert request.settings == {"steps": 8, "denoise": 0.62, "mask": selection}
