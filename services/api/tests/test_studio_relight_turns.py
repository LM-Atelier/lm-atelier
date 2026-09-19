"""A relight turn is a two-picture edit with the adapter, finished against its source."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image, ImageDraw
from sqlalchemy import select

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.auxiliary_assets import checkpoint_lora_extension
from local_lm.db import SessionLocal
from local_lm.domain import Operation, PartType, utcnow
from local_lm.models import (
    Artifact,
    Job,
    Message,
    MessagePart,
    ModelAssetInstall,
    ModelProfile,
    ModelSource,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
)
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.scheduler import JobClaim
from local_lm.schemas import TurnRequest
from local_lm.studio_capabilities import tool_capabilities
from local_lm.studio_masks import MaskSelection
from local_lm.studio_region_edit import RegionEdit
from local_lm.studio_relight import (
    LIGHTING_ADAPTER,
    RelightFinish,
    RelightSetting,
    finish_relight,
    warmth_grade,
)

BOX = (16, 12, 40, 30)
RELIT = (40, 60, 80)


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


def _light_map() -> Image.Image:
    return Image.linear_gradient("L").rotate(90).resize((64, 48)).convert("RGB")


def _relight_workflow() -> str:
    """An edit workflow that takes two pictures and lets a LoRA be added."""
    graph: dict[str, Any] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "neutral.safetensors"},
        },
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "${prompt}"}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "${input_image_0}"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": "${input_image_1}"}},
    }
    extension = checkpoint_lora_extension(graph)
    assert extension is not None
    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Two-picture editor", operation="image_to_image")
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
                "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
            },
            dependencies_json={"extensions": {"lora": extension}},
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        return revision.id


def _lora(
    *, sha256: str = LIGHTING_ADAPTER.sha256, revision: str = LIGHTING_ADAPTER.revision
) -> str:
    with SessionLocal() as session:
        source = ModelSource(
            provider=LIGHTING_ADAPTER.provider,
            remote_id=LIGHTING_ADAPTER.remote_id,
            revision=revision,
        )
        session.add(source)
        session.flush()
        asset = ModelAssetInstall(
            name="Neutral lighting adapter",
            kind="lora",
            family=None,
            local_path="C:/managed/neutral-lighting",
            size_bytes=1024,
            source_id=source.id,
            manifest_json={
                "sha256": sha256,
                "comfy_name": LIGHTING_ADAPTER.filename,
                "expected_sha256": {LIGHTING_ADAPTER.filename: sha256},
                "metadata": {"trigger_words": []},
            },
            active=True,
            verified_at=utcnow(),
        )
        session.add(asset)
        session.commit()
        return asset.id


async def _upload(client: AsyncClient, name: str, content: bytes) -> str:
    response = await client.post("/api/artifacts", files={"file": (name, content, "image/png")})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


RELIGHT = {"direction": "left", "intensity": 0.5, "kelvin": None}


async def test_a_relight_turn_needs_its_map_and_the_adapter_at_full_strength(
    client: AsyncClient,
) -> None:
    revision_id = _relight_workflow()
    adapter_id = _lora()
    lookalike_id = _lora(sha256="d" * 64, revision="main")
    source_id = await _upload(client, "source.png", _png(_source()))
    map_id = await _upload(client, "light.png", _png(_light_map()))
    chat = (await client.post("/api/chats", json={"title": "Relight"})).json()

    async def post(inputs: list[str], loras: list[dict[str, Any]]) -> Any:
        return await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Relight from the left",
                "mode": "image",
                "input_artifact_ids": inputs,
                "workflow_revision_id": revision_id,
                "settings": {"relight": RELIGHT, "loras": loras},
            },
        )

    full = [{"asset_id": adapter_id, "model_strength": 1}]
    only_source = await post([source_id], full)
    assert only_source.status_code != 202
    assert "the picture and its light map" in only_source.text
    no_adapter = await post([source_id, map_id], [])
    assert no_adapter.status_code != 202
    assert "lighting LoRA installed and applied" in no_adapter.text
    lookalike = await post([source_id, map_id], [{"asset_id": lookalike_id, "model_strength": 1}])
    assert lookalike.status_code != 202
    assert "lighting LoRA installed and applied" in lookalike.text
    gentle = await post([source_id, map_id], [{"asset_id": adapter_id, "model_strength": 0.6}])
    assert gentle.status_code != 202
    assert "only at full strength" in gentle.text

    accepted = await post([source_id, map_id], full)

    assert accepted.status_code == 202, accepted.text
    with SessionLocal() as session:
        run = session.get(Run, accepted.json()["run"]["id"])
        assert run is not None
        assert run.settings_json["relight"] == RELIGHT


async def test_the_report_offers_relight_once_both_halves_are_installed(
    client: AsyncClient,
) -> None:
    async def relight() -> dict[str, Any]:
        tools = (await client.get("/api/studio/capabilities")).json()["tools"]
        return next(tool for tool in tools if tool["kind"] == "relight")

    nothing = await relight()
    assert nothing["available"] is False
    assert "takes a second picture and a LoRA" in nothing["reason"]

    revision_id = _relight_workflow()
    # Same repository, another revision and another file: not the adapter.
    _lora(sha256="d" * 64, revision="main")
    workflow_only = await relight()
    assert workflow_only["available"] is False
    assert "Multi-Angle Lighting LoRA" in workflow_only["reason"]

    adapter_id = _lora()
    both = await relight()

    assert both["available"] is True
    assert both["reason"] is None
    assert both["workflow_revision_id"] == revision_id
    assert both["adapter_asset_id"] == adapter_id


def test_several_relight_workflows_leave_the_choice_to_the_studio() -> None:
    tools = {
        tool.kind: tool
        for tool in tool_capabilities(
            edit_input_schemas=[None, None],
            relight_workflow_ids=["wfrev_a", "wfrev_b"],
            lighting_adapter_ids=["asset_light"],
        )
    }

    assert tools["relight"].available is True
    assert tools["relight"].workflow_revision_id is None
    assert tools["relight"].adapter_asset_id == "asset_light"
    assert tools["instruct"].workflow_revision_id is None
    assert tools["instruct"].adapter_asset_id is None


async def test_the_stored_picture_is_the_relit_result_mixed_toward_the_source(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = app.state.services.orchestrator
    revision_id = _relight_workflow()
    adapter_id = _lora()
    source = _source()
    source_bytes = _png(source)
    source_id = await _upload(client, "source.png", source_bytes)
    map_id = await _upload(client, "light.png", _png(_light_map()))
    relit = _png(Image.new("RGB", (96, 72), RELIT))
    preview = _png(Image.new("RGB", (32, 24), (90, 90, 90)))
    seen: list[MediaRequest] = []

    async def generate(request: MediaRequest) -> Any:
        seen.append(request)
        yield MediaEvent(
            type="complete",
            assets=[
                GeneratedAsset(
                    content=relit,
                    kind="image",
                    media_type="image/png",
                    name="relit.png",
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
    chat = (await client.post("/api/chats", json={"title": "Relight"})).json()
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(
                    text="Relight from the left",
                    mode="image",
                    input_artifact_ids=[source_id, map_id],
                    workflow_revision_id=revision_id,
                    settings={
                        "relight": RELIGHT,
                        "loras": [{"asset_id": adapter_id, "model_strength": 1}],
                    },
                ),
                freeze_context=True,
                activate_branch=False,
            )
            job = session.scalar(select(Job).where(Job.run_id == accepted.run.id))
            assert job is not None
            claim = JobClaim(token="relight-attempt", attempt=1)
            job.status = "running"
            job.claim_owner, job.attempt = claim.token, claim.attempt
            job_id, run_id = job.id, accepted.run.id
            session.commit()
        await orchestrator._execute_media(job_id, run_id, claim)

    assert len(seen) == 1
    assert "relight" not in seen[0].parameters
    assert len(seen[0].input_paths) == 2
    expected = finish_relight(
        RelightFinish(
            setting=RelightSetting("left", 0.5, None),
            source_artifact_id=source_id,
            source=source_bytes,
        ),
        relit,
    )
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.status == "complete", run.error
        outputs = run.provenance_json["outputs"]
        stored = {
            entry["artifact_id"]: session.get(Artifact, entry["artifact_id"]) for entry in outputs
        }
        kept = next(entry for entry in outputs if "relight" in entry)
        picture = stored[kept["artifact_id"]]
        assert picture is not None
        assert orchestrator.artifacts.resolve(picture).read_bytes() == expected.content
        assert kept["relight"] == expected.record
        assert picture.metadata_json["relight"] == expected.record
        throwaway = next(
            artifact
            for artifact_id, artifact in stored.items()
            if artifact is not None and artifact_id != picture.id
        )
        assert orchestrator.artifacts.resolve(throwaway).read_bytes() == preview
        assert not any(
            "relight" in entry for entry in outputs if entry["artifact_id"] == throwaway.id
        )


async def test_relight_then_the_selection_blend_in_one_pipeline() -> None:
    """The order is the contract: a selection keeps even the relight outside it.

    With warmth the two orders differ: grading after the blend would warm the
    pixels outside the selection too.
    """
    source = _source()
    source_bytes = _png(source)
    alpha = Image.new("L", (64, 48), 0)
    ImageDraw.Draw(alpha).rectangle(BOX, fill=255)
    mask = Image.new("RGBA", (64, 48), (255, 255, 255, 0))
    mask.putalpha(alpha)
    setting = RelightSetting("left", 0.5, 4500)
    relight = RelightFinish(
        setting=setting, source_artifact_id="sha256:" + "b" * 64, source=source_bytes
    )
    region = RegionEdit(
        selection=MaskSelection(
            artifact_id="sha256:" + "c" * 64, feather_px=0, invert=False, blend=True
        ),
        source_artifact_id="sha256:" + "b" * 64,
        source=source_bytes,
        mask=_png(mask),
    )
    produced = GeneratedAsset(
        content=_png(Image.new("RGB", (64, 48), RELIT)),
        kind="image",
        media_type="image/png",
        name="r.png",
    )
    video = GeneratedAsset(content=b"neutral-video", kind="video", media_type="video/mp4", name="v")

    finished, passed = await ConversationOrchestrator._studio_finished_outputs(
        [produced, video], relight=relight, region_edit=region, media_engine="comfyui"
    )

    assert passed == video
    out = Image.open(io.BytesIO(finished.content))
    mixed = warmth_grade(Image.blend(source, Image.new("RGB", (64, 48), RELIT), 0.5), 4500)
    for y in range(48):
        for x in range(64):
            inside = BOX[0] <= x <= BOX[2] and BOX[1] <= y <= BOX[3]
            assert out.getpixel((x, y)) == (mixed if inside else source).getpixel((x, y)), (x, y)
    assert finished.metadata["relight"]["intensity"] == 0.5
    assert finished.metadata["region_edit"]["mode"] == "blend"


async def test_an_automatic_retry_keeps_the_relight_and_the_selection_apart() -> None:
    relight = {"direction": "top", "intensity": 0.75, "kelvin": 4500}
    selection = {
        "artifact_id": "sha256:" + "c" * 64,
        "feather_px": 4,
        "invert": False,
        "apply": "blend",
    }
    from local_lm.image_edit_verification import (
        ImageEditRetryDecision,
        ImageEditVerificationJobPayload,
        VerificationReason,
        image_edit_verification_job_id,
    )

    user = Message(
        id="message-user",
        chat_id="chat-retry",
        parent_id="message-before",
        role="user",
        status="complete",
        parts=[MessagePart(position=0, type=PartType.TEXT.value, text="Relight from the top")],
    )
    assistant = Message(
        id="message-assistant",
        chat_id="chat-retry",
        parent_id=user.id,
        role="assistant",
        status="complete",
    )
    source_run = SimpleNamespace(
        id="run-source",
        chat_id="chat-retry",
        user_message_id=user.id,
        assistant_message_id=assistant.id,
        operation=Operation.IMAGE_TO_IMAGE.value,
        workflow_revision_id="workflow-revision",
        profile_id="profile-image",
        settings_json={"denoise": 0.5, "relight": relight, "mask": selection},
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
        (Message, assistant.id): assistant,
        (WorkflowRevision, "workflow-revision"): SimpleNamespace(
            input_schema_json={"type": "object"}
        ),
        (ModelProfile, "profile-image"): SimpleNamespace(engine="comfyui"),
    }

    class FakeSession:
        def get(self, model: Any, identity: str) -> Any:
            return lookups.get((model, identity))

        def scalar(self, _statement: Any) -> Any:
            return user

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
        return_value={"denoise": 0.5}
    )
    orchestrator.input_artifact_ids_for_run = Mock(  # type: ignore[method-assign]
        return_value=["artifact-original", "artifact-map"]
    )
    orchestrator.create_turn = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            run=SimpleNamespace(id="run-retry", work_plan_id="plan-retry", provenance_json={})
        )
    )

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
    assert request.settings == {"denoise": 0.62, "relight": relight, "mask": selection}
