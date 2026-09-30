"""A crop is accepted, kept, uploaded and finished as an ordinary edit of the crop."""

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
from sqlalchemy.orm import Session
from test_accepted_source_fit import save
from test_source_fit_image import artifact_session as artifact_session
from test_source_fit_image import ingest, png
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_crop_route import graph

from local_lm.accepted_turn_context import accepted_context
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.artifacts import ArtifactStore
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus, MessageStatus, utcnow
from local_lm.image_edit_verification import (
    ImageEditRetryDecision,
    ImageEditVerificationJobPayload,
    VerificationReason,
    image_edit_verification_job_id,
)
from local_lm.models import (
    Artifact,
    Chat,
    Job,
    Message,
    ModelInstall,
    ModelProfile,
    Run,
    RunContextArtifact,
    WorkflowDefinition,
    WorkflowRevision,
)
from local_lm.scheduler import JobClaim
from local_lm.source_crop_recipe import SourceCropRecipe, capture_source_crop
from local_lm.source_fit_image import prepare_source_fit_image, replay_source_fit_image
from local_lm.source_fit_restore import fit_to_canvas, fits
from local_lm.workflow_compatibility import mirror_legacy_chat_workflow_selections
from local_lm.workflow_revision_reviews import build_review_snapshot, record_review


def object_info(value: dict[str, Any] | None = None) -> dict[str, Any]:
    """How the fixture runtime describes the crop graph's nodes."""
    arities = {
        "LoadImage": 2,
        "CheckpointLoaderSimple": 3,
        "CLIPTextEncode": 1,
        "VAEEncode": 1,
        "VAELoader": 1,
        "KSampler": 1,
        "VAEDecode": 1,
        "SaveImage": 0,
    }
    return {
        node["class_type"]: {
            "python_module": "nodes",
            "input": {"required": {key: ["*", {}] for key in node["inputs"]}},
            "output": ["*"] * arities[node["class_type"]],
        }
        for node in (graph() if value is None else value).values()
    }


def make_crop_run(pair: tuple[ArtifactStore, Session]) -> tuple[Run, SourceCropRecipe]:
    """A run whose accepted fit crops a two-by-three source to a three-by-two canvas."""
    store, session = pair
    source = ingest(pair, png(orientation=6))
    workflow = graph()
    recipe = capture_source_crop(
        session,
        store,
        prepare_source_fit_image(store, source),
        canvas_width=3,
        canvas_height=2,
        api_graph=workflow,
        save_node_id="save",
    )
    chat = Chat(title="Source crop fixture")
    definition = WorkflowDefinition(name="Source crop fixture", operation="image_to_image")
    session.add_all([chat, definition])
    session.flush()
    messages = [Message(chat_id=chat.id, role="user"), Message(chat_id=chat.id, role="assistant")]
    session.add_all(messages)
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        api_graph_json=workflow,
        input_schema_json={},
        dependencies_json={},
    )
    session.add(revision)
    session.flush()
    run = Run(
        chat_id=chat.id,
        user_message_id=messages[0].id,
        assistant_message_id=messages[1].id,
        workflow_revision_id=revision.id,
        operation="image_to_image",
        standalone_prompt="Crop the neutral color grid",
        settings_json={},
        provenance_json={},
    )
    session.add(run)
    session.flush()
    return run, recipe


def test_a_crop_keeps_its_source_and_the_crop_and_reads_back_as_a_crop(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    run, recipe = make_crop_run(artifact_session)
    save(session, run, recipe)
    session.commit()

    snapshot = accepted_context(session, run)

    assert snapshot is not None
    assert isinstance(snapshot.source_fit, SourceCropRecipe)
    assert snapshot.source_fit == recipe
    kept = set(session.scalars(select(RunContextArtifact.artifact_id)))
    assert kept == set(recipe.retained_artifact_ids)
    assert len(kept) == 3
    crop = replay_source_fit_image(
        session, store, recipe.upload_image, selected_source_id=recipe.image.source_artifact_id
    )
    with Image.open(BytesIO(crop.content)) as picture:
        assert picture.size == (3, 2)
    cropped = session.get(Artifact, recipe.cropped_artifact_id)
    assert cropped is not None and cropped.kind == "input"
    assert "content" not in recipe.model_dump()


def test_a_refused_crop_keeps_nothing(artifact_session: tuple[ArtifactStore, Session]) -> None:
    store, session = artifact_session
    source = ingest(artifact_session, png(orientation=6))
    unrouted = graph()
    unrouted["encode"]["class_type"] = "VAEEncodeForInpaint"
    before = set(session.scalars(select(Artifact.id)))

    with pytest.raises(ValueError, match="source_fit_graph"):
        capture_source_crop(
            session,
            store,
            prepare_source_fit_image(store, source),
            canvas_width=3,
            canvas_height=2,
            api_graph=unrouted,
            save_node_id="save",
        )

    assert set(session.scalars(select(Artifact.id))) == before


async def test_dispatch_uploads_the_crop_and_runs_the_graph_as_it_is(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = app.state.services
    orchestrator = services.orchestrator
    seen: list[MediaRequest] = []

    async def describe() -> dict[str, Any]:
        return object_info()

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        seen.append(request)
        yield MediaEvent(type="cancelled")

    monkeypatch.setattr(services.engines.settings, "media_engine", "comfyui")
    monkeypatch.setattr(services.engines.media, "object_info", describe, raising=False)
    monkeypatch.setattr(services.engines.media, "generate", generate)
    monkeypatch.setattr(orchestrator, "_ensure_media_worker", AsyncMock())

    async with services.scheduler.lease("primary"):
        with SessionLocal() as session:
            run, recipe = make_crop_run((orchestrator.artifacts, session))
            revision = session.get(WorkflowRevision, run.workflow_revision_id)
            assert revision is not None
            revision.input_schema_json = {
                "type": "object",
                "properties": {
                    "input_image": {"type": "string"},
                    "checkpoint": {"type": "string", "default": "fixture.safetensors"},
                    "denoise": {"type": "number", "default": 0.6},
                },
            }
            definition = session.get(WorkflowDefinition, revision.workflow_id)
            assert definition is not None
            record_review(
                session,
                revision,
                build_review_snapshot(session, definition, revision, object_info=object_info()),
                approved=True,
            )
            save(session, run, recipe)
            expected = replay_source_fit_image(
                session,
                orchestrator.artifacts,
                recipe.upload_image,
                selected_source_id=recipe.image.source_artifact_id,
            ).content
            claim = JobClaim(token="source-crop-dispatch", attempt=1)
            job = Job(
                kind="image",
                status="running",
                run_id=run.id,
                claim_owner=claim.token,
                attempt=claim.attempt,
                queue_group="primary",
                queue_resource="media",
            )
            session.add(job)
            session.flush()
            run_id, job_id = run.id, job.id
            session.commit()
        await orchestrator._execute_media(job_id, run_id, claim)

    assert len(seen) == 1
    # The crop goes in the source's place, and the graph is the workflow's own.
    assert seen[0].input_contents == (expected,)
    assert seen[0].workflow == graph()


def encoded(size: tuple[int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, (40, 90, 160)).save(output, format="PNG")
    return output.getvalue()


def test_a_crop_the_vae_trimmed_is_brought_back_to_its_canvas(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    _, recipe = make_crop_run(artifact_session)
    wide = recipe.model_copy(update={"canvas_width": 64, "canvas_height": 40})

    fitted = fit_to_canvas(wide, encoded((56, 40)))

    assert fitted is not None
    with Image.open(BytesIO(fitted.content)) as picture:
        assert picture.size == (64, 40)
    assert (fitted.record["result_width"], fitted.record["width"]) == (56, 64)
    assert fitted.record["cropped_artifact_id"] == recipe.cropped_artifact_id
    # A picture already at its canvas, larger than it, or short by a whole
    # block or more is left as the workflow saved it.
    assert fit_to_canvas(wide, encoded((64, 40))) is None
    assert fit_to_canvas(wide, encoded((70, 40))) is None
    assert fit_to_canvas(recipe.model_copy(update={"canvas_width": 70}), encoded((6, 2))) is None


def test_only_the_recipe_s_own_saved_picture_is_fitted(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    _, recipe = make_crop_run(artifact_session)
    origin = {
        "state": "attributed",
        "node_id": "save",
        "output_type": "output",
        "collection": "images",
    }

    assert fits(recipe, origin)
    assert not fits(recipe, {**origin, "node_id": "other"})
    assert not fits(recipe, {**origin, "output_type": "temp"})
    assert not fits(recipe, {**origin, "state": "unattributed"})
    assert not fits(recipe, None)


async def prepared_crop_turn(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, dict[str, Any]]:
    """A reviewed picture-to-picture workflow, a two-by-three picture and a chat."""
    workflow = graph()

    async def describe() -> dict[str, Any]:
        return object_info(workflow)

    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    monkeypatch.setattr(app.state.services.engines.media, "object_info", describe, raising=False)
    with SessionLocal() as session:
        profile = ModelProfile(name="Source crop fixture", role="image", engine="comfyui")
        session.add(profile)
        session.flush()
        profile_id = profile.id
        session.commit()
    workflow["positive"]["inputs"]["text"] = "${prompt}"
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Source crop fixture",
            "operation": "image_to_image",
            "engine": "comfyui",
            "api_graph": workflow,
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "input_image": {"type": "string"},
                    "checkpoint": {"type": "string", "default": "fixture.safetensors"},
                    "denoise": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.6},
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
        "/api/artifacts", files={"file": ("grid.png", png(orientation=6), "image/png")}
    )
    assert uploaded.status_code == 201
    chat = await client.post("/api/chats", json={"title": "Source crop fixture"})
    assert chat.status_code == 201
    return chat.json()["id"], {
        "text": "Crop the neutral color grid",
        "mode": "image",
        "input_artifact_ids": [uploaded.json()["id"]],
        "workflow_revision_id": revision_id,
        "profile_id": profile_id,
        "settings": {},
    }


async def test_a_workflow_that_edits_a_picture_offers_crop_and_previews_the_kept_part(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request = await prepared_crop_turn(app, client, monkeypatch)
    base = f"/api/workflow-revisions/{request['workflow_revision_id']}/source-fit"

    capability = await client.get(base)
    shown = await client.post(
        base + "/preview",
        json={
            "source_artifact_id": request["input_artifact_ids"][0],
            "source_fit": {"mode": "crop", "width": 3, "height": 2},
        },
    )

    assert capability.status_code == 200, capability.text
    assert capability.json()["modes"] == ["crop"]
    assert shown.status_code == 200, shown.text
    preview = shown.json()
    assert preview["mode"] == "crop"
    assert preview["source"] == {"width": 2, "height": 3}
    assert preview["canvas"] == {"width": 3, "height": 2}
    # The whole width, and the middle four thirds of the height, fill the canvas.
    assert preview["kept"] == pytest.approx({"left": 0, "top": 5 / 6, "width": 2, "height": 4 / 3})
    assert preview["margins"] == {"left": 0, "top": 0, "right": 0, "bottom": 0}


async def test_a_crop_is_accepted_as_an_ordinary_edit_of_the_crop(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, request = await prepared_crop_turn(app, client, monkeypatch)
    async with app.state.services.scheduler.lease("primary"):
        plain = await client.post(f"/api/chats/{chat_id}/turns", json=request)
        cropped = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={**request, "source_fit": {"mode": "crop", "width": 3, "height": 2}},
        )
        assert plain.status_code == 202, plain.text
        assert cropped.status_code == 202, cropped.text
        with SessionLocal() as session:
            ordinary = session.get(Run, plain.json()["run"]["id"])
            run = session.get(Run, cropped.json()["run"]["id"])
            assert ordinary is not None and run is not None
            snapshot = accepted_context(session, run)
            assert snapshot is not None
            recipe = snapshot.source_fit
            assert isinstance(recipe, SourceCropRecipe)
            assert (recipe.image.width, recipe.image.height) == (2, 3)
            assert (recipe.canvas_width, recipe.canvas_height) == (3, 2)
            assert recipe.image.source_artifact_id == request["input_artifact_ids"][0]
            assert run.provenance_json["source_fit_request"] == {
                "mode": "crop",
                "width": 3,
                "height": 2,
            }
            # No full strength is pinned, as it is for new canvas: the edit runs at
            # the strength the same turn without a crop would have.
            assert run.settings_json == ordinary.settings_json
            assert run.settings_json["denoise"] < 1


async def test_an_automatic_retry_edits_the_same_crop_again(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = app.state.services.orchestrator
    chat_id, request = await prepared_crop_turn(app, client, monkeypatch)
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={**request, "source_fit": {"mode": "crop", "width": 3, "height": 2}},
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run"]["id"]
        with SessionLocal() as session:
            source = session.get(Run, run_id)
            assert source is not None
            strength = source.provenance_json["image_edit"]["strength"]
            # A crop is an ordinary edit, so its strength is chosen, and can be retried.
            assert strength["mode"] == "auto"
            # The edit finished, as a verified one has, so its response can be replaced.
            response_message = session.get(Message, source.assistant_message_id)
            assert response_message is not None
            response_message.status = MessageStatus.COMPLETE.value
            verification_id = image_edit_verification_job_id(run_id)
            session.add(
                Job(
                    id=verification_id,
                    kind=JobKind.EDIT_VERIFY.value,
                    status=JobStatus.RUNNING.value,
                    progress=0.0,
                    progress_json={},
                    queue_resource="interactive_compute",
                    queue_group="primary",
                    queue_ticket=verification_id,
                    enqueued_at=utcnow(),
                    payload_json={},
                    cancellable=True,
                )
            )
            session.commit()
            accepted = await orchestrator._create_image_edit_verification_retry(
                session,
                ImageEditVerificationJobPayload(
                    chat_id=chat_id,
                    source_run_id=run_id,
                    source_job_id="job-source",
                    source_artifact_id=request["input_artifact_ids"][0],
                    result_artifact_id="artifact-first-result",
                    vision_profile_id="profile-vision",
                    automatic_strength=True,
                    strength_parameter="denoise",
                    current_strength=strength["value"],
                    minimum=0.3,
                    maximum=0.8,
                ),
                ImageEditRetryDecision(
                    retry=True,
                    reason=VerificationReason.ELIGIBLE,
                    attempt=1,
                    parameter="denoise",
                    value_before=strength["value"],
                    value_after=0.7,
                    minimum=0.3,
                    maximum=0.8,
                ),
            )
        with SessionLocal() as session:
            source = session.get(Run, run_id)
            retry = session.get(Run, accepted.run.id)
            assert source is not None and retry is not None
            kept = accepted_context(session, source)
            again = accepted_context(session, retry)
            assert kept is not None and again is not None
            assert isinstance(again.source_fit, SourceCropRecipe)
            # The very crop the source kept, read back rather than cut again.
            assert again.source_fit == kept.source_fit
            assert retry.provenance_json["source_fit_request"] == {
                "mode": "crop",
                "width": 3,
                "height": 2,
            }
            assert retry.settings_json["denoise"] == 0.7


@pytest.mark.parametrize("cropped", [False, True])
async def test_an_edit_check_judges_a_crop_against_the_crop_and_other_edits_against_the_source(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, cropped: bool
) -> None:
    orchestrator = app.state.services.orchestrator
    monkeypatch.setattr(orchestrator, "_profile_has_verified_vision", lambda *_: True)
    chat_id, request = await prepared_crop_turn(app, client, monkeypatch)
    with SessionLocal() as session:
        install = ModelInstall(
            id="model_crop_checker",
            name="Constructed checker",
            role="chat",
            engine="mock",
            local_path="synthetic",
            active=True,
        )
        profile = ModelProfile(
            id="profile_crop_checker",
            name="Crop checker",
            role="chat",
            engine="mock",
            model_install_id=install.id,
        )
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.active_vision_profile_id = profile.id
        chat.vision_settings_json = {"verify_image_edits": True}
        session.add_all([install, profile])
        session.flush()
        mirror_legacy_chat_workflow_selections(session, chat, ["vision"])
        session.commit()
    if cropped:
        request["source_fit"] = {"mode": "crop", "width": 3, "height": 2}

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(f"/api/chats/{chat_id}/turns", json=request)
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            run = session.get(Run, response.json()["run"]["id"])
            assert run is not None
            if accepted_context(session, run) is None:
                orchestrator._freeze_turn_context(session, run)
                session.flush()
            snapshot = accepted_context(session, run)
            assert snapshot is not None and snapshot.verification_profile is not None
            source = request["input_artifact_ids"][0]
            source_job = session.scalar(select(Job).where(Job.run_id == run.id))
            assert source_job is not None
            # The source picture stands in for the result: what is checked here
            # is which picture the edit is judged against, before any model looks.
            queued = orchestrator._queue_image_edit_verification(
                session, run, source_job.id, [source]
            )
            assert queued is not None, run.provenance_json.get("image_edit_verification")
            job = session.get(Job, queued)
            assert job is not None
            if cropped:
                assert isinstance(snapshot.source_fit, SourceCropRecipe)
                assert snapshot.source_fit.cropped_artifact_id != source
                assert job.payload_json["source_artifact_id"] == (
                    snapshot.source_fit.cropped_artifact_id
                )
            else:
                assert snapshot.source_fit is None
                assert job.payload_json["source_artifact_id"] == source
