from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.adapters.base import GeneratedAsset, MediaEvent, MediaRequest
from local_lm.db import SessionLocal
from local_lm.models import Artifact, Job, Message, Run, WorkflowDefinition, WorkflowRevision
from local_lm.scheduler import JobClaim
from local_lm.schemas import TurnRequest

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.mark.parametrize(
    "later_change",
    [
        "prompt_settings",
        "input_bindings",
        "workflow_graph",
        "workflow_selection",
        "mask_binding",
        "engine_default",
        "auxiliary_provenance",
        "completion",
        "missing_workflow",
    ],
)
async def test_accepted_media_request_uses_frozen_inputs(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, later_change: str
) -> None:
    orchestrator = app.state.services.orchestrator
    artifacts = []
    for index in range(2):
        response = await client.post(
            "/api/artifacts",
            files={"file": ("neutral.png", PNG + bytes([index]), "image/png")},
        )
        assert response.status_code == 201
        artifacts.append(response.json()["id"])
    source_id, mask_id = artifacts
    graph = {"nodes": [{"inputs": {"image": "${input_image}", "mask": "${mask}"}}]}
    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Neutral masked editor", operation="image_to_image")
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
                "properties": {"mask": {"type": "object", "x-lm-atelier-kind": "mask"}},
            },
            dependencies_json={},
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        revision_id = revision.id
        source = session.get(Artifact, source_id)
        mask = session.get(Artifact, mask_id)
        assert source is not None and mask is not None
        expected_paths = [orchestrator.artifacts.resolve(source)]
        expected_mask_path = str(orchestrator.artifacts.resolve(mask))
        session.commit()
    chat = (await client.post("/api/chats", json={"title": "Frozen media request"})).json()
    seen: list[MediaRequest] = []

    async def generate(request: MediaRequest) -> Any:
        seen.append(request)
        if later_change == "completion":
            with SessionLocal() as session:
                executing = session.get(Run, request.run_id)
                assert executing is not None
                executing.standalone_prompt = "Later metadata."
                executing.settings_json = {"seed": 999}
                session.commit()
            yield MediaEvent(
                type="complete",
                assets=[
                    GeneratedAsset(
                        content=PNG + b"output",
                        kind="image",
                        media_type="image/png",
                        name="neutral-output.png",
                    )
                ],
            )
        else:
            yield MediaEvent(type="cancelled")

    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)
    monkeypatch.setattr(
        orchestrator,
        "_ensure_media_worker",
        AsyncMock(side_effect=AssertionError("Later default launched another engine")),
    )
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(
                    text="Repaint the selected area blue.",
                    mode="image",
                    input_artifact_ids=[source_id],
                    workflow_revision_id=revision_id,
                    settings={"mask": {"artifact_id": mask_id, "feather_px": 4, "invert": False}},
                ),
                freeze_context=True,
                activate_branch=False,
            )
            run = session.get(Run, accepted.run.id)
            assert run is not None
            run_id = run.id
            assert run.workflow_revision_id == revision_id
            expected_prompt = orchestrator._media_prompt(run)
            expected_settings = deepcopy(run.settings_json)
            job = session.scalar(select(Job).where(Job.run_id == run_id))
            assert job is not None
            job.status = "running"
            claim = JobClaim(token="accepted-media-attempt", attempt=1)
            job.claim_owner = claim.token
            job.attempt = claim.attempt
            job_id = job.id
            if later_change == "prompt_settings":
                run.standalone_prompt = "A later different request."
                run.settings_json = {**run.settings_json, "seed": 999, "width": 64}
                run.operation = "text_to_video"
            elif later_change == "input_bindings":
                user = session.get(Message, run.user_message_id)
                assert user is not None
                user.parts[:] = [part for part in user.parts if not part.artifact_id]
                run.provenance_json = {**run.provenance_json, "input_artifact_ids": []}
            elif later_change == "workflow_graph":
                stored_revision = session.get(WorkflowRevision, revision_id)
                assert stored_revision is not None
                stored_revision.api_graph_json = {"later": {"inputs": {}}}
            elif later_change == "workflow_selection":
                run.workflow_revision_id = None
            elif later_change == "auxiliary_provenance":
                run.provenance_json = {
                    **run.provenance_json,
                    "auxiliary_assets": {"trigger_words_applied": ["later added prompt"]},
                }
            elif later_change == "missing_workflow":
                stored_revision = session.get(WorkflowRevision, revision_id)
                assert stored_revision is not None
                session.delete(stored_revision)
            elif later_change == "mask_binding":
                run.settings_json = {
                    key: value for key, value in run.settings_json.items() if key != "mask"
                }
            session.commit()
        if later_change == "engine_default":
            monkeypatch.setattr(orchestrator.engines.settings, "media_engine", "comfyui")
            with pytest.raises(RuntimeError, match="Accepted media engine"):
                await orchestrator._execute_media(job_id, run_id, claim)
            assert seen == []
            return
        if later_change == "missing_workflow":
            with pytest.raises(RuntimeError, match="Accepted workflow revision"):
                await orchestrator._execute_media(job_id, run_id, claim)
            assert seen == []
            return
        await orchestrator._execute_media(job_id, run_id, claim)
        if later_change == "completion":
            with SessionLocal() as session:
                outputs = [
                    artifact
                    for artifact in session.scalars(select(Artifact))
                    if artifact.metadata_json.get("run_id") == run_id
                ]
                assert len(outputs) == 1
                assert outputs[0].metadata_json["semantic_description"] != "Later metadata."
                assert outputs[0].metadata_json["settings"] == expected_settings
        assert len(seen) == 1
        request = seen[0]
        assert request.operation == "image_to_image"
        assert request.prompt == expected_prompt
        assert request.workflow == graph
        assert request.input_paths == expected_paths
        assert request.parameters.get("mask", {}).get("artifact_id") == mask_id
        assert request.parameters["mask"]["path"] == expected_mask_path
        assert {key: value for key, value in request.parameters.items() if key != "mask"} == {
            key: value for key, value in expected_settings.items() if key != "mask"
        }
        from local_lm.models import RunContextArtifact

        with SessionLocal() as session:
            assert mask_id in set(
                session.scalars(
                    select(RunContextArtifact.artifact_id).where(
                        RunContextArtifact.run_id == run_id
                    )
                )
            ), "the frozen mask was not retained with its accepted context"


@pytest.mark.parametrize("later_change", ["selection", "provenance", "manifest", "unavailable"])
async def test_accepted_media_lora_binding(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, later_change: str
) -> None:
    from local_lm.auxiliary_assets import (
        checkpoint_lora_extension,
        resolve_lora_stack,
        transform_lora_graph,
    )
    from local_lm.domain import utcnow
    from local_lm.models import ModelAssetInstall

    graph = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "neutral.safetensors"},
        },
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    extension = checkpoint_lora_extension(graph)
    assert extension is not None
    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Neutral LoRA workflow", operation="text_to_image")
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
        asset = ModelAssetInstall(
            name="Neutral ink",
            kind="lora",
            family="sdxl",
            local_path="C:/managed/neutral-ink",
            size_bytes=1024,
            manifest_json={
                "sha256": "b" * 64,
                "comfy_name": "neutral-ink.safetensors",
                "metadata": {"trigger_words": ["ink wash"]},
            },
            active=True,
            verified_at=utcnow(),
        )
        session.add_all([revision, asset])
        session.flush()
        definition.current_revision_id = revision.id
        revision_id, asset_id = revision.id, asset.id
        session.commit()
    chat = (await client.post("/api/chats", json={"title": "Frozen LoRA binding"})).json()
    orchestrator = app.state.services.orchestrator
    seen: list[MediaRequest] = []

    async def generate(request: MediaRequest) -> Any:
        seen.append(request)
        yield MediaEvent(type="cancelled")

    monkeypatch.setattr(orchestrator.engines.media, "generate", generate)
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(
                    text="Draw a blue circle.",
                    mode="image",
                    workflow_revision_id=revision_id,
                    settings={
                        "loras": [
                            {
                                "asset_id": asset_id,
                                "model_strength": 0.8,
                                "clip_strength": 0.65,
                                "enabled": True,
                            }
                        ]
                    },
                ),
                freeze_context=True,
                activate_branch=False,
            )
            run = session.get(Run, accepted.run.id)
            assert run is not None
            revision = session.get(WorkflowRevision, revision_id)
            assert revision is not None
            resolved = resolve_lora_stack(session, revision, run.settings_json["loras"])
            expected_graph = transform_lora_graph(
                graph,
                extension,
                [
                    {key: item[key] for key in ("comfy_name", "model_strength", "clip_strength")}
                    for item in resolved.provenance
                    if item["enabled"]
                ],
            )
            expected_prompt = orchestrator._media_prompt(run)
            assert "ink wash" in expected_prompt
            job = session.scalar(select(Job).where(Job.run_id == run.id))
            assert job is not None
            job.status = "running"
            claim = JobClaim(token="accepted-lora-attempt", attempt=1)
            job.claim_owner, job.attempt = claim.token, claim.attempt
            job_id, run_id = job.id, run.id
            if later_change == "selection":
                run.settings_json = {**run.settings_json, "loras": []}
            elif later_change == "provenance":
                run.provenance_json = {**run.provenance_json, "auxiliary_assets": {}}
            else:
                asset = session.get(ModelAssetInstall, asset_id)
                assert asset is not None
                if later_change == "manifest":
                    asset.manifest_json = {**asset.manifest_json, "sha256": "c" * 64}
                else:
                    asset.active = False
            session.commit()
        if later_change in {"manifest", "unavailable"}:
            with pytest.raises((RuntimeError, ValueError)):
                await orchestrator._execute_media(job_id, run_id, claim)
            assert seen == []
        else:
            await orchestrator._execute_media(job_id, run_id, claim)
            assert len(seen) == 1
            assert seen[0].workflow == expected_graph
            assert seen[0].prompt == expected_prompt
            assert seen[0].parameters["loras"][0]["asset_id"] == asset_id
