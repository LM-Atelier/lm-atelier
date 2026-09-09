from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import ModelInstall, ModelProfile, Run, WorkflowDefinition, WorkflowRevision
from local_lm.schemas import EngineCapabilities, TurnRequest


@pytest.mark.parametrize(
    "later_change",
    ["none", "run_selection", "workflow_details", "install_manifest", "inactive", "trust"],
)
async def test_media_evidence_describes_the_accepted_execution(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, later_change: str
) -> None:
    from local_lm.comfy_templates import COMFY_TEMPLATE_COMPILER_VERSION

    orchestrator = app.state.services.orchestrator
    recorder = Mock(return_value=SimpleNamespace(evidence_key="constructed-evidence"))
    monkeypatch.setattr("local_lm.orchestrator.record_capability_evidence", recorder)
    chat = (await client.post("/api/chats", json={"title": "Accepted capability evidence"})).json()
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session, chat["id"], TurnRequest(text="A blue paper boat", mode="image")
            )
            install = ModelInstall(
                name="Constructed image model",
                role="image",
                engine="comfyui",
                active=True,
                local_path="constructed-model.safetensors",
                manifest_json={
                    "expected_sha256": {"model.safetensors": "a" * 64},
                    "workflow_template_id": "image_edit",
                    "workflow_template_sha256": "b" * 64,
                },
            )
            definition = WorkflowDefinition(
                name="Constructed image contract", operation="text_to_image"
            )
            session.add_all([install, definition])
            session.flush()
            profile = ModelProfile(
                name="Constructed image profile",
                role="image",
                engine="comfyui",
                model_install_id=install.id,
            )
            performance = {"version": 1, "signals": [{"kind": "model-cache"}]}
            revision = WorkflowRevision(
                workflow_id=definition.id,
                version=1,
                engine="comfyui",
                trusted=True,
                api_graph_json={},
                artifact_sha256="d" * 64,
                input_schema_json={"x-lm-atelier-workflow-performance": performance},
                dependencies_json={
                    "model_install_ids": [install.id],
                    "compiler_version": COMFY_TEMPLATE_COMPILER_VERSION,
                    "template_id": "image_edit",
                    "template_sha256": "b" * 64,
                },
            )
            session.add_all([profile, revision])
            session.flush()
            run = session.get(Run, accepted.run.id)
            assert run is not None
            run.profile_id = profile.id
            run.workflow_revision_id = revision.id
            orchestrator._freeze_turn_context(session, run)
            session.flush()
            if later_change == "run_selection":
                run.profile_id = None
                run.workflow_revision_id = None
                run.operation = "text_to_video"
            elif later_change == "workflow_details":
                revision.artifact_sha256 = "e" * 64
                revision.input_schema_json = {"x-lm-atelier-workflow-performance": {"version": 2}}
            elif later_change == "install_manifest":
                install.manifest_json = {
                    **install.manifest_json,
                    "expected_sha256": {"later.safetensors": "f" * 64},
                }
            elif later_change == "inactive":
                install.active = False
            elif later_change == "trust":
                revision.trusted = False
            session.flush()
            capabilities = EngineCapabilities(
                engine="comfyui",
                version="constructed-runtime",
                roles=["image"],
                operations=["text_to_image"],
                formats=["png"],
                devices=["cpu:0"],
                streaming=False,
                tool_calling=False,
                settings=[],
                healthy=True,
            )
            result = orchestrator._record_successful_media_evidence(
                session, run, capabilities, output_count=1
            )
            if later_change in {"install_manifest", "inactive", "trust"}:
                assert result is None
                recorder.assert_not_called()
            else:
                assert result == "constructed-evidence"
                assert recorder.call_args.args[1].id == install.id
                assert recorder.call_args.kwargs["workflow_contract_version"] == "d" * 64
                assert recorder.call_args.kwargs["component_hashes"] == {
                    "model.safetensors": "a" * 64
                }
                assert recorder.call_args.kwargs["details"]["operation"] == "text_to_image"
                assert recorder.call_args.kwargs["details"]["workflow_performance"] == performance
            session.rollback()
