"""Replay ordinary media settings and enlarge retained ComfyUI revisions."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_workflow_recipe_request_cases import _picture, _workflow

from local_lm.accepted_turn_context import accepted_context
from local_lm.comfy_templates import ComfyTemplate, CompiledComfyTemplate
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import (
    Chat,
    Job,
    Message,
    MessagePart,
    ModelInstall,
    ModelProfile,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
)


async def legacy_generation(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frozen: bool = False,
    operation: str = "image_to_image",
) -> dict[str, str]:
    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")
    family_id, revision_id = _workflow("image_upscale", operation, "image")
    picture = await _picture(client)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.engine = "comfyui"
        revision.input_schema_json = {
            "type": "object",
            "properties": {
                "upscale_factor": {
                    "type": "number",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 8,
                    "x-lm-atelier-kind": "upscale",
                },
            },
        }
        revision.api_graph_json = {
            "source": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
            "scale": {
                "class_type": "ImageScaleBy",
                "inputs": {"image": ["source", 0], "scale_by": 4},
            },
            "output": {"class_type": "SaveImage", "inputs": {"images": ["scale", 0]}},
        }
        if operation == "text_to_image":
            revision.api_graph_json["source"] = {
                "class_type": "EmptyImage",
                "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0},
            }
        profile = ModelProfile(name="Neutral ComfyUI profile", role="image", engine="comfyui")
        session.add(profile)
        session.flush()
        chat = Chat(title="Recorded image study", active_image_profile_id=profile.id)
        session.add(chat)
        session.flush()
        user = Message(
            chat_id=chat.id,
            role="user",
            status="complete",
            parts=[MessagePart(position=0, type="text", text="A blue geometric square")]
            + (
                [MessagePart(position=1, type="image", artifact_id=picture)]
                if operation == "image_to_image"
                else []
            ),
        )
        session.add(user)
        session.flush()
        assistant = Message(
            chat_id=chat.id,
            role="assistant",
            status="complete",
            parent_id=user.id,
            parts=[MessagePart(position=0, type="image", artifact_id=picture)],
        )
        session.add(assistant)
        session.flush()
        run = Run(
            chat_id=chat.id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            operation=operation,
            status="complete",
            profile_id=profile.id,
            workflow_revision_id=revision_id,
            standalone_prompt="A blue geometric square",
            settings_json={"seed": 17, "upscale_factor": 2},
            completed_at=utcnow(),
            provenance_json={
                "upscale": False,
                "input_artifact_ids": [picture] if operation == "image_to_image" else [],
                "resolved_settings": {"seed": 17, "upscale_factor": 2},
            },
        )
        session.add(run)
        session.flush()
        session.add(
            Job(run_id=run.id, kind="image", status="complete", attempt=1, completed_at=utcnow())
        )
        chat.active_head_message_id = assistant.id
        if frozen:
            app.state.services.orchestrator._freeze_turn_context(session, run)
        session.commit()
        result = {
            "chat": chat.id,
            "user": user.id,
            "assistant": assistant.id,
            "run": run.id,
            "revision": revision_id,
            "family": family_id,
            "profile": profile.id,
            "picture": picture,
        }
    selected = await client.put(
        f"/api/chats/{result['chat']}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": family_id},
    )
    assert selected.status_code == 200, selected.text
    return result


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("action", ["regenerate", "try_another", "edit"])
@pytest.mark.parametrize("operation", ["image_to_image", "text_to_image"])
async def test_an_ordinary_comfyui_generation_replays_its_unbound_numeric_default(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    frozen: bool,
    operation: str,
) -> None:
    source = await legacy_generation(app, client, monkeypatch, frozen=frozen, operation=operation)
    with SessionLocal() as session:
        original = session.get(Run, source["run"])
        assert original is not None
        original_snapshot = accepted_context(session, original)
    async with app.state.services.scheduler.lease("primary"):
        if action == "regenerate":
            response = await client.post(
                f"/api/messages/{source['assistant']}/regenerate", json={"settings": {}}
            )
        elif action == "edit":
            response = await client.post(
                f"/api/messages/{source['user']}/edits",
                json={"text": "A green geometric square", "idempotency_key": "legacy-image-edit"},
            )
        else:
            response = await client.post(
                f"/api/chats/{source['chat']}/turns",
                json={
                    "text": "A blue geometric square",
                    "mode": "image",
                    "profile_id": source["profile"],
                    "workflow_revision_id": source["revision"],
                    "input_artifact_ids": [source["picture"]]
                    if operation == "image_to_image"
                    else [],
                    "settings": {"seed": -1, "upscale_factor": 2},
                },
            )
        assert response.status_code == 202, response.text
        run = response.json()["run"]
        assert "upscale_factor" not in run["settings_json"]
        assert run["provenance_json"]["upscale"] is False
        with SessionLocal() as session:
            original = session.get(Run, source["run"])
            assert original is not None and original.settings_json["upscale_factor"] == 2
            assert accepted_context(session, original) == original_snapshot


async def test_regenerate_uses_a_refreshed_fixed_revision_without_its_old_numeric_default(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = await legacy_generation(app, client, monkeypatch)
    with SessionLocal() as session:
        old = session.get(WorkflowRevision, source["revision"])
        assert old is not None
        new = WorkflowRevision(
            workflow_id=old.workflow_id,
            version=old.version + 1,
            engine="comfyui",
            trusted=True,
            api_graph_json=deepcopy(old.api_graph_json),
            dependencies_json={},
            input_schema_json={
                "properties": {
                    "upscale_factor": {
                        "type": "number",
                        "readOnly": True,
                        "x-lm-atelier-kind": "upscale",
                    }
                }
            },
        )
        session.add(new)
        session.flush()
        definition = session.get(WorkflowDefinition, old.workflow_id)
        assert definition is not None
        definition.current_revision_id = new.id
        session.commit()
        revision_id = new.id
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/messages/{source['assistant']}/regenerate", json={"settings": {}}
        )
        assert response.status_code == 202, response.text
        assert response.json()["run"]["workflow_revision_id"] == revision_id
        assert "upscale_factor" not in response.json()["run"]["settings_json"]


async def test_enhance_projects_a_retained_unbound_comfyui_declaration_as_fixed(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = await legacy_generation(app, client, monkeypatch)
    manager = app.state.services.downloads
    with SessionLocal() as session:
        retained = session.get(WorkflowRevision, source["revision"])
        assert retained is not None
        graph = deepcopy(retained.api_graph_json)
        schema = deepcopy(retained.input_schema_json)
        session.add(
            ModelInstall(
                name="Retained enlargement model",
                role="image",
                engine="comfyui",
                active=True,
                local_path=str(app.state.services.settings.model_dir / "retained-enlargement"),
                manifest_json={
                    "workflow_template_id": "image_enlargement",
                    "workflow_template_sha256": "a" * 64,
                },
            )
        )
        session.commit()
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="image_enlargement",
            path=app.state.services.settings.data_dir / "changed-template.json",
            role="image",
            operation="image_to_image",
            score=100,
            sha256="b" * 64,
            dependencies=(),
        ),
        ui_graph={},
        api_graph=graph,
        input_schema=schema,
    )

    async def object_info() -> dict[str, Any]:
        return {}

    monkeypatch.setattr(manager, "media_adapter", SimpleNamespace(object_info=object_info))
    compile_template = Mock(return_value=compiled)
    monkeypatch.setattr(manager.comfy_templates, "compile", compile_template)
    assert await manager.refresh_installed_media_workflows() == 0
    compile_template.assert_called_once()
    payload: dict[str, Any] = {
        "text": "Enlarge this picture",
        "mode": "image",
        "upscale": True,
        "input_artifact_ids": [source["picture"]],
    }
    preview = await client.post(f"/api/chats/{source['chat']}/upscale/preview", json=payload)
    assert preview.status_code == 200, preview.text
    assert preview.json()["factor"] is None and preview.json()["fixed_factor"] == 4
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(f"/api/chats/{source['chat']}/turns", json=payload)
        assert response.status_code == 202, response.text
        assert response.json()["run"]["provenance_json"]["upscale"] is True
        assert "upscale_factor" not in response.json()["run"]["settings_json"]
    with SessionLocal() as session:
        old = session.get(WorkflowRevision, source["revision"])
        assert (
            old is not None
            and old.input_schema_json["properties"]["upscale_factor"]["default"] == 2
        )
