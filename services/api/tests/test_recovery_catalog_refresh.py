"""Catalog refresh creates usable workflows without changing deleted family history."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_built_in_matting_workflow import MODEL_FILE, _asset, _object_info
from test_chat_recovery import _command, _impact
from test_recovery_lifecycle_boundaries import _trash
from test_workflow_recovery_graph import _consumer

from local_lm.comfy_templates import ComfyTemplate, CompiledComfyTemplate
from local_lm.db import SessionLocal
from local_lm.models import ModelInstall, WorkflowDefinition, WorkflowRevision


@pytest.mark.parametrize("kind", ["catalog", "matting"])
@pytest.mark.parametrize("state", ["trash", "purge-history"])
async def test_installed_workflow_refresh_avoids_deleted_template_families(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    state: str,
) -> None:
    manager = app.state.services.downloads
    settings = app.state.services.settings
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="recovery_enlargement",
            path=settings.data_dir / "enlargement.json",
            role="image",
            operation="image_to_image",
            score=100,
            sha256="a" * 64,
            dependencies=(),
        ),
        ui_graph={"nodes": []},
        api_graph={
            "source": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
            "scale": {
                "class_type": "ImageScaleBy",
                "inputs": {"image": ["source", 0], "scale_by": 4},
            },
            "save": {"class_type": "SaveImage", "inputs": {"images": ["scale", 0]}},
        },
        input_schema={"type": "object", "properties": {}},
    )
    object_info = _object_info(MODEL_FILE)
    monkeypatch.setattr(
        manager, "media_adapter", SimpleNamespace(object_info=AsyncMock(return_value=object_info))
    )
    monkeypatch.setattr(manager.comfy_templates, "compile", Mock(return_value=compiled))
    with SessionLocal() as session:
        if kind == "catalog":
            install = ModelInstall(
                name="Catalog enlargement",
                role="image",
                engine="comfyui",
                local_path="enlargement",
                active=True,
                manifest_json={
                    "workflow_template_id": compiled.template.id,
                    "workflow_template_sha256": "a" * 64,
                },
            )
            session.add(install)
            session.flush()
            old = manager._ensure_template_workflow(session, compiled, install)
        else:
            asset = _asset(session)
            old = manager._ensure_matting_workflow(session, asset, object_info)
        definition = session.get(WorkflowDefinition, old.workflow_id)
        assert definition is not None and definition.family_id
        family_id, definition_id = definition.family_id, definition.id
        if state == "purge-history":
            _consumer(session, "run", old.id, "complete")
        session.commit()
    item = await _trash(client, f"/api/workflow-families/{family_id}", "trash-catalog-template")
    if state == "purge-history":
        impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
        response = await client.post(
            f"/api/recovery-items/{item['deletion_id']}/purge",
            json=_command(impact, "purge-catalog-history")
            | {"acknowledgement": "permanently-delete"},
        )
        assert response.status_code == 200, response.text
    with SessionLocal() as session:
        before = [
            dict(row)
            for row in session.execute(
                select(WorkflowRevision.__table__).where(
                    WorkflowRevision.workflow_id == definition_id
                )
            ).mappings()
        ]
    assert await manager.refresh_installed_media_workflows() == 1
    assert await manager.refresh_installed_media_workflows() == 0
    with SessionLocal() as session:
        assert [
            dict(row)
            for row in session.execute(
                select(WorkflowRevision.__table__).where(
                    WorkflowRevision.workflow_id == definition_id
                )
            ).mappings()
        ] == before
        alternatives = list(
            session.scalars(
                select(WorkflowDefinition).where(
                    WorkflowDefinition.name == definition.name,
                    WorkflowDefinition.id != definition_id,
                    WorkflowDefinition.current_revision_id.is_not(None),
                )
            )
        )
        assert len(alternatives) == 1
        assert alternatives[0].family_id != family_id
