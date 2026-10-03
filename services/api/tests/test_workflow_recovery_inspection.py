"""Deleted workflows do not supply inspections or graphs to an engine."""

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from test_chat_recovery import _command, _impact
from test_prompt_library_api import _contract, _create_payload
from test_workflow_recovery_api import _seed
from test_workflow_recovery_graph import _consumer

from local_lm.db import SessionLocal
from local_lm.models import (
    Project,
    ProjectWorkflowSelection,
    PromptTemplateDefinition,
    PromptTemplateRevision,
    WorkflowDefinition,
    WorkflowRevision,
)
from local_lm.prompt_library import prompt_template_workflow_revision_is_ready


def _fixed_contract(revision_id: str) -> dict:
    return _contract(
        resource_policy={
            "mode": "fixed",
            "workflow_revision_id": revision_id,
            "lora_policy": {"mode": "none"},
        }
    )


async def test_saved_template_reference_prevents_workflow_purge_without_changing_its_contract(
    client: AsyncClient,
) -> None:
    family_id, _definition_id, revision_id, graph = _seed()
    with SessionLocal() as session:
        session.get(WorkflowRevision, revision_id).trusted = True
        session.commit()
    contract = _fixed_contract(revision_id)
    created = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="create-saved-garden-resource",
            name="Garden resource",
            contract=contract,
        ),
    )
    assert created.status_code == 201
    with SessionLocal() as session:
        saved = session.scalar(select(PromptTemplateRevision))
        saved_id, saved_contract, saved_digest = (
            saved.id,
            saved.contract_json,
            saved.contract_sha256,
        )
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    trashed = await client.post(
        f"/api/workflow-families/{family_id}/trash",
        json=_command(preview, "trash-saved-template-workflow"),
    )
    assert trashed.status_code == 200
    item = trashed.json()
    preview = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
    assert preview["available_actions"] == ["restore"]
    refused = await client.post(
        f"/api/recovery-items/{item['deletion_id']}/purge",
        json={
            **_command(preview, "purge-saved-template-workflow"),
            "acknowledgement": "permanently-delete",
        },
    )
    assert refused.status_code == 409
    with SessionLocal() as session:
        saved = session.get(PromptTemplateRevision, saved_id)
        assert (saved.contract_json, saved.contract_sha256) == (saved_contract, saved_digest)
        assert session.get(WorkflowRevision, revision_id).api_graph_json == graph


async def test_sql_template_writer_cannot_add_a_reference_after_workflow_trash(
    client: AsyncClient,
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        session.add(PromptTemplateDefinition(id="garden-template", name="Garden resource"))
        session.commit()
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    assert (
        await client.post(
            f"/api/workflow-families/{family_id}/trash",
            json=_command(preview, "trash-before-sql-template"),
        )
    ).status_code == 200
    with SessionLocal() as session:
        with pytest.raises(IntegrityError, match="workflow-recovery-write-refused"):
            session.execute(
                PromptTemplateRevision.__table__.insert().values(
                    id="garden-template-revision",
                    prompt_template_id="garden-template",
                    version=1,
                    schema_version=1,
                    contract_json=_fixed_contract(revision_id),
                    contract_sha256="a" * 64,
                )
            )
        session.rollback()
        assert session.get(PromptTemplateRevision, "garden-template-revision") is None


@pytest.mark.parametrize("action", ["trash", "purge"])
@pytest.mark.parametrize("inspection", ["geometry", "resolve", "source", "validate"])
async def test_deleted_workflow_inspections_refuse_before_engine_or_source_access(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    inspection: str,
) -> None:
    family_id, definition_id, revision_id, graph = _seed()
    with SessionLocal() as session:
        _consumer(session, "run", revision_id, "complete")
        session.commit()
    calls = []

    async def validate(value: dict) -> list[str]:
        calls.append(value)
        return []

    monkeypatch.setattr(app.state.services.engines.media, "validate_workflow", validate)
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    trashed = await client.post(
        f"/api/workflow-families/{family_id}/trash",
        json=_command(preview, "trash-inspection-workflow"),
    )
    assert trashed.status_code == 200
    if action == "purge":
        item = trashed.json()
        preview = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
        response = await client.post(
            f"/api/recovery-items/{item['deletion_id']}/purge",
            json={
                **_command(preview, "purge-inspection-workflow"),
                "acknowledgement": "permanently-delete",
            },
        )
        assert response.status_code == 200
    geometry = f"/api/workflow-revisions/{revision_id}/output-geometry"
    if inspection == "geometry":
        response = await client.get(geometry)
    elif inspection == "resolve":
        response = await client.post(f"{geometry}/resolve", json={})
    elif inspection == "source":
        response = await client.post(
            f"{geometry}/match-source", json={"source_artifact_id": "sha256:" + "a" * 64}
        )
    else:
        response = await client.post(f"/api/workflows/{definition_id}/validate")
    assert response.status_code == 404
    assert response.json()["code"] == "workflow-not-found"
    assert calls == []
    with SessionLocal() as session:
        assert session.get(WorkflowRevision, revision_id).api_graph_json == (
            graph if action == "trash" else {}
        )


@pytest.mark.parametrize("include_archived", [False, True])
async def test_operation_filters_hide_deleted_workflows_and_restore_the_original_choice(
    client: AsyncClient,
    include_archived: bool,
) -> None:
    family_id, definition_id, _revision_id, _graph = _seed()
    with SessionLocal() as session:
        session.execute(update(WorkflowDefinition).values(operation="text"))
        session.get(WorkflowDefinition, definition_id).operation = "image_to_video"
        session.commit()
    params = {"include_archived": include_archived}
    path = "/api/workflow-family-operations"
    original = await client.get(path, params=params)
    assert original.status_code == 200 and "image_to_video" in original.json()
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    trashed = await client.post(
        f"/api/workflow-families/{family_id}/trash",
        json=_command(preview, "trash-operation-choice"),
    )
    assert trashed.status_code == 200
    hidden = await client.get(path, params=params)
    assert hidden.status_code == 200 and "image_to_video" not in hidden.json()
    item = trashed.json()
    preview = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
    restored = await client.post(
        f"/api/recovery-items/{item['deletion_id']}/restore",
        json=_command(preview, "restore-operation-choice"),
    )
    assert restored.status_code == 200
    assert (await client.get(path, params=params)).json() == original.json()


@pytest.mark.parametrize("action", ["trash", "restore"])
@pytest.mark.parametrize("cached", [False, True])
async def test_unavailable_workflow_cannot_be_admitted_as_a_fixed_template_resource(
    client: AsyncClient,
    action: str,
    cached: bool,
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        session.get(WorkflowRevision, revision_id).trusted = True
        session.commit()
    with SessionLocal() as reader:
        revision = reader.get(WorkflowRevision, revision_id) if cached else None
        if revision is not None:
            assert prompt_template_workflow_revision_is_ready(
                reader, revision, expected_engine="mock"
            )
        preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
        trashed = await client.post(
            f"/api/workflow-families/{family_id}/trash",
            json=_command(preview, "trash-template-resource"),
        )
        assert trashed.status_code == 200
        if action == "restore":
            item = trashed.json()
            preview = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
            assert (
                await client.post(
                    f"/api/recovery-items/{item['deletion_id']}/restore",
                    json=_command(preview, "restore-template-resource"),
                )
            ).status_code == 200
        revision = revision or reader.get(WorkflowRevision, revision_id)
        assert not prompt_template_workflow_revision_is_ready(
            reader, revision, expected_engine="mock"
        )
    response = await client.post(
        "/api/prompt-templates",
        json=_create_payload(
            key="create-unavailable-resource",
            name="Garden resource",
            contract=_contract(
                resource_policy={
                    "mode": "fixed",
                    "workflow_revision_id": revision_id,
                    "lora_policy": {"mode": "none"},
                }
            ),
        ),
    )
    assert response.status_code == 409
    assert response.json()["code"] == "prompt-template-resources-unavailable"


async def test_project_revision_selection_refuses_a_deleted_workflow_without_partial_writes(
    client: AsyncClient,
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    created = await client.post("/api/projects", json={"name": "Garden project"})
    assert created.status_code == 201
    project_id = created.json()["id"]
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    assert (
        await client.post(
            f"/api/workflow-families/{family_id}/trash",
            json=_command(preview, "trash-before-project-selection"),
        )
    ).status_code == 200
    response = await client.put(
        f"/api/projects/{project_id}/workflow-selections/image",
        json={"mode": "revision", "workflow_revision_id": revision_id},
    )
    assert response.status_code == 404
    assert response.json()["code"] == "workflow-revision-not-found"
    with SessionLocal() as session:
        assert session.get(Project, project_id).image_workflow_revision_id is None
        assert (
            session.scalar(
                select(ProjectWorkflowSelection).where(
                    ProjectWorkflowSelection.project_id == project_id,
                )
            )
            is None
        )
