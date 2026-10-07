"""Recovery respects terminal work, hidden resource identity and permanent deletion."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from recovery_requests import permanently_delete_chat
from sqlalchemy import select
from test_chat_deletion import _image_exchange, _text_exchange, wait_for_run
from test_chat_recovery import _command, _impact
from test_project_recovery_api import _project
from test_turn_workflow_choices import _text_workflow

from local_lm.db import Base, SessionLocal
from local_lm.domain import JobStatus
from local_lm.models import Chat, Job, Run, WorkflowDefinition, WorkflowRevision


async def _trash(client: AsyncClient, path: str, key: str) -> dict[str, Any]:
    impact = await _impact(client, f"{path}/deletion-impact")
    response = await client.post(f"{path}/trash", json=_command(impact, key))
    assert response.status_code == 200, response.text
    return dict(response.json())


async def _restore(client: AsyncClient, deletion_id: str) -> None:
    impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    restored = await client.post(
        f"/api/recovery-items/{deletion_id}/restore",
        json=_command(impact, f"restore-{deletion_id}"),
    )
    assert restored.status_code == 200, restored.text


async def test_a_chat_interrupted_by_a_restart_can_be_trashed(
    app: FastAPI, client: AsyncClient
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Interrupted"})).json()
    exchange = await _text_exchange(client, chat["id"], "Describe a quiet harbour")
    run_id = exchange["run"]["id"]
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        assert job is not None
        # As a job looks when the app stops while it runs.
        job.status = JobStatus.RUNNING.value
        job.completed_at = None
        session.commit()
    app.state.services.orchestrator.recover_interrupted()
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        assert job is not None and job.status == JobStatus.INTERRUPTED.value

    impact = await _impact(client, f"/api/chats/{chat['id']}/deletion-impact")

    assert "trash" in impact["available_actions"], impact


async def test_renaming_a_chat_while_its_project_is_trashed_keeps_its_filing(
    client: AsyncClient,
) -> None:
    project_id, chat_id = await _project(client, "Filing project")
    trashed = await _trash(client, f"/api/projects/{project_id}", "trash-filing-project")
    # What the chat settings dialog sends after a title-only edit.
    renamed = await client.patch(
        f"/api/chats/{chat_id}", json={"title": "Renamed", "project_id": None, "archived": False}
    )
    assert renamed.status_code == 200, renamed.text
    await _restore(client, trashed["deletion_id"])

    metadata = (await client.get(f"/api/chats/{chat_id}/metadata")).json()

    assert metadata["project_id"] == project_id, metadata


async def test_an_image_turn_skips_a_trashed_workflow_family(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        definition = session.scalars(
            select(WorkflowDefinition)
            .where(
                WorkflowDefinition.operation == "text_to_image",
                WorkflowDefinition.current_revision_id.is_not(None),
            )
            .order_by(WorkflowDefinition.created_at.desc())
        ).first()
        assert definition is not None and definition.family_id
        family_id = definition.family_id
    await _trash(client, f"/api/workflow-families/{family_id}", "trash-image-family")
    chat = (await client.post("/api/chats", json={"title": "Image after trash"})).json()

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create an image of a red cube", "mode": "image"},
    )

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "turn-invalid"
    assert response.json()["detail"].startswith(
        "No ready workflow matches the active media engine."
    )
    with SessionLocal() as session:
        assert session.scalar(select(Run.id).where(Run.chat_id == chat["id"])) is None
        assert (
            session.scalar(
                select(Job.id).join(Run, Job.run_id == Run.id).where(Run.chat_id == chat["id"])
            )
            is None
        )


async def test_a_permanently_deleted_chat_leaves_no_title_behind(client: AsyncClient) -> None:
    marker = "Sentinel-orchard-417"
    project_id, _other = await _project(client, "Sentinel-project-417")
    chat = await client.post("/api/chats", json={"title": marker, "project_id": project_id})
    assert chat.status_code == 201, chat.text

    await permanently_delete_chat(client, chat.json()["id"])

    with SessionLocal() as session:
        for table in ("recovery_items", "recovery_operations", "recovery_batches"):
            rows = session.execute(select(Base.metadata.tables[table])).all()
            dumped = json.dumps([[str(value) for value in row] for row in rows])
            assert marker not in dumped, table
            assert "Sentinel-project-417" not in dumped, table


async def test_a_trashed_chats_edit_source_and_branches_are_hidden(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Edit source"})).json()
    exchange = await _text_exchange(client, chat["id"], "Describe a quiet harbour")
    user_message = exchange["user_message"]["id"]
    assert (await client.get(f"/api/messages/{user_message}/edit-source")).status_code == 200
    assert (await client.get(f"/api/chats/{chat['id']}/edited-branches")).status_code == 200
    await _trash(client, f"/api/chats/{chat['id']}", "trash-edit-source")

    source = await client.get(f"/api/messages/{user_message}/edit-source")
    branches = await client.get(f"/api/chats/{chat['id']}/edited-branches")

    assert (source.status_code, branches.status_code) == (404, 404), (source.text, branches.text)


async def test_importing_a_project_does_not_reuse_its_trashed_workflow(client: AsyncClient) -> None:
    project_id, chat_id = await _project(client, "Archive project")
    exchange = await _image_exchange(client, chat_id, "Create an image of a red cube")
    with SessionLocal() as session:
        run = session.get(Run, exchange["run"]["id"])
        assert run is not None and run.workflow_revision_id
        definition = session.scalar(
            select(WorkflowDefinition).where(
                WorkflowDefinition.current_revision_id == run.workflow_revision_id
            )
        )
        assert definition is not None and definition.family_id
        family_id = definition.family_id
    exported = await client.post(f"/api/projects/{project_id}/export")
    assert exported.status_code == 201, exported.text
    archive = await client.get(exported.json()["url"])
    await _trash(client, f"/api/workflow-families/{family_id}", "trash-archive-family")

    imported = await client.post(
        "/api/projects/import",
        files={"archive": ("archive.lm-atelier.zip", archive.content, "application/zip")},
    )

    assert imported.status_code == 201, imported.text
    with SessionLocal() as session:
        imported_runs = session.scalars(
            select(Run)
            .join(Chat, Run.chat_id == Chat.id)
            .where(Chat.project_id == imported.json()["id"])
        ).all()
        assert len(imported_runs) == 1
        imported_run = imported_runs[0]
        assert imported_run.workflow_revision_id != run.workflow_revision_id
        assert imported_run.workflow_revision_id is not None
        selection = imported_run.provenance_json["model_selection"]
        assert selection["workflow_family_id"] is None
        assert selection["workflow_definition_id"] != definition.id
        assert selection["workflow_revision_id"] == imported_run.workflow_revision_id


@pytest.mark.parametrize("receipt", ["activation", "lora", "nested"])
async def test_project_import_remaps_nested_workflow_identity_after_trash(
    client: AsyncClient, receipt: str
) -> None:
    project_id, chat_id = await _project(client, "Activated archive")
    with SessionLocal() as session:
        revision_id, _newer, profile_id, activation_id = _text_workflow(session, "ready")
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None and revision.definition.family_id
        family_id = revision.definition.family_id
        definition_id = revision.workflow_id
        session.commit()
    accepted = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={
            "text": "Describe a paper boat in one sentence",
            "mode": "text",
            "workflow_revision_id": revision_id,
        },
    )
    assert accepted.status_code == 202, accepted.text
    await wait_for_run(client, accepted.json()["run"]["id"])
    target = {
        "workflow_family_id": family_id,
        "workflow_definition_id": definition_id,
        "workflow_revision_id": revision_id,
    }
    with SessionLocal() as session:
        run = session.get(Run, accepted.json()["run"]["id"])
        assert run is not None
        assert run.provenance_json["model_selection"]["workflow_activation_id"] == activation_id
        provenance = dict(run.provenance_json)
        if receipt != "activation":
            provenance["model_selection"] = {
                **provenance["model_selection"],
                "workflow_activation_id": None,
            }
        if receipt == "lora":
            provenance["workflow_lora"] = {"override_resolution": {"target": dict(target)}}
            provenance["composition"] = {
                "workflow_native": {"override_resolution": {"target": dict(target)}}
            }
        elif receipt == "nested":
            provenance["history"] = [
                {"id": activation_id},
                {"family_id": family_id},
                {"revision_id": revision_id},
                {"id": definition_id},
                {"id": "unrelated-record", "revision_id": "unrelated-revision"},
            ]
        run.provenance_json = provenance
        session.commit()
    exported = await client.post(f"/api/projects/{project_id}/export")
    assert exported.status_code == 201, exported.text
    archive = await client.get(exported.json()["url"])
    await _trash(client, f"/api/workflow-families/{family_id}", f"trash-{receipt}-family")

    imported = await client.post(
        "/api/projects/import",
        files={"archive": ("archive.lm-atelier.zip", archive.content, "application/zip")},
    )

    assert imported.status_code == 201, imported.text
    with SessionLocal() as session:
        imported_run = session.scalar(
            select(Run).join(Chat).where(Chat.project_id == imported.json()["id"])
        )
        assert imported_run is not None
        assert imported_run.workflow_revision_id not in {None, revision_id}
        imported_revision = session.get(WorkflowRevision, imported_run.workflow_revision_id)
        assert imported_revision is not None
        provenance = imported_run.provenance_json
        selection = provenance["model_selection"]
        assert selection["workflow_activation_id"] is None
        assert selection["workflow_family_id"] is None
        assert selection["workflow_revision_id"] == imported_revision.id
        assert selection["profile_id"] == imported_run.profile_id
        assert imported_run.profile_id not in {None, profile_id}
        if receipt == "lora":
            expected = {
                "workflow_family_id": None,
                "workflow_definition_id": imported_revision.workflow_id,
                "workflow_revision_id": imported_revision.id,
            }
            assert provenance["workflow_lora"]["override_resolution"]["target"] == expected
            assert (
                provenance["composition"]["workflow_native"]["override_resolution"]["target"]
                == expected
            )
        elif receipt == "nested":
            assert provenance["history"] == [
                {"id": None},
                {"family_id": None},
                {"revision_id": imported_revision.id},
                {"id": imported_revision.workflow_id},
                {"id": "unrelated-record", "revision_id": "unrelated-revision"},
            ]
