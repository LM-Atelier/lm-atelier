"""Keep accepted recipe values bound to the imported workflow identity."""

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_project_portability import _manifest, _rewrite_archive
from test_workflow_recipe_turn_admission import _recipe

from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.models import Chat, Run, WorkflowRevision
from local_lm.schemas import TurnRequest


@pytest.mark.parametrize("frozen,tampered", [(False, False), (True, False), (False, True)])
async def test_project_import_remaps_recipe_revision_without_resolving_new_defaults(
    app: FastAPI,
    client: AsyncClient,
    frozen: bool,
    tampered: bool,
) -> None:
    _recipe({"steps": 7})
    project = (await client.post("/api/projects", json={"name": "Portable recipe"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={
                "title": "Recipe source",
                "project_id": project["id"],
            },
        )
    ).json()
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await app.state.services.orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(text="A blue square", mode="image"),
                freeze_context=frozen,
            )
            original_revision_id = accepted.run.workflow_revision_id
        exported = await client.post(f"/api/projects/{project['id']}/export")
        assert exported.status_code == 201, exported.text
        archive = await client.get(f"/api/artifacts/{exported.json()['id']}/content")
        assert archive.status_code == 200
        archive_bytes = archive.content
        if tampered:
            manifest = _manifest(archive_bytes)
            manifest["runs"][0]["provenance_json"]["workflow_use_case_preset"][
                "workflow_revision_id"
            ] = "unrelated-revision"
            archive_bytes = _rewrite_archive(archive_bytes, manifest)
        with SessionLocal() as session:
            original = session.get(WorkflowRevision, original_revision_id)
            assert original is not None
            original.api_graph_json = {"changed": {"inputs": {"seed": 11}}}
            session.commit()
        imported = await client.post(
            "/api/projects/import",
            files={"archive": ("recipe.zip", archive_bytes, "application/zip")},
        )
        if tampered:
            assert imported.status_code == 422, imported.text
            assert "workflow-use-case-preset-snapshot-revision-mismatch" in imported.text
            return
        assert imported.status_code == 201, imported.text
        with SessionLocal() as session:
            run = session.scalar(
                select(Run)
                .join(Chat, Chat.id == Run.chat_id)
                .where(Chat.project_id == imported.json()["id"])
            )
            assert run is not None and run.workflow_revision_id != original_revision_id
            receipt = run.provenance_json["workflow_use_case_preset"]
            assert receipt["workflow_revision_id"] == run.workflow_revision_id
            assert receipt["settings_json"] == {"steps": 7}
            if frozen:
                snapshot = accepted_context(session, run)
                assert snapshot is not None and snapshot.workflow_use_case_preset is not None
                assert snapshot.workflow_use_case_preset.model_dump(mode="json") == receipt
