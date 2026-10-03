"""Permanent deletion erases copied recovery labels while keeping idempotent identities."""

import json

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_media_recovery_api import _media
from test_recovery_bulk_api import _apply, _batch
from test_recovery_lifecycle_boundaries import _restore, _trash
from test_workflow_recovery_api import _seed as _workflow

from local_lm.db import SessionLocal
from local_lm.models import (
    ArtifactLibraryEntry,
    RecoveryBatchRecord,
    RecoveryItem,
    RecoveryOperation,
    WorkflowFamily,
)


def _labels() -> str:
    with SessionLocal() as session:
        values = []
        for model in (RecoveryItem, RecoveryOperation, RecoveryBatchRecord):
            values.extend(
                [dict(row) for row in session.execute(model.__table__.select()).mappings()]
            )
        return json.dumps(values, default=str)


@pytest.mark.parametrize("kind", ["chat", "project", "workflow", "media"])
async def test_permanent_bulk_deletion_scrubs_labels_and_retains_exact_retry_results(
    app: FastAPI,
    client: AsyncClient,
    kind: str,
) -> None:
    marker = "Orchard deletion sentinel"
    if kind == "chat":
        created = (await client.post("/api/chats", json={"title": marker})).json()
        path = f"/api/chats/{created['id']}"
    elif kind == "project":
        created = (await client.post("/api/projects", json={"name": marker})).json()
        path = f"/api/projects/{created['id']}"
    elif kind == "workflow":
        family_id, _definition_id, _revision_id, _graph = _workflow()
        with SessionLocal() as session:
            session.get(WorkflowFamily, family_id).name = marker
            session.commit()
        path = f"/api/workflow-families/{family_id}"
    else:
        entry_id, _artifact_id = _media(app)
        with SessionLocal() as session:
            entry = session.get(ArtifactLibraryEntry, entry_id)
            assert entry is not None
            entry.display_name = marker
            entry.version += 1
            session.commit()
        path = f"/api/artifact-library/{entry_id}"
    first = await _trash(client, path, "trash-first-cycle")
    await _restore(client, first["deletion_id"])
    item = await _trash(client, path, "trash-second-cycle")
    assert marker in _labels()
    preview = await _batch(client, [item], "purge")
    response = await _apply(client, preview, "purge-labels")
    assert response.status_code == 200, response.text
    assert marker not in _labels()
    replay = await _apply(client, preview, "purge-labels")
    assert replay.status_code == 200 and replay.json() == response.json()


async def test_project_purge_scrubs_its_label_in_other_recovery_records(
    client: AsyncClient,
) -> None:
    marker = "Retired project sentinel"
    project = (await client.post("/api/projects", json={"name": marker})).json()
    chat = (
        await client.post(
            "/api/chats", json={"title": "Retained notes", "project_id": project["id"]}
        )
    ).json()
    await _trash(client, f"/api/chats/{chat['id']}", "trash-child-with-project-label")
    item = await _trash(client, f"/api/projects/{project['id']}", "trash-labelled-project")
    preview = await _batch(client, [item], "purge")
    response = await _apply(client, preview, "purge-project-label")
    assert response.status_code == 200, response.text
    assert marker not in _labels()
    assert "Retained notes" in _labels()
