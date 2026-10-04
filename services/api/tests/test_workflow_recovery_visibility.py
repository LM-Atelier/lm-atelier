"""Deleted workflow definitions cannot appear as selectable, exportable or executable content."""

import pytest
from httpx2 import AsyncClient
from test_chat_recovery import _command, _impact
from test_workflow_recovery_api import _seed
from test_workflow_recovery_graph import _consumer

from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import WorkflowFamily, WorkflowRevision
from local_lm.output_recipe_check import _workflow_state
from local_lm.workflow_selection import (
    WorkflowFamilySelectionError,
    resolve_exact_workflow_revision,
)


async def _trashed(client: AsyncClient) -> tuple[str, str, str]:
    family_id, definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.trusted = True
        session.commit()
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    response = await client.post(
        f"/api/workflow-families/{family_id}/trash", json=_command(preview, "trash-hidden-workflow")
    )
    assert response.status_code == 200
    return family_id, definition_id, revision_id


@pytest.mark.parametrize("kind", ["list", "summaries", "choices", "detail", "schema", "export"])
async def test_deleted_workflow_is_absent_from_normal_reads_and_export(
    client: AsyncClient, kind: str
) -> None:
    _family_id, definition_id, revision_id = await _trashed(client)
    paths = {
        "list": "/api/workflows",
        "summaries": "/api/workflow-summaries",
        "choices": "/api/workflow-revision-choices",
        "detail": f"/api/workflows/{definition_id}",
        "schema": f"/api/workflow-revisions/{revision_id}/settings-schema",
        "export": f"/api/workflows/{definition_id}/export",
    }
    response = await client.get(paths[kind])
    if kind in {"list", "summaries", "choices"}:
        assert response.status_code == 200
        assert not any(
            row.get("id") == definition_id or row.get("workflow_id") == definition_id
            for row in response.json()
        )
    else:
        assert response.status_code == 404


async def test_a_deleted_exact_workflow_revision_is_not_accepted_for_generation(
    client: AsyncClient,
) -> None:
    _family_id, _definition_id, revision_id = await _trashed(client)
    with SessionLocal() as session, pytest.raises(WorkflowFamilySelectionError) as refused:
        resolve_exact_workflow_revision(
            session,
            revision_id,
            capability="image",
            operation=Operation.TEXT_TO_IMAGE,
            engine="mock",
        )
    assert refused.value.reason == "revision_missing"


async def test_a_permanently_deleted_historical_workflow_is_hidden_and_not_ready(
    client: AsyncClient,
) -> None:
    family_id, definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.trusted = True
        revision.artifact_sha256 = "a" * 64
        _consumer(session, "run", revision_id, "complete")
        session.commit()
        assert _workflow_state(session, "a" * 64) == "present"
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    response = await client.post(
        f"/api/workflow-families/{family_id}/trash", json=_command(preview, "trash-record-workflow")
    )
    assert response.status_code == 200
    with SessionLocal() as session:
        assert _workflow_state(session, "a" * 64) == "missing"
    item = response.json()
    impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
    response = await client.post(
        f"/api/recovery-items/{item['deletion_id']}/purge",
        json={**_command(impact, "purge-record-workflow"), "acknowledgement": "permanently-delete"},
    )
    assert response.status_code == 200
    assert (await client.get(f"/api/workflows/{definition_id}")).status_code == 404
    assert (await client.get(f"/api/workflows/{definition_id}/export")).status_code == 404
    with SessionLocal() as session:
        assert session.get(WorkflowRevision, revision_id) is not None
        assert _workflow_state(session, "a" * 64) == "missing"
        with pytest.raises(WorkflowFamilySelectionError) as refused:
            resolve_exact_workflow_revision(
                session,
                revision_id,
                capability="image",
                operation=Operation.TEXT_TO_IMAGE,
                engine="mock",
            )
        assert refused.value.reason == "revision_missing"


@pytest.mark.parametrize("cached", [False, True])
async def test_a_restored_disabled_workflow_is_not_accepted_through_its_exact_revision(
    client: AsyncClient, cached: bool
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.trusted = True
        session.commit()
    with SessionLocal() as reader:
        old_family = reader.get(WorkflowFamily, family_id) if cached else None
        if old_family is not None:
            assert old_family.enabled
        preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
        response = await client.post(
            f"/api/workflow-families/{family_id}/trash",
            json=_command(preview, "trash-disabled-revision"),
        )
        assert response.status_code == 200
        item = response.json()
        impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
        response = await client.post(
            f"/api/recovery-items/{item['deletion_id']}/restore",
            json=_command(impact, "restore-disabled-revision"),
        )
        assert response.status_code == 200
        with pytest.raises(WorkflowFamilySelectionError) as refused:
            resolve_exact_workflow_revision(
                reader,
                revision_id,
                capability="image",
                operation=Operation.TEXT_TO_IMAGE,
                engine="mock",
            )
        assert refused.value.reason == "family_disabled"
        if old_family is not None:
            assert old_family.enabled


async def test_a_restored_workflow_record_remains_inactive_until_the_family_is_enabled(
    client: AsyncClient,
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.trusted = True
        revision.artifact_sha256 = "b" * 64
        session.commit()
        assert _workflow_state(session, "b" * 64) == "present"
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    response = await client.post(
        f"/api/workflow-families/{family_id}/trash",
        json=_command(preview, "trash-inactive-record"),
    )
    assert response.status_code == 200
    item = response.json()
    impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
    response = await client.post(
        f"/api/recovery-items/{item['deletion_id']}/restore",
        json=_command(impact, "restore-inactive-record"),
    )
    assert response.status_code == 200
    with SessionLocal() as reader:
        family = reader.get(WorkflowFamily, family_id)
        assert family is not None and not family.enabled
        assert _workflow_state(reader, "b" * 64) == "inactive"
        with SessionLocal() as writer:
            stored_family = writer.get(WorkflowFamily, family_id)
            assert stored_family is not None
            stored_family.enabled = True
            writer.commit()
        assert not family.enabled
        assert _workflow_state(reader, "b" * 64) == "present"
