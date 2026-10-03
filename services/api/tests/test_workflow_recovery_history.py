"""Permanent workflow deletion retains immutable identities needed by completed history."""

import copy

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from test_chat_recovery import _command, _impact
from test_workflow_recovery_api import _seed
from test_workflow_recovery_graph import _consumer

from local_lm.db import Base, SessionLocal
from local_lm.models import (
    RecoveryItem,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
    WorkflowUseCasePreset,
)


def _history() -> dict[str, list[dict]]:
    with SessionLocal() as session:
        return {
            name: [
                copy.deepcopy(dict(row))
                for row in session.execute(select(Base.metadata.tables[name])).mappings()
            ]
            for name in ("runs", "work_steps", "jobs", "run_context_snapshots")
        }


@pytest.mark.parametrize("kind", ["job", "step", "run", "snapshot"])
async def test_permanent_workflow_deletion_keeps_only_identity_and_completed_history(
    client: AsyncClient, kind: str
) -> None:
    family_id, definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        _consumer(session, kind, revision_id, "complete")
        revision = session.get(WorkflowRevision, revision_id)
        revision.trusted = True
        revision.ui_graph_json = {"nodes": [{"id": 1}]}
        revision.dependencies_json = {"note": "Garden layout configuration"}
        revision.capabilities_json = ["garden-layout"]
        revision.artifact_sha256 = "a" * 64
        session.commit()
    before = _history()
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    trashed = await client.post(
        f"/api/workflow-families/{family_id}/trash",
        json=_command(preview, "trash-workflow-history"),
    )
    assert trashed.status_code == 200, trashed.text
    item = trashed.json()
    impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
    assert impact["counts"]["references"] > 0
    assert impact["available_actions"] == ["restore", "purge"]
    command = {
        **_command(impact, "purge-workflow-history"),
        "acknowledgement": "permanently-delete",
    }
    result = await client.post(f"/api/recovery-items/{item['deletion_id']}/purge", json=command)
    assert result.status_code == 200, result.text
    repeated = await client.post(f"/api/recovery-items/{item['deletion_id']}/purge", json=command)
    assert repeated.status_code == 200 and repeated.json() == result.json()
    assert _history() == before
    with SessionLocal() as session:
        family = session.get(WorkflowFamily, family_id)
        definition = session.get(WorkflowDefinition, definition_id)
        revision = session.get(WorkflowRevision, revision_id)
        assert family is not None and not family.enabled
        assert definition is not None and definition.current_revision_id is None
        assert (
            revision is not None and revision.workflow_id == definition_id and revision.version == 1
        )
        assert revision.artifact_sha256 == "a" * 64 and not revision.trusted
        assert revision.ui_graph_json == revision.api_graph_json == revision.input_schema_json == {}
        assert revision.dependencies_json == {} and revision.capabilities_json == []
        assert not session.scalars(
            select(WorkflowPreference).where(WorkflowPreference.workflow_family_id == family_id)
        ).all()
        recovery = session.get(RecoveryItem, item["deletion_id"])
        assert recovery.state == "purged"
        with pytest.raises(IntegrityError, match="workflow-recovery-write-refused"):
            session.execute(
                update(WorkflowRevision)
                .where(WorkflowRevision.id == revision_id)
                .values(trusted=True)
            )
        session.rollback()
    assert (await client.get(f"/api/workflow-families/{family_id}")).status_code == 404
    assert (await client.get("/api/recovery-items", params={"kind": "workflow_family"})).json()[
        "items"
    ] == []
    refused = await client.post(
        f"/api/recovery-items/{item['deletion_id']}/restore",
        json=_command(impact, "restore-purged-workflow"),
    )
    assert refused.status_code == 409
    assert refused.json()["code"] == "recovery-item-not-recoverable"


async def test_saved_workflow_configuration_still_prevents_permanent_deletion(
    client: AsyncClient,
) -> None:
    family_id, _definition_id, revision_id, graph = _seed()
    with SessionLocal() as session:
        session.add(
            WorkflowUseCasePreset(
                name="Garden configuration",
                use_case="image_generation",
                settings_json={"workflow_revision_id": revision_id},
                is_default=False,
            )
        )
        session.commit()
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    response = await client.post(
        f"/api/workflow-families/{family_id}/trash",
        json=_command(preview, "trash-saved-configuration"),
    )
    assert response.status_code == 200
    item = response.json()
    impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
    assert impact["available_actions"] == ["restore"]
    response = await client.post(
        f"/api/recovery-items/{item['deletion_id']}/purge",
        json={
            **_command(impact, "purge-saved-configuration"),
            "acknowledgement": "permanently-delete",
        },
    )
    assert response.status_code == 409
    with SessionLocal() as session:
        assert session.get(WorkflowRevision, revision_id).api_graph_json == graph
