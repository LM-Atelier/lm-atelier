"""Deleted workflow families retain revision identity without regaining execution authority."""

import copy

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select, update
from test_chat_recovery import _command, _impact
from test_workflow_library_api import _family

from local_lm.db import SessionLocal
from local_lm.models import (
    RecoveryItem,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)


def _seed(archived: bool = False) -> tuple[str, str, str, dict]:
    with SessionLocal() as session:
        family, definition, revision, preference, *_ = _family(name="Garden layout")
        family.archived = archived
        family.enabled = not archived
        preference.enabled = not archived
        revision.trusted = False
        session.add_all([family, definition, revision, preference])
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        return family.id, definition.id, revision.id, copy.deepcopy(revision.api_graph_json)


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_workflow_recovery_preserves_archive_and_never_retrusts_or_executes(
    client: AsyncClient, archived: bool, action: str
) -> None:
    family_id, definition_id, revision_id, graph = _seed(archived)
    assert (await client.get(f"/api/workflow-families/{family_id}")).status_code == 200
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    assert preview["kind"] == "workflow_family"
    assert preview["available_actions"] == ["trash"]
    assert preview["counts"]["workflow_definitions"] == 1
    assert preview["counts"]["workflow_revisions"] == 1
    command = _command(preview, "trash-garden-workflow")
    trashed = await client.post(f"/api/workflow-families/{family_id}/trash", json=command)
    assert trashed.status_code == 200, trashed.text
    item = trashed.json()
    repeated = await client.post(f"/api/workflow-families/{family_id}/trash", json=command)
    assert repeated.status_code == 200 and repeated.json() == item
    assert (await client.get(f"/api/workflow-families/{family_id}")).status_code == 404
    families = (
        await client.get("/api/workflow-families", params={"include_archived": True})
    ).json()
    assert not any(family["id"] == family_id for family in families)
    page = await client.get("/api/recovery-items", params={"kind": "workflow_family"})
    assert page.status_code == 200 and page.json()["items"] == [item]
    with SessionLocal() as session:
        family = session.get(WorkflowFamily, family_id)
        assert family.archived == archived and not family.enabled
        revision = session.get(WorkflowRevision, revision_id)
        assert revision.api_graph_json == graph and not revision.trusted
    impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
    transition = _command(impact, f"{action}-garden-workflow")
    if action == "purge":
        transition["acknowledgement"] = "permanently-delete"
    result = await client.post(
        f"/api/recovery-items/{item['deletion_id']}/{action}", json=transition
    )
    assert result.status_code == 200, result.text
    assert result.json()["reclaimed_bytes"] == 0
    replay = await client.post(
        f"/api/recovery-items/{item['deletion_id']}/{action}", json=transition
    )
    assert replay.status_code == 200 and replay.json() == result.json()
    with SessionLocal() as session:
        family = session.get(WorkflowFamily, family_id)
        if action == "restore":
            assert family.archived == archived and not family.enabled
            assert session.get(WorkflowRevision, revision_id).api_graph_json == graph
            assert not session.get(WorkflowRevision, revision_id).trusted
            preferences = session.scalars(
                select(WorkflowPreference).where(WorkflowPreference.workflow_family_id == family_id)
            ).all()
            assert preferences and all(
                not row.enabled and not row.is_default for row in preferences
            )
        else:
            assert family is None
            assert session.get(WorkflowDefinition, definition_id) is None
            assert session.get(WorkflowRevision, revision_id) is None
    assert (await client.get("/api/recovery-items", params={"kind": "workflow_family"})).json()[
        "items"
    ] == []


async def test_workflow_trash_refuses_an_active_default_without_changing_it(
    client: AsyncClient,
) -> None:
    family_id, _definition_id, _revision_id, _graph = _seed()
    with SessionLocal() as session:
        session.execute(
            update(WorkflowPreference)
            .where(WorkflowPreference.selector_capability == "image")
            .values(is_default=False)
        )
        preference = session.scalar(
            select(WorkflowPreference).where(WorkflowPreference.workflow_family_id == family_id)
        )
        preference.is_default = True
        session.commit()
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    assert preview["available_actions"] == []
    assert "active_selection" in preview["conflicts"]
    refused = await client.post(
        f"/api/workflow-families/{family_id}/trash",
        json=_command(preview, "trash-default-workflow"),
    )
    assert refused.status_code == 409
    with SessionLocal() as session:
        assert session.get(WorkflowFamily, family_id).enabled
        preference = session.scalar(
            select(WorkflowPreference).where(WorkflowPreference.workflow_family_id == family_id)
        )
        assert preference.is_default and preference.enabled
        assert (
            session.scalar(
                select(RecoveryItem).where(
                    RecoveryItem.kind == "workflow_family", RecoveryItem.subject_id == family_id
                )
            )
            is None
        )
