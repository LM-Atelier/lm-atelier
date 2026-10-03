"""Legacy profile updates and selections cannot recreate deleted workflow authority."""

import copy

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_chat_recovery import _command, _impact
from test_workflow_recovery_graph import _consumer

from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import (
    Chat,
    ChatWorkflowSelection,
    ModelProfile,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowProfileCompatibility,
    WorkflowRevision,
)
from local_lm.workflow_compatibility import (
    WorkflowSelectionInvalid,
    ensure_legacy_profile_workflow,
    reconcile_legacy_workflow_compatibility,
    resolve_chat_workflow_selection,
)
from local_lm.workflow_selection import WorkflowFamilySelectionError, resolve_workflow_family


async def _deleted(client: AsyncClient, action: str) -> tuple[str, str]:
    with SessionLocal() as session:
        profile = ModelProfile(name="Garden legacy renderer", role="image", engine="mock")
        session.add(profile)
        session.flush()
        reconcile_legacy_workflow_compatibility(session)
        mapping = session.get(WorkflowProfileCompatibility, profile.id)
        family_id, profile_id = mapping.workflow_family_id, profile.id
        if action == "purge-history":
            definition = session.scalar(
                select(WorkflowDefinition).where(
                    WorkflowDefinition.family_id == family_id,
                    WorkflowDefinition.variant_key == "create",
                )
            )
            revision = WorkflowRevision(
                definition=definition,
                version=1,
                engine="mock",
                api_graph_json={"1": {"class_type": "MockImage"}},
            )
            session.add(revision)
            session.flush()
            definition.current_revision_id = revision.id
            _consumer(session, "run", revision.id, "complete")
        session.commit()
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    response = await client.post(
        f"/api/workflow-families/{family_id}/trash",
        json=_command(preview, "trash-legacy-producer"),
    )
    assert response.status_code == 200, response.text
    if action != "trash":
        item = response.json()
        impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
        transition = "restore" if action == "restore" else "purge"
        command = _command(impact, f"{transition}-legacy-producer")
        if transition == "purge":
            command["acknowledgement"] = "permanently-delete"
        response = await client.post(
            f"/api/recovery-items/{item['deletion_id']}/{transition}", json=command
        )
        assert response.status_code == 200, response.text
    return profile_id, family_id


@pytest.mark.parametrize("action", ["trash", "purge", "purge-history"])
async def test_a_legacy_profile_update_leaves_its_deleted_workflow_rows_unchanged(
    client: AsyncClient, action: str
) -> None:
    profile_id, family_id = await _deleted(client, action)
    models = (
        WorkflowFamily,
        WorkflowDefinition,
        WorkflowRevision,
        WorkflowPreference,
        WorkflowProfileCompatibility,
    )
    with SessionLocal() as session:
        before = {
            model.__tablename__: copy.deepcopy(
                [dict(row) for row in session.execute(select(model.__table__)).mappings()]
            )
            for model in models
        }
        profile = session.get(ModelProfile, profile_id)
        profile.name = "Renamed garden legacy renderer"
        assert ensure_legacy_profile_workflow(session, profile) is None
        session.commit()
        for model in models:
            assert [
                dict(row) for row in session.execute(select(model.__table__)).mappings()
            ] == before[model.__tablename__]
        assert profile.name == "Renamed garden legacy renderer"
        if action == "purge":
            assert session.get(WorkflowFamily, family_id) is None


@pytest.mark.parametrize("action", ["trash", "purge", "purge-history", "restore"])
async def test_a_chat_cannot_select_a_deleted_workflow_through_its_legacy_profile(
    client: AsyncClient, action: str
) -> None:
    profile_id, _family_id = await _deleted(client, action)
    created = await client.post("/api/chats", json={"title": "Garden notes"})
    assert created.status_code == 201
    chat = created.json()
    with SessionLocal() as session:
        before = session.scalar(select(func.count()).select_from(Chat))
        selections = session.scalar(select(func.count()).select_from(ChatWorkflowSelection))
    response = await client.patch(
        f"/api/chats/{chat['id']}", json={"active_image_profile_id": profile_id}
    )
    assert response.status_code == 409
    assert response.json()["code"] == "workflow-selection-unavailable"
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Chat)) == before
        assert (
            session.get(Chat, chat["id"]).active_image_profile_id == chat["active_image_profile_id"]
        )
        assert session.scalar(select(func.count()).select_from(ChatWorkflowSelection)) == selections


@pytest.mark.parametrize("action", ["trash", "purge", "purge-history", "restore"])
async def test_a_legacy_chat_without_a_mirrored_selection_cannot_run_an_unavailable_family(
    client: AsyncClient, action: str
) -> None:
    profile_id, _family_id = await _deleted(client, action)
    with SessionLocal() as session:
        chat = Chat(title="Legacy garden notes", active_image_profile_id=profile_id)
        session.add(chat)
        session.commit()
        assert not session.scalars(
            select(ChatWorkflowSelection).where(ChatWorkflowSelection.chat_id == chat.id)
        ).all()
        with pytest.raises(WorkflowSelectionInvalid) as refused:
            resolve_chat_workflow_selection(session, chat, "image")
        assert refused.value.reason == "family_unavailable"
        assert not session.dirty


async def test_a_cached_graphless_text_workflow_cannot_run_after_disabled_restore(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        profile = ModelProfile(name="Garden text", role="chat", engine="mock")
        session.add(profile)
        session.flush()
        family = ensure_legacy_profile_workflow(session, profile)
        assert family is not None
        session.commit()
        family_id = family.id
    with SessionLocal() as reader:
        old_family = reader.get(WorkflowFamily, family_id)
        old_preference = reader.scalar(
            select(WorkflowPreference).where(
                WorkflowPreference.workflow_family_id == family_id,
                WorkflowPreference.selector_capability == "chat",
            )
        )
        assert old_family.enabled and old_preference.enabled
        selection = resolve_workflow_family(
            reader,
            capability="chat",
            operation=Operation.TEXT,
            mode="explicit",
            workflow_family_id=family_id,
            engine="mock",
        )
        assert selection.compatibility and selection.workflow_revision_id is None
        preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
        response = await client.post(
            f"/api/workflow-families/{family_id}/trash",
            json=_command(preview, "trash-cached-text-workflow"),
        )
        assert response.status_code == 200
        item = response.json()
        impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
        response = await client.post(
            f"/api/recovery-items/{item['deletion_id']}/restore",
            json=_command(impact, "restore-cached-text-workflow"),
        )
        assert response.status_code == 200
        with pytest.raises(WorkflowFamilySelectionError) as refused:
            resolve_workflow_family(
                reader,
                capability="chat",
                operation=Operation.TEXT,
                mode="explicit",
                workflow_family_id=family_id,
                engine="mock",
            )
        assert refused.value.reason == "family_disabled"
        assert old_family.enabled and old_preference.enabled
        enabled = await client.patch(f"/api/workflow-families/{family_id}", json={"enabled": True})
        assert enabled.status_code == 200, enabled.text
        with pytest.raises(WorkflowFamilySelectionError) as refused:
            resolve_workflow_family(
                reader,
                capability="chat",
                operation=Operation.TEXT,
                mode="explicit",
                workflow_family_id=family_id,
                engine="mock",
            )
        assert refused.value.reason == "selector_disabled"
        selected = await client.put(
            f"/api/workflow-families/{family_id}/preferences/chat", json={"enabled": True}
        )
        assert selected.status_code == 200, selected.text
        selection = resolve_workflow_family(
            reader,
            capability="chat",
            operation=Operation.TEXT,
            mode="explicit",
            workflow_family_id=family_id,
            engine="mock",
        )
        assert selection.compatibility and selection.workflow_revision_id is None
