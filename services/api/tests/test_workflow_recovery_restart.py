"""Startup does not recreate or mutate a deleted legacy workflow family."""

import pytest
from sqlalchemy import select
from test_chat_recovery import _command, _impact
from test_media_recovery_restart import _open
from test_workflow_recovery_graph import _consumer

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    ModelProfile,
    RecoveryItem,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.workflow_compatibility import (
    compatibility_family_id,
    reconcile_legacy_workflow_compatibility,
)


@pytest.mark.parametrize("action", ["trash", "purge", "purge-history"])
async def test_restart_leaves_deleted_compatibility_workflows_and_their_deadlines_unchanged(
    settings: Settings, action: str
) -> None:
    async with _open(settings) as (_app, client):
        with SessionLocal() as session:
            profile = ModelProfile(name="Garden renderer", role="image", engine="mock")
            session.add(profile)
            session.flush()
            reconcile_legacy_workflow_compatibility(session)
            family_id = compatibility_family_id(profile.id)
            profile_id = profile.id
            if action == "purge-history":
                definition = session.scalar(
                    select(WorkflowDefinition).where(
                        WorkflowDefinition.family_id == family_id,
                        WorkflowDefinition.variant_key == "create",
                    )
                )
                assert definition is not None
                revision = WorkflowRevision(
                    definition=definition,
                    version=1,
                    engine="mock",
                    api_graph_json={"1": {"class_type": "MockImage"}},
                    trusted=True,
                )
                session.add(revision)
                session.flush()
                definition.current_revision_id = revision.id
                _consumer(session, "run", revision.id, "complete")
            session.commit()
        preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
        trash_command = _command(preview, "trash-before-workflow-restart")
        response = await client.post(
            f"/api/workflow-families/{family_id}/trash", json=trash_command
        )
        assert response.status_code == 200, response.text
        item = response.json()
        expected_replay = dict(item)
        if action != "trash":
            impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
            response = await client.post(
                f"/api/recovery-items/{item['deletion_id']}/purge",
                json={
                    **_command(impact, "purge-before-workflow-restart"),
                    "acknowledgement": "permanently-delete",
                },
            )
            assert response.status_code == 200, response.text
            expected_replay["display_label"] = "Deleted item"
        with SessionLocal() as session:
            stored_profile = session.get(ModelProfile, profile_id)
            assert stored_profile is not None
            stored_profile.name = "Renamed garden renderer"
            session.commit()
            recovery = session.get(RecoveryItem, item["deletion_id"])
            assert recovery is not None
            identity = (
                recovery.deleted_at,
                recovery.purge_after,
                recovery.state,
                recovery.subject_revision,
            )
            revisions = session.execute(
                select(WorkflowRevision.__table__)
                .join(WorkflowDefinition, WorkflowDefinition.id == WorkflowRevision.workflow_id)
                .where(WorkflowDefinition.family_id == family_id)
                .order_by(WorkflowRevision.id)
            ).all()

    async with _open(settings) as (_app, client):
        assert (await client.get(f"/api/workflow-families/{family_id}")).status_code == 404
        repeated = await client.post(
            f"/api/workflow-families/{family_id}/trash", json=trash_command
        )
        assert repeated.status_code == 200 and repeated.json() == expected_replay
        with SessionLocal() as session:
            recovery = session.get(RecoveryItem, item["deletion_id"])
            assert recovery is not None
            assert (
                recovery.deleted_at,
                recovery.purge_after,
                recovery.state,
                recovery.subject_revision,
            ) == identity
            assert (
                session.execute(
                    select(WorkflowRevision.__table__)
                    .join(WorkflowDefinition, WorkflowDefinition.id == WorkflowRevision.workflow_id)
                    .where(WorkflowDefinition.family_id == family_id)
                    .order_by(WorkflowRevision.id)
                ).all()
                == revisions
            )
            family = session.get(WorkflowFamily, family_id)
            if action == "purge":
                assert family is None
            else:
                assert family is not None and not family.enabled
            preferences = session.scalars(
                select(WorkflowPreference).where(WorkflowPreference.workflow_family_id == family_id)
            ).all()
            assert all(
                not preference.enabled and not preference.is_default for preference in preferences
            )
