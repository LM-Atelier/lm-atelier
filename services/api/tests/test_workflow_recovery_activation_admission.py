"""Activation paths read current family eligibility without mutating deleted history."""

import copy

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from test_chat_recovery import _command, _impact
from test_workflow_activations import _contract, _revision

from local_lm.db import SessionLocal
from local_lm.model_planner import workflow_artifact_contract
from local_lm.models import WorkflowActivation, WorkflowDefinition, WorkflowFamily
from local_lm.workflow_activation_requests import (
    WorkflowActivationCreate,
    activate_reviewed_revision,
    activation_subject,
)
from local_lm.workflow_activations import (
    WorkflowActivationError,
    activate_workflow_revision,
    revalidate_workflow_activation,
)


@pytest.mark.parametrize("action", ["trash", "restore", "restore-enabled"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("path", ["direct", "subject"])
async def test_activation_admission_reads_current_family_state_before_any_authority_write(
    client: AsyncClient, action: str, cached: bool, path: str
) -> None:
    with SessionLocal() as session:
        family = WorkflowFamily(name="Garden activation", enabled=True)
        session.add(family)
        session.flush()
        revision = _revision(session, _contract(), suffix="recovery-admission")
        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        definition.family_id = family.id
        definition.variant_key = "create"
        definition.current_revision_id = revision.id
        revision.trusted = True
        revision.dependencies_json = {"version": 1, "slots": []}
        revision.artifact_sha256 = workflow_artifact_contract(
            operation=definition.operation,
            engine=revision.engine,
            api_graph=revision.api_graph_json,
            input_schema=revision.input_schema_json,
            dependencies=revision.dependencies_json,
        )
        scope = activate_workflow_revision(session, revision, [])
        session.commit()
        family_id, definition_id, revision_id = family.id, definition.id, revision.id
        assert (
            activation_subject(session, definition_id, revision_id).workflow_revision_id
            == revision_id
        )
    with SessionLocal() as reader:
        old_family = reader.get(WorkflowFamily, family_id) if cached else None
        old_activation = reader.get(WorkflowActivation, scope.activation_id) if cached else None
        preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
        response = await client.post(
            f"/api/workflow-families/{family_id}/trash",
            json=_command(preview, "trash-activation-admission"),
        )
        assert response.status_code == 200, response.text
        if action != "trash":
            item = response.json()
            impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
            response = await client.post(
                f"/api/recovery-items/{item['deletion_id']}/restore",
                json=_command(impact, "restore-activation-admission"),
            )
            assert response.status_code == 200, response.text
        if action == "restore-enabled":
            enabled = await client.patch(
                f"/api/workflow-families/{family_id}", json={"enabled": True}
            )
            assert enabled.status_code == 200, enabled.text
        with SessionLocal() as session:
            before = copy.deepcopy(
                [
                    dict(row)
                    for row in session.execute(select(WorkflowActivation.__table__)).mappings()
                ]
            )
        assert all(not row["is_active"] and row["state"] == "disabled" for row in before)
        if action == "restore-enabled" and path == "subject":
            subject = activation_subject(reader, definition_id, revision_id)
            assert subject.workflow_revision_id == revision_id
            assert not reader.dirty
            with SessionLocal() as session:
                activation = session.get(WorkflowActivation, scope.activation_id)
                assert activation is not None
                assert not activation.is_active and activation.state == "disabled"
                result = activate_reviewed_revision(
                    session,
                    definition_id,
                    revision_id,
                    WorkflowActivationCreate(
                        workflow_artifact_sha256=subject.workflow_artifact_sha256,
                        dependency_contract_sha256=subject.dependency_contract_sha256,
                        selections=[],
                    ),
                )
                session.commit()
                assert result.id == scope.activation_id
                assert activation.is_active and activation.state == "ready"
            return
        with pytest.raises(WorkflowActivationError) as refused:
            if path == "direct":
                revalidate_workflow_activation(reader, scope.activation_id)
            else:
                activation_subject(reader, definition_id, revision_id)
        assert (
            refused.value.code
            == {
                "trash": "workflow_revision_unavailable",
                "restore": "workflow_family_unavailable",
                "restore-enabled": "workflow_activation_disabled",
            }[action]
        )
        assert not reader.dirty
        with SessionLocal() as session:
            assert [
                dict(row)
                for row in session.execute(select(WorkflowActivation.__table__)).mappings()
            ] == before
        if old_family is not None:
            assert old_family.enabled
        if old_activation is not None:
            assert old_activation.is_active is (action != "restore-enabled")
