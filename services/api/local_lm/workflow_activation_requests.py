"""Explicit activation requests preserve reviewed content and exact dependency identity."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from sqlalchemy.orm import Session

from .model_planner import workflow_artifact_contract
from .models import WorkflowDefinition, WorkflowFamily, WorkflowRevision
from .schemas import ApiModel
from .workflow_activations import (
    WorkflowActivationError,
    WorkflowRuntimeMaterializer,
    activate_workflow_revision,
)
from .workflow_bindings import WorkflowBindingSelection
from .workflow_dependencies import (
    WorkflowDependencyResourceKind,
    parse_workflow_dependency_contract,
    workflow_dependency_contract_payload,
    workflow_dependency_contract_sha256,
)
from .workflow_revision_reviews import review_is_current


class WorkflowActivationSelectionIn(ApiModel):
    slot_name: str = Field(min_length=1, max_length=200)
    requirement_key: str = Field(min_length=1, max_length=200)
    local_kind: WorkflowDependencyResourceKind
    local_id: str = Field(min_length=1, max_length=200)
    recorded_resource_identity_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mount: dict[str, Any] = Field(default_factory=dict)

    def binding(self) -> WorkflowBindingSelection:
        return WorkflowBindingSelection(
            self.slot_name,
            self.requirement_key,
            self.local_kind,
            self.local_id,
            self.recorded_resource_identity_sha256,
            self.mount,
        )


class WorkflowActivationCreate(ApiModel):
    workflow_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selections: list[WorkflowActivationSelectionIn] = Field(default_factory=list, max_length=512)


class WorkflowActivationSubject(ApiModel):
    workflow_revision_id: str
    workflow_artifact_sha256: str
    dependency_contract_sha256: str
    slots: list[dict[str, Any]]


class WorkflowActivationOut(ApiModel):
    id: str
    workflow_revision_id: str
    dependency_contract_sha256: str
    binding_sha256: str
    launch_sha256: str
    state: Literal["ready"] = "ready"
    is_active: Literal[True] = True


def _eligible_revision(session: Session, workflow_id: str, revision_id: str) -> WorkflowRevision:
    definition = session.get(WorkflowDefinition, workflow_id)
    revision = session.get(WorkflowRevision, revision_id)
    if definition is None or revision is None or revision.workflow_id != workflow_id:
        raise WorkflowActivationError(
            "workflow_revision_unavailable", "Workflow revision is unavailable"
        )
    if definition.current_revision_id != revision.id:
        raise WorkflowActivationError(
            "workflow_revision_not_current", "Workflow revision is not current"
        )
    if definition.family_id is not None:
        family = session.get(WorkflowFamily, definition.family_id)
        if family is None or family.archived or not family.enabled:
            raise WorkflowActivationError(
                "workflow_family_unavailable", "Workflow family is unavailable"
            )
    artifact = revision.artifact_sha256
    contract = revision.dependency_contract_sha256
    if (
        not isinstance(artifact, str)
        or re.fullmatch(r"[0-9a-f]{64}", artifact) is None
        or not isinstance(contract, str)
        or re.fullmatch(r"[0-9a-f]{64}", contract) is None
        or not review_is_current(session, definition, revision)
    ):
        raise WorkflowActivationError(
            "workflow_review_unavailable", "Workflow review is unavailable"
        )
    actual_artifact = workflow_artifact_contract(
        operation=definition.operation,
        engine=revision.engine,
        api_graph=revision.api_graph_json,
        input_schema=revision.input_schema_json,
        dependencies=revision.dependencies_json,
    )
    declared = parse_workflow_dependency_contract(revision.dependencies_json)
    if artifact != actual_artifact or contract != workflow_dependency_contract_sha256(declared):
        raise WorkflowActivationError("workflow_contract_drift", "Workflow content changed")
    return revision


def activation_subject(
    session: Session, workflow_id: str, revision_id: str
) -> WorkflowActivationSubject:
    revision = _eligible_revision(session, workflow_id, revision_id)
    contract = parse_workflow_dependency_contract(revision.dependencies_json)
    return WorkflowActivationSubject(
        workflow_revision_id=revision.id,
        workflow_artifact_sha256=str(revision.artifact_sha256),
        dependency_contract_sha256=str(revision.dependency_contract_sha256),
        slots=workflow_dependency_contract_payload(contract)["slots"],
    )


def activate_reviewed_revision(
    session: Session,
    workflow_id: str,
    revision_id: str,
    payload: WorkflowActivationCreate,
    *,
    runtime_materializer: WorkflowRuntimeMaterializer | None = None,
    custom_node_root: Path | None = None,
    registry_environment_root: Path | None = None,
) -> WorkflowActivationOut:
    revision = _eligible_revision(session, workflow_id, revision_id)
    _assert_requested_identity(revision, payload)
    scope = activate_workflow_revision(
        session,
        revision,
        [choice.binding() for choice in payload.selections],
        runtime_materializer=runtime_materializer,
        custom_node_root=custom_node_root,
        registry_environment_root=registry_environment_root,
    )
    # The activation writer now owns the transaction. Re-read durable approval
    # and content before the caller commits; an earlier read is not authority.
    session.expire_all()
    fresh = _eligible_revision(session, workflow_id, revision_id)
    _assert_requested_identity(fresh, payload)
    return WorkflowActivationOut(
        id=scope.activation_id,
        workflow_revision_id=scope.workflow_revision_id,
        dependency_contract_sha256=payload.dependency_contract_sha256,
        binding_sha256=scope.binding_sha256,
        launch_sha256=scope.launch_sha256,
    )


def _assert_requested_identity(
    revision: WorkflowRevision, payload: WorkflowActivationCreate
) -> None:
    if (
        payload.workflow_artifact_sha256 != revision.artifact_sha256
        or payload.dependency_contract_sha256 != revision.dependency_contract_sha256
    ):
        raise WorkflowActivationError("workflow_contract_drift", "Workflow content changed")
