"""Build validated workflow revisions and stage their ownership without committing."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .auxiliary_assets import validate_lora_workflow_contract
from .model_planner import workflow_artifact_contract
from .models import WorkflowDefinition, WorkflowRevision
from .revision_dependency_contract import (
    declared_dependency_contract_sha256,
    persist_dependency_contract,
)
from .schemas import WorkflowRevisionCreate
from .settings_registry import validate_workflow_input_schema
from .video_length import video_length_reaches_graph, workflow_video_length
from .workflow_edit_calibration import (
    edit_calibration_reaches_graph,
    validate_workflow_edit_calibration,
)
from .workflow_ownership import ensure_workflow_family_ownership


def build_workflow_revision(
    definition: WorkflowDefinition,
    payload: WorkflowRevisionCreate,
    *,
    version: int,
    engine: str,
    trusted: bool = False,
) -> WorkflowRevision:
    """Validate the same execution fields for previews and durable revision writes."""

    validate_lora_workflow_contract(payload.api_graph, payload.input_schema, payload.dependencies)
    validate_workflow_edit_calibration(payload.input_schema)
    validate_workflow_input_schema(payload.input_schema)
    workflow_video_length(payload.input_schema)
    video_length_reaches_graph(payload.api_graph, payload.input_schema)
    edit_calibration_reaches_graph(payload.api_graph, payload.input_schema)
    dependency_hash = declared_dependency_contract_sha256(payload.dependencies)
    return WorkflowRevision(
        workflow_id=definition.id,
        version=version,
        engine=engine,
        engine_version=payload.engine_version,
        ui_graph_json=payload.ui_graph,
        api_graph_json=payload.api_graph,
        input_schema_json=payload.input_schema,
        dependencies_json=payload.dependencies,
        dependency_contract_sha256=dependency_hash,
        capabilities_json=[],
        trusted=trusted,
        artifact_sha256=workflow_artifact_contract(
            operation=definition.operation,
            engine=engine,
            api_graph=payload.api_graph,
            input_schema=payload.input_schema,
            dependencies=payload.dependencies,
        ),
    )


def stage_workflow_revision(
    session: Session,
    definition: WorkflowDefinition,
    payload: WorkflowRevisionCreate,
    *,
    trusted: bool = False,
) -> WorkflowRevision:
    """Stage a new current revision, dependency slots and family in one transaction."""

    version = (
        session.scalar(
            select(func.max(WorkflowRevision.version)).where(
                WorkflowRevision.workflow_id == definition.id
            )
        )
        or 0
    )
    current = session.get(WorkflowRevision, definition.current_revision_id)
    revision = build_workflow_revision(
        definition,
        payload,
        version=version + 1,
        engine=current.engine if current else "comfyui",
        trusted=trusted,
    )
    session.add(revision)
    session.flush()
    persist_dependency_contract(session, revision)
    definition.current_revision_id = revision.id
    ensure_workflow_family_ownership(session, definition, revision)
    return revision
