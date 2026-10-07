from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy.orm import Session

from .comfy_workflow_packages import analyze_comfyui_workflow_package
from .model_planner import workflow_artifact_contract
from .models import WorkflowDefinition, WorkflowRevision
from .revision_dependency_contract import persist_dependency_contract
from .schemas import WorkflowPackageDraftRequest

WORKFLOW_PACKAGE_DRAFT_MARKER = "workflow_package_draft"
MAX_COMPARED_GRAPH_CHARACTERS = 8_000_000


class WorkflowPackageDraftError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def canonical_package_graph(graph: dict[str, Any]) -> str:
    """Bound exact graph comparisons while retaining their existing Unicode identity."""

    encoded = json.dumps(
        graph, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    if len(encoded) > MAX_COMPARED_GRAPH_CHARACTERS:
        raise WorkflowPackageDraftError(
            "workflow-graph-too-large",
            "That workflow is too large to compare against the stored revision.",
            status_code=422,
        )
    return encoded


def workflow_package_draft_identity(canonical_graph: str) -> tuple[str, str, str]:
    """Return stable local identities for one exact source graph."""

    digest = hashlib.sha256(canonical_graph.encode("utf-8")).hexdigest()
    short = digest[:24]
    return f"wfpkgdraft_{short}", f"wfpkgdrev_{short}", digest


def stage_workflow_package_draft(
    session: Session, payload: WorkflowPackageDraftRequest
) -> tuple[WorkflowDefinition, WorkflowRevision]:
    """Stage an untrusted source draft in the caller's installation transaction.

    The caller commits this draft together with any accepted offer and jobs.
    Reopening a source preserves its original operation and any later revision.
    """

    analyze_comfyui_workflow_package(payload.ui_graph)
    canonical = canonical_package_graph(payload.ui_graph)
    workflow_id, revision_id, digest = workflow_package_draft_identity(canonical)
    definition = session.get(WorkflowDefinition, workflow_id)
    revision = session.get(WorkflowRevision, revision_id)
    if bool(definition) != bool(revision):
        raise WorkflowPackageDraftError(
            "workflow-package-draft-collision", "The workflow draft identity is already in use."
        )
    if definition and revision:
        if (
            revision.workflow_id != definition.id
            or revision.dependencies_json != workflow_package_draft_dependencies(digest)
            or canonical_package_graph(revision.ui_graph_json) != canonical
        ):
            raise WorkflowPackageDraftError(
                "workflow-package-draft-collision",
                "The workflow draft identity is already in use.",
            )
        if definition.current_revision_id == revision.id:
            definition.name = payload.name
            definition.description = payload.description
        return definition, revision

    dependencies = workflow_package_draft_dependencies(digest)
    definition = WorkflowDefinition(
        id=workflow_id,
        name=payload.name,
        operation=payload.operation.value,
        description=payload.description,
    )
    session.add(definition)
    revision = WorkflowRevision(
        id=revision_id,
        workflow_id=workflow_id,
        version=1,
        engine="comfyui",
        ui_graph_json=payload.ui_graph,
        api_graph_json={},
        input_schema_json={},
        dependencies_json=dependencies,
        trusted=False,
        artifact_sha256=workflow_artifact_contract(
            operation=payload.operation.value,
            engine="comfyui",
            api_graph={},
            input_schema={},
            dependencies=dependencies,
        ),
    )
    session.add(revision)
    session.flush()
    persist_dependency_contract(session, revision)
    definition.current_revision_id = revision.id
    return definition, revision


def workflow_package_draft_dependencies(graph_sha256: str) -> dict[str, Any]:
    return {WORKFLOW_PACKAGE_DRAFT_MARKER: {"graph_sha256": graph_sha256}}


def is_workflow_package_draft(revision: WorkflowRevision | None) -> bool:
    if not revision or not isinstance(revision.dependencies_json, dict):
        return False
    return isinstance(
        revision.dependencies_json.get(WORKFLOW_PACKAGE_DRAFT_MARKER),
        dict,
    )
