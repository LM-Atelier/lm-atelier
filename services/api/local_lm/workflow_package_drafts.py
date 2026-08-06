from __future__ import annotations

from typing import Any

from .models import WorkflowRevision

WORKFLOW_PACKAGE_DRAFT_MARKER = "workflow_package_draft"


def workflow_package_draft_dependencies(graph_sha256: str) -> dict[str, Any]:
    return {WORKFLOW_PACKAGE_DRAFT_MARKER: {"graph_sha256": graph_sha256}}


def is_workflow_package_draft(revision: WorkflowRevision | None) -> bool:
    if not revision or not isinstance(revision.dependencies_json, dict):
        return False
    return isinstance(
        revision.dependencies_json.get(WORKFLOW_PACKAGE_DRAFT_MARKER),
        dict,
    )
