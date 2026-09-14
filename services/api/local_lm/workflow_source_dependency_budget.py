"""Reserve space for every identity an approved installation can produce."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .model_manifests import COMFY_MODEL_FOLDERS
from .workflow_bindings import MAX_WORKFLOW_IDENTITY_TEXT_LENGTH, _stable_text
from .workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyError,
    canonical_workflow_dependency_json,
)
from .workflow_source_dependency_contract import WorkflowSourceDependencyBuilder

if TYPE_CHECKING:
    from .models import InstallPlan
    from .schemas import DownloadRequest
    from .workflow_package_execution_plan import WorkflowPackageExecutionPlan


def validate_workflow_source_dependency_budget(
    original: WorkflowDependencyContract,
    requests: Sequence[DownloadRequest],
    plans: Mapping[str, InstallPlan],
    extensions: Mapping[str, WorkflowPackageExecutionPlan],
    runtime_engine: str,
) -> None:
    builder = WorkflowSourceDependencyBuilder(original)
    for request in requests:
        asset_kind = request.workflow_asset_kind or request.auxiliary_kind
        if asset_kind:
            identities = [
                {
                    "kind": "model_asset",
                    "asset_kind": asset_kind,
                    "runtime_reference": path,
                    "sha256": request.expected_sha256[path],
                }
                for path in request.allow_patterns
            ]
            if not identities:
                raise WorkflowDependencyError(
                    "invalid_workflow_dependencies", "The planned asset has no component identity."
                )
            builder.include(
                "model_asset",
                max(identities, key=lambda item: len(canonical_workflow_dependency_json(item))),
            )
        else:
            plan = plans.get(request.install_plan_id or "")
            if plan is None:
                raise WorkflowDependencyError(
                    "invalid_workflow_dependencies", "The model installation plan is unavailable."
                )
            builder.include("model_install", _model_identity_budget(request, plan))
    for key in sorted(extensions):
        resolution = extensions[key].resolution
        builder.include(
            "registry_package",
            {
                "package_id": resolution.package_id,
                "package_version": resolution.declared_version,
                "archive_sha256": "0" * 64,
                "manifest_sha256": "0" * 64,
                "wheel_closure_sha256": "0" * 64,
                "wheel_environment_sha256": "0" * 64,
                "node_types": list(resolution.node_types),
            },
        )
    builder.include("runtime", {"engine": runtime_engine})
    builder.payload()


def _model_identity_budget(request: DownloadRequest, plan: InstallPlan) -> dict[str, Any]:
    artifacts = {item["path"]: item for item in plan.artifacts_json if item.get("required", True)}
    components = []
    for path in request.allow_patterns:
        artifact = artifacts[path]
        kind = _stable_text(artifact.get("kind"), "Model component kind")
        folder = _stable_text(artifact.get("target_folder"), "Model component folder")
        if kind in {"checkpoint", "unknown_safetensors"}:
            # Header inspection can refine generic kinds and folders. Reserve the
            # identity contract's full text bound and the longest allowed folder.
            kind = "x" * MAX_WORKFLOW_IDENTITY_TEXT_LENGTH
            folder = max(COMFY_MODEL_FOLDERS, key=len)
        components.append(
            {
                "kind": kind,
                "target_folder": folder,
                # A loader root can only shorten the original relative filename.
                "runtime_reference": path,
                "sha256": request.expected_sha256[path],
            }
        )
    return {
        "kind": "model_install",
        "role": request.role,
        "engine": request.engine,
        "components": components,
    }
