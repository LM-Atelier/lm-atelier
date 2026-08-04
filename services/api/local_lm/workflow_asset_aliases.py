"""Materialize exact workflow filenames from immutable provider artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from sqlalchemy.orm import Session

from .adapters.contracts import ADAPTER_CONTRACT_VERSION
from .comfy_workflow_packages import WorkflowAssetReference
from .model_planner import (
    ACTIVATION_PROBE_VERSION,
    LAUNCH_CONTRACT_VERSION,
    PlannedArtifact,
    ResolvedInstallPlan,
    persist_install_plan,
)
from .models import InstallPlan
from .workflow_asset_bindings import (
    WorkflowAssetBindingError,
    WorkflowAssetPlanSelection,
    validate_workflow_asset_candidate,
)
from .workflow_asset_downloads import (
    WorkflowAssetDownloadError,
    install_plan_download_request,
)

WORKFLOW_ASSET_ALIAS_VERSION = 1


class WorkflowAssetAliasError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def materialize_workflow_asset_aliases(
    session: Session,
    references: Sequence[WorkflowAssetReference],
    selections: Sequence[WorkflowAssetPlanSelection],
    install_plans: Mapping[str, InstallPlan],
) -> tuple[tuple[WorkflowAssetPlanSelection, ...], dict[str, InstallPlan]]:
    """Return selections whose artifacts use the workflow's exact filenames.

    A provider's filename is provenance, while the workflow filename is a
    loader-visible runtime contract. When they differ, create a content-addressed
    one-artifact plan that preserves the provider identity but downloads to the
    workflow reference. No graph is rewritten and no filename is guessed.
    """

    references_by_name = {reference.filename.casefold(): reference for reference in references}
    materialized_plans = dict(install_plans)
    materialized: list[WorkflowAssetPlanSelection] = []
    used_sources: set[tuple[str, str]] = set()
    for selection in selections:
        reference = references_by_name.get(selection.reference_filename.casefold())
        if reference is None:
            materialized.append(selection)
            continue
        source_plan = install_plans.get(selection.install_plan_id)
        if source_plan is None:
            materialized.append(selection)
            continue
        source_key = (source_plan.id, selection.artifact_path)
        if source_key in used_sources:
            raise WorkflowAssetAliasError(
                "duplicate_alias_source",
                "one provider artifact cannot satisfy more than one workflow reference",
            )
        used_sources.add(source_key)
        if selection.artifact_path == reference.filename:
            materialized.append(selection)
            continue
        source_runtime = source_plan.runtime_contract_json
        if not isinstance(source_runtime, Mapping):
            raise WorkflowAssetAliasError(
                "invalid_install_plan", "install plan has no runtime contract"
            )
        if source_runtime.get("workflow_asset_alias"):
            raise WorkflowAssetAliasError(
                "nested_asset_alias", "a workflow asset alias cannot be aliased again"
            )
        try:
            install_plan_download_request(source_plan)
            source_artifact = validate_workflow_asset_candidate(
                reference, source_plan, selection.artifact_path
            )
            alias = _resolved_alias_plan(
                source_plan=source_plan,
                source_artifact=source_artifact,
                reference_filename=reference.filename,
            )
        except (WorkflowAssetBindingError, WorkflowAssetDownloadError) as exc:
            raise WorkflowAssetAliasError(exc.code, str(exc)) from exc
        derived = persist_install_plan(session, alias)
        materialized_plans[derived.id] = derived
        materialized.append(
            WorkflowAssetPlanSelection(
                reference_filename=reference.filename,
                install_plan_id=derived.id,
                artifact_path=reference.filename,
            )
        )
    return tuple(materialized), materialized_plans


def _resolved_alias_plan(
    *,
    source_plan: InstallPlan,
    source_artifact: Mapping[str, Any],
    reference_filename: str,
) -> ResolvedInstallPlan:
    if source_plan.engine != "comfyui" or source_plan.role not in {"image", "video"}:
        raise WorkflowAssetAliasError(
            "unsupported_asset_runtime", "workflow assets require a ComfyUI media plan"
        )
    source_path = _safe_path(source_artifact.get("path"), "invalid_artifact_path")
    destination_path = _safe_path(reference_filename, "invalid_asset_reference")
    kind = source_artifact.get("kind")
    target_folder = source_artifact.get("target_folder")
    size = source_artifact.get("size_bytes")
    digest = source_artifact.get("sha256")
    if (
        not isinstance(kind, str)
        or not kind
        or not isinstance(target_folder, str)
        or not target_folder
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or not isinstance(digest, str)
    ):
        raise WorkflowAssetAliasError(
            "unverified_plan_artifact", "workflow asset provenance is incomplete"
        )
    source_remote_id = source_artifact.get("source_remote_id")
    source_revision = source_artifact.get("source_revision")
    provider_source_path = source_artifact.get("source_path")
    if source_plan.provider == "huggingface":
        source_remote_id = source_remote_id or source_plan.remote_id
        source_revision = source_revision or source_plan.revision
        provider_source_path = provider_source_path or source_path
    artifact = PlannedArtifact(
        path=destination_path,
        kind=kind,
        target_folder=target_folder,
        size_bytes=size,
        sha256=digest,
        source_remote_id=source_remote_id if isinstance(source_remote_id, str) else None,
        source_revision=source_revision if isinstance(source_revision, str) else None,
        source_path=provider_source_path if isinstance(provider_source_path, str) else None,
        source_version_id=(
            str(source_artifact["source_version_id"])
            if source_artifact.get("source_version_id") is not None
            else None
        ),
        source_file_id=(
            str(source_artifact["source_file_id"])
            if source_artifact.get("source_file_id") is not None
            else None
        ),
    )
    return ResolvedInstallPlan(
        provider=source_plan.provider,
        remote_id=source_plan.remote_id,
        revision=source_plan.revision,
        role=source_plan.role,
        engine="comfyui",
        architecture=source_plan.architecture,
        family=source_plan.family,
        compatibility="supported",
        artifacts=(artifact,),
        runtime_contract={
            "engine": "comfyui",
            "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
            "launch_contract_version": LAUNCH_CONTRACT_VERSION,
            "workflow_template_id": None,
            "workflow_template_sha256": None,
            "workflow_compiler_version": None,
            "comfy_paths": {target_folder: "."},
            "workflow_component_folders": {destination_path: target_folder},
            "source_remote_id": None,
            "auxiliary_kind": None,
            "workflow_asset_kind": kind,
            "workflow_asset_alias": {
                "version": WORKFLOW_ASSET_ALIAS_VERSION,
                "source_plan_hash": source_plan.plan_hash,
                "source_artifact_path": source_path,
                "destination_path": destination_path,
            },
        },
        activation_probe={
            "version": ACTIVATION_PROBE_VERSION,
            "kind": "workflow_asset",
            "timeout_seconds": 300,
            "required": False,
        },
    )


def _safe_path(value: object, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1_000
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise WorkflowAssetAliasError(code, "asset path is not a safe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise WorkflowAssetAliasError(code, "asset path is not a safe relative path")
    return value
