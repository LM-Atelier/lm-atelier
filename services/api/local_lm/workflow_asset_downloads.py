from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .comfy_templates import COMFY_TEMPLATE_COMPILER_VERSION
from .model_planner import INSTALL_RESOLVER_VERSION
from .models import InstallPlan
from .schemas import CatalogFileSource, DownloadRequest
from .workflow_asset_bindings import (
    MAX_WORKFLOW_ASSET_BINDINGS,
    BoundWorkflowAsset,
    WorkflowAssetBindingPlan,
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CIVITAI_ID = re.compile(r"^[1-9][0-9]{0,19}$")
_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64}|[1-9][0-9]{0,19})$")
_PROVIDERS = frozenset({"huggingface", "civitai"})


class WorkflowAssetDownloadError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def compose_workflow_asset_download_requests(
    binding_plan: WorkflowAssetBindingPlan,
    install_plans: Mapping[str, InstallPlan],
    *,
    expected_binding_plan_hash: str,
) -> tuple[DownloadRequest, ...]:
    """Build one immutable request per distinct plan without queueing it.

    The expected binding hash is the confirmation boundary: callers rebuild the
    binding from current server records, then compare it with the exact review
    the user accepted. DownloadManager remains the final enforcement boundary.
    """

    if (
        not _DIGEST.fullmatch(expected_binding_plan_hash)
        or binding_plan.plan_hash != expected_binding_plan_hash
    ):
        raise WorkflowAssetDownloadError(
            "binding_plan_changed", "workflow asset selections changed; review them again"
        )
    if len(binding_plan.assets) > MAX_WORKFLOW_ASSET_BINDINGS:
        raise WorkflowAssetDownloadError(
            "too_many_asset_downloads", "workflow has too many asset downloads"
        )

    ordered_plans: list[InstallPlan] = []
    seen_plan_ids: set[str] = set()
    seen_bindings: set[tuple[str, str]] = set()
    for binding in binding_plan.assets:
        plan = install_plans.get(binding.install_plan_id)
        if plan is None:
            raise WorkflowAssetDownloadError(
                "install_plan_not_found", "a selected install plan no longer exists"
            )
        _validate_bound_asset(binding, plan)
        binding_key = (plan.id, binding.artifact_path)
        if binding_key in seen_bindings:
            raise WorkflowAssetDownloadError(
                "duplicate_asset_download", "one install artifact was selected more than once"
            )
        seen_bindings.add(binding_key)
        if plan.id not in seen_plan_ids:
            seen_plan_ids.add(plan.id)
            ordered_plans.append(plan)

    return tuple(install_plan_download_request(plan) for plan in ordered_plans)


def install_plan_download_request(plan: InstallPlan) -> DownloadRequest:
    """Derive the only download request authorized by an immutable plan."""

    _validate_plan_state(plan)
    artifacts = _required_artifacts(plan)
    allow_patterns: list[str] = []
    expected_sha256: dict[str, str] = {}
    file_sources: dict[str, CatalogFileSource] = {}
    for artifact in artifacts:
        path = _safe_relative_path(artifact.get("path"))
        if path in expected_sha256:
            raise WorkflowAssetDownloadError(
                "ambiguous_install_artifact", "install plan repeats an artifact path"
            )
        size = artifact.get("size_bytes")
        digest = artifact.get("sha256")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(digest, str)
            or not _DIGEST.fullmatch(digest)
        ):
            raise WorkflowAssetDownloadError(
                "unverified_install_artifact",
                "every required workflow asset needs exact size and SHA-256",
            )
        allow_patterns.append(path)
        expected_sha256[path] = digest
        if plan.provider == "civitai":
            version_id = str(artifact.get("source_version_id") or "")
            file_id = str(artifact.get("source_file_id") or "")
            if (
                not _CIVITAI_ID.fullmatch(version_id)
                or not _CIVITAI_ID.fullmatch(file_id)
                or version_id != plan.revision
            ):
                raise WorkflowAssetDownloadError(
                    "incomplete_civitai_provenance",
                    "CivitAI artifact lacks exact version and file identity",
                )
        elif any(
            artifact.get(field) is not None
            for field in ("source_remote_id", "source_revision", "source_path")
        ):
            source_remote_id = artifact.get("source_remote_id")
            source_revision = artifact.get("source_revision")
            source_path = artifact.get("source_path")
            if (
                not isinstance(source_remote_id, str)
                or not source_remote_id
                or not isinstance(source_revision, str)
                or not _IMMUTABLE_REVISION.fullmatch(source_revision)
                or not isinstance(source_path, str)
            ):
                raise WorkflowAssetDownloadError(
                    "incomplete_artifact_source",
                    "a companion artifact lacks exact source identity",
                )
            source_path = _safe_relative_path(source_path)
            file_sources[path] = CatalogFileSource(
                remote_id=source_remote_id,
                revision=source_revision,
                filename=source_path,
                size_bytes=size,
                sha256=digest,
            )

    runtime = plan.runtime_contract_json
    if not isinstance(runtime, dict):
        raise WorkflowAssetDownloadError(
            "invalid_install_plan", "install plan has no runtime contract"
        )
    try:
        return DownloadRequest(
            install_plan_id=plan.id,
            remote_id=plan.remote_id,
            source_remote_id=runtime.get("source_remote_id"),
            revision=plan.revision,
            role=plan.role,  # type: ignore[arg-type]
            engine=plan.engine,
            allow_patterns=allow_patterns,
            expected_sha256=expected_sha256,
            file_sources=file_sources,
            comfy_paths=runtime.get("comfy_paths") or {},
            workflow_template_id=runtime.get("workflow_template_id"),
            workflow_template_sha256=runtime.get("workflow_template_sha256"),
            auxiliary_kind=runtime.get("auxiliary_kind"),
        )
    except ValueError as exc:
        raise WorkflowAssetDownloadError(
            "invalid_install_plan", "install plan cannot form a valid download request"
        ) from exc


def _validate_plan_state(plan: InstallPlan) -> None:
    if plan.provider not in _PROVIDERS:
        raise WorkflowAssetDownloadError(
            "unsupported_install_provider", "install plan provider is unsupported"
        )
    if (
        not isinstance(plan.id, str)
        or not plan.id
        or not isinstance(plan.remote_id, str)
        or not plan.remote_id
        or not isinstance(plan.revision, str)
        or not _IMMUTABLE_REVISION.fullmatch(plan.revision)
        or not isinstance(plan.plan_hash, str)
        or not _DIGEST.fullmatch(plan.plan_hash)
    ):
        raise WorkflowAssetDownloadError(
            "invalid_install_plan", "install plan lacks immutable identity"
        )
    if plan.status != "planned":
        raise WorkflowAssetDownloadError(
            "install_plan_not_pending", "install plan is no longer ready to queue"
        )
    if plan.compatibility != "supported" or plan.failure_code:
        raise WorkflowAssetDownloadError(
            "install_plan_not_supported", "install plan is not supported"
        )
    if plan.resolver_version != INSTALL_RESOLVER_VERSION:
        raise WorkflowAssetDownloadError(
            "install_contract_changed", "install contract changed; review the assets again"
        )
    runtime = plan.runtime_contract_json
    if not isinstance(runtime, dict):
        raise WorkflowAssetDownloadError(
            "invalid_install_plan", "install plan has no runtime contract"
        )
    if (
        runtime.get("workflow_template_id")
        and runtime.get("workflow_compiler_version") != COMFY_TEMPLATE_COMPILER_VERSION
    ):
        raise WorkflowAssetDownloadError(
            "workflow_contract_changed", "workflow contract changed; review the assets again"
        )


def _required_artifacts(plan: InstallPlan) -> list[Mapping[str, Any]]:
    raw = plan.artifacts_json
    if not isinstance(raw, list) or not raw or len(raw) > MAX_WORKFLOW_ASSET_BINDINGS:
        raise WorkflowAssetDownloadError(
            "invalid_install_plan", "install plan has no bounded artifact list"
        )
    required: list[Mapping[str, Any]] = []
    for artifact in raw:
        if not isinstance(artifact, Mapping) or not isinstance(
            artifact.get("required", True), bool
        ):
            raise WorkflowAssetDownloadError(
                "invalid_install_plan", "install plan has a malformed artifact"
            )
        if artifact.get("required", True) is True:
            required.append(artifact)
    if not required:
        raise WorkflowAssetDownloadError(
            "invalid_install_plan", "install plan has no required artifacts"
        )
    return required


def _validate_bound_asset(binding: BoundWorkflowAsset, plan: InstallPlan) -> None:
    if binding.install_plan_hash != plan.plan_hash:
        raise WorkflowAssetDownloadError(
            "install_plan_changed", "a selected install plan changed; review the assets again"
        )
    artifacts = plan.artifacts_json
    if not isinstance(artifacts, list) or len(artifacts) > MAX_WORKFLOW_ASSET_BINDINGS:
        raise WorkflowAssetDownloadError(
            "invalid_install_plan", "install plan has no bounded artifact list"
        )
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("path") == binding.artifact_path
    ]
    if len(matches) != 1:
        raise WorkflowAssetDownloadError(
            "binding_asset_changed", "the selected install artifact changed"
        )
    artifact = matches[0]
    if (
        plan.id != binding.install_plan_id
        or plan.provider != binding.provider
        or plan.remote_id != binding.remote_id
        or plan.revision != binding.revision
        or binding.reference_filename != binding.artifact_path
        or artifact.get("kind") != binding.artifact_kind
        or artifact.get("target_folder") != binding.target_folder
        or artifact.get("size_bytes") != binding.size_bytes
        or artifact.get("sha256") != binding.sha256
        or artifact.get("required") is not True
        or artifact.get("reuse") != "download"
    ):
        raise WorkflowAssetDownloadError(
            "binding_asset_changed", "the selected install artifact changed"
        )


def _safe_relative_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1_000
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise WorkflowAssetDownloadError(
            "invalid_install_artifact", "install artifact path is unsafe"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise WorkflowAssetDownloadError(
            "invalid_install_artifact", "install artifact path is unsafe"
        )
    return value
