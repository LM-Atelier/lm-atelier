from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .comfy_workflow_packages import WorkflowAssetReference
from .models import InstallPlan

WORKFLOW_ASSET_BINDING_VERSION = 1
MAX_WORKFLOW_ASSET_BINDINGS = 512

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64}|[1-9][0-9]{0,19})$")
_PLAN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")

_REFERENCE_ARTIFACT_KINDS: dict[str, frozenset[str]] = {
    "checkpoint": frozenset(
        {
            "checkpoint",
            "clip_vision",
            "controlnet",
            "diffusion_model",
            "gguf_model",
            "ip_adapter",
            "text_encoder",
        }
    ),
    "embedding": frozenset({"embedding"}),
    "lora": frozenset({"lora"}),
    "upscaler": frozenset({"upscaler"}),
    "vae": frozenset({"vae"}),
}

_ARTIFACT_TARGET_FOLDERS: dict[str, frozenset[str]] = {
    "checkpoint": frozenset({"checkpoints"}),
    "clip_vision": frozenset({"clip_vision"}),
    "controlnet": frozenset({"controlnet"}),
    "diffusion_model": frozenset({"diffusion_models", "unet"}),
    "embedding": frozenset({"embeddings"}),
    "gguf_model": frozenset(
        {"checkpoints", "clip_vision", "diffusion_models", "models", "text_encoders", "unet"}
    ),
    "ip_adapter": frozenset({"ipadapter"}),
    "lora": frozenset({"loras"}),
    "text_encoder": frozenset({"text_encoders"}),
    "upscaler": frozenset({"upscale_models"}),
    "vae": frozenset({"vae"}),
}


class WorkflowAssetBindingError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WorkflowAssetPlanSelection:
    reference_filename: str
    install_plan_id: str
    artifact_path: str


@dataclass(frozen=True)
class BoundWorkflowAsset:
    reference_filename: str
    kind: str
    install_plan_id: str
    install_plan_hash: str
    provider: str
    remote_id: str
    revision: str
    artifact_path: str
    artifact_kind: str
    target_folder: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "reference_filename": self.reference_filename,
            "kind": self.kind,
            "install_plan_id": self.install_plan_id,
            "install_plan_hash": self.install_plan_hash,
            "provider": self.provider,
            "remote_id": self.remote_id,
            "revision": self.revision,
            "artifact_path": self.artifact_path,
            "artifact_kind": self.artifact_kind,
            "target_folder": self.target_folder,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class WorkflowAssetBindingPlan:
    assets: tuple[BoundWorkflowAsset, ...]

    @property
    def plan_hash(self) -> str:
        payload = {
            "version": WORKFLOW_ASSET_BINDING_VERSION,
            "assets": [asset.as_dict() for asset in self.assets],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()


def bind_workflow_assets_to_install_plans(
    references: Sequence[WorkflowAssetReference],
    selections: Sequence[WorkflowAssetPlanSelection],
    install_plans: Mapping[str, InstallPlan],
) -> WorkflowAssetBindingPlan:
    """Bind missing workflow filenames to explicit immutable install artifacts."""

    if (
        len(references) > MAX_WORKFLOW_ASSET_BINDINGS
        or len(selections) > MAX_WORKFLOW_ASSET_BINDINGS
    ):
        raise WorkflowAssetBindingError(
            "too_many_asset_bindings", "workflow has too many asset bindings"
        )

    references_by_name: dict[str, WorkflowAssetReference] = {}
    missing_names: list[str] = []
    for reference in references:
        normalized = _validated_path(reference.filename, code="invalid_asset_reference")
        if (
            not isinstance(reference.suffix, str)
            or PurePosixPath(normalized).suffix.casefold() != reference.suffix.casefold()
        ):
            raise WorkflowAssetBindingError(
                "invalid_asset_reference", "workflow asset suffix does not match its filename"
            )
        folded = normalized.casefold()
        if folded in references_by_name:
            raise WorkflowAssetBindingError(
                "duplicate_asset_reference", "workflow repeats an asset reference"
            )
        references_by_name[folded] = reference
        if reference.policy != "supported":
            raise WorkflowAssetBindingError(
                "unsupported_asset_reference",
                "workflow includes an asset format that cannot be installed",
            )
        if not reference.present_locally:
            missing_names.append(folded)

    selections_by_name: dict[str, WorkflowAssetPlanSelection] = {}
    for selection in selections:
        normalized = _validated_path(selection.reference_filename, code="invalid_asset_selection")
        folded = normalized.casefold()
        if folded in selections_by_name:
            raise WorkflowAssetBindingError(
                "duplicate_asset_selection", "an asset has more than one install selection"
            )
        selections_by_name[folded] = selection

    missing = set(missing_names)
    selected = set(selections_by_name)
    if missing != selected:
        if missing - selected:
            raise WorkflowAssetBindingError(
                "missing_asset_selection", "every missing workflow asset needs an install selection"
            )
        raise WorkflowAssetBindingError(
            "unexpected_asset_selection",
            "an install selection does not belong to a missing workflow asset",
        )

    bound: list[BoundWorkflowAsset] = []
    used_artifacts: set[tuple[str, str]] = set()
    for folded in missing_names:
        reference = references_by_name[folded]
        selection = selections_by_name[folded]
        if selection.reference_filename != reference.filename:
            raise WorkflowAssetBindingError(
                "asset_reference_case_mismatch",
                "asset selection must preserve the workflow filename exactly",
            )
        plan = install_plans.get(selection.install_plan_id)
        if plan is None:
            raise WorkflowAssetBindingError(
                "install_plan_not_found", "selected install plan does not exist"
            )
        _validate_plan(plan, selection.install_plan_id)
        artifact_path = _validated_path(selection.artifact_path, code="invalid_artifact_path")
        matches = [
            artifact
            for artifact in plan.artifacts_json
            if isinstance(artifact, Mapping) and artifact.get("path") == artifact_path
        ]
        if len(matches) != 1:
            code = "artifact_not_found" if not matches else "ambiguous_plan_artifact"
            raise WorkflowAssetBindingError(code, "install plan does not name one exact artifact")
        artifact = matches[0]
        if artifact_path != reference.filename:
            raise WorkflowAssetBindingError(
                "artifact_path_mismatch",
                "install artifact must preserve the workflow filename exactly",
            )
        key = (plan.id, artifact_path)
        if key in used_artifacts:
            raise WorkflowAssetBindingError(
                "duplicate_artifact_binding", "one install artifact cannot satisfy two references"
            )
        used_artifacts.add(key)
        bound.append(_bound_asset(reference, plan, artifact))

    return WorkflowAssetBindingPlan(tuple(bound))


def _validate_plan(plan: InstallPlan, expected_id: str) -> None:
    if not _PLAN_ID.fullmatch(expected_id) or plan.id != expected_id:
        raise WorkflowAssetBindingError(
            "install_plan_identity_mismatch", "install plan identity changed"
        )
    if plan.status != "planned":
        raise WorkflowAssetBindingError(
            "install_plan_not_pending", "install plan is not ready to be queued"
        )
    if plan.compatibility != "supported" or plan.failure_code:
        raise WorkflowAssetBindingError(
            "install_plan_not_supported", "install plan is not supported"
        )
    if (
        not isinstance(plan.artifacts_json, list)
        or len(plan.artifacts_json) > MAX_WORKFLOW_ASSET_BINDINGS
    ):
        raise WorkflowAssetBindingError(
            "invalid_install_plan", "install plan has no bounded artifact list"
        )
    if (
        not isinstance(plan.provider, str)
        or not plan.provider
        or not isinstance(plan.remote_id, str)
        or not plan.remote_id
        or not isinstance(plan.revision, str)
        or not _IMMUTABLE_REVISION.fullmatch(plan.revision)
        or not isinstance(plan.plan_hash, str)
        or not _DIGEST.fullmatch(plan.plan_hash)
    ):
        raise WorkflowAssetBindingError(
            "mutable_install_plan", "install plan lacks immutable source identity"
        )


def _bound_asset(
    reference: WorkflowAssetReference,
    plan: InstallPlan,
    artifact: Mapping[str, Any],
) -> BoundWorkflowAsset:
    if artifact.get("required") is not True or artifact.get("reuse") != "download":
        raise WorkflowAssetBindingError(
            "artifact_not_downloadable", "selected artifact is not a required download"
        )
    artifact_kind = artifact.get("kind")
    target_folder = artifact.get("target_folder")
    expected_kinds = _REFERENCE_ARTIFACT_KINDS.get(reference.kind)
    if (
        expected_kinds is None
        or not isinstance(artifact_kind, str)
        or artifact_kind not in expected_kinds
    ):
        raise WorkflowAssetBindingError(
            "artifact_kind_mismatch", "install artifact kind does not match the workflow reference"
        )
    if not isinstance(target_folder, str) or target_folder not in _ARTIFACT_TARGET_FOLDERS.get(
        artifact_kind, frozenset()
    ):
        raise WorkflowAssetBindingError(
            "artifact_folder_mismatch", "install artifact has no valid runtime target folder"
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
        raise WorkflowAssetBindingError(
            "unverified_plan_artifact", "install artifact needs exact size and SHA-256"
        )
    return BoundWorkflowAsset(
        reference_filename=reference.filename,
        kind=reference.kind,
        install_plan_id=plan.id,
        install_plan_hash=plan.plan_hash,
        provider=plan.provider,
        remote_id=plan.remote_id,
        revision=plan.revision,
        artifact_path=str(artifact["path"]),
        artifact_kind=artifact_kind,
        target_folder=target_folder,
        size_bytes=size,
        sha256=digest,
    )


def _validated_path(value: object, *, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1_000
        or chr(92) in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise WorkflowAssetBindingError(code, "asset path is not a safe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise WorkflowAssetBindingError(code, "asset path is not a safe relative path")
    return value
