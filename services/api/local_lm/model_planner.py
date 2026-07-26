from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .adapters.contracts import ADAPTER_CONTRACT_VERSION
from .comfy_templates import COMFY_TEMPLATE_COMPILER_VERSION
from .domain import new_id
from .model_manifests import ModelManifestInspection
from .models import InstallPlan

INSTALL_RESOLVER_VERSION = "install-resolver-v1"
ACTIVATION_PROBE_VERSION = "activation-probe-v2"
LAUNCH_CONTRACT_VERSION = "worker-launch-v1"

InstallCompatibility = Literal[
    "supported",
    "unsupported",
    "trusted_extension_required",
]


def media_workflow_contract_version(template_sha256: str) -> str:
    payload = f"{COMFY_TEMPLATE_COMPILER_VERSION}:{template_sha256}".encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PlannedArtifact:
    path: str
    kind: str
    target_folder: str
    size_bytes: int | None
    sha256: str | None
    required: bool = True
    reuse: str = "download"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "target_folder": self.target_folder,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "required": self.required,
            "reuse": self.reuse,
        }


@dataclass(frozen=True)
class ResolvedInstallPlan:
    provider: str
    remote_id: str
    revision: str
    role: str
    engine: str
    architecture: str | None
    family: str | None
    compatibility: InstallCompatibility
    artifacts: tuple[PlannedArtifact, ...]
    runtime_contract: dict[str, Any]
    activation_probe: dict[str, Any]
    failure_code: str | None = None
    failure_reason: str | None = None

    def blocked(self, code: str, reason: str) -> ResolvedInstallPlan:
        """Return the same immutable artifact plan with activation disabled."""

        return ResolvedInstallPlan(
            provider=self.provider,
            remote_id=self.remote_id,
            revision=self.revision,
            role=self.role,
            engine=self.engine,
            architecture=self.architecture,
            family=self.family,
            compatibility="unsupported",
            artifacts=self.artifacts,
            runtime_contract=self.runtime_contract,
            activation_probe={**self.activation_probe, "required": False},
            failure_code=code,
            failure_reason=reason,
        )

    @property
    def plan_hash(self) -> str:
        payload = {
            "provider": self.provider,
            "remote_id": self.remote_id,
            "revision": self.revision,
            "role": self.role,
            "engine": self.engine,
            "architecture": self.architecture,
            "family": self.family,
            "compatibility": self.compatibility,
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
            "runtime_contract": self.runtime_contract,
            "activation_probe": self.activation_probe,
            "resolver_version": INSTALL_RESOLVER_VERSION,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()


def resolve_install_plan(
    *,
    remote_id: str,
    revision: str,
    role: str,
    engine: str,
    selected_files: list[dict[str, Any]],
    inspection: ModelManifestInspection,
    workflow_template_id: str | None = None,
    workflow_template_sha256: str | None = None,
    comfy_paths: dict[str, str] | None = None,
    source_remote_id: str | None = None,
) -> ResolvedInstallPlan:
    metadata_by_path = {component.path: component for component in inspection.components}
    artifacts = tuple(
        PlannedArtifact(
            path=str(item["filename"]),
            kind=metadata_by_path[str(item["filename"])].kind,
            target_folder=metadata_by_path[str(item["filename"])].target_folder,
            size_bytes=(
                int(item["size"])
                if isinstance(item.get("size"), int) and not isinstance(item["size"], bool)
                else None
            ),
            sha256=(str(item["sha256"]).lower() if isinstance(item.get("sha256"), str) else None),
        )
        for item in selected_files
    )
    compatibility: InstallCompatibility = "supported"
    failure_code = None
    failure_reason = None
    if role == "chat":
        if engine != "llama.cpp" or not any(item.kind == "gguf_model" for item in artifacts):
            compatibility = "unsupported"
            failure_code = "unsupported_chat_layout"
            failure_reason = "Chat installation requires one complete GGUF model."
    elif engine != "comfyui":
        compatibility = "unsupported"
        failure_code = "unsupported_media_engine"
        failure_reason = "Image and video installation requires the managed ComfyUI runtime."
    elif any(
        not artifact.path.casefold().endswith((".safetensors", ".json")) for artifact in artifacts
    ):
        compatibility = "unsupported"
        failure_code = "unsafe_model_format"
        failure_reason = (
            "Automatic media installation accepts data-only safetensors and JSON metadata."
        )
    elif any(item.kind == "lora" for item in artifacts):
        compatibility = "unsupported"
        failure_code = "auxiliary_asset_not_primary"
        failure_reason = "This repository contains a LoRA, not a primary generation model."
    elif not workflow_template_id:
        compatibility = "trusted_extension_required"
        failure_code = "workflow_contract_missing"
        failure_reason = (
            "The model layout is safe to inspect, but no declarative workflow contract "
            "can activate it yet."
        )

    runtime_contract = {
        "engine": engine,
        "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
        "launch_contract_version": LAUNCH_CONTRACT_VERSION,
        "workflow_template_id": workflow_template_id,
        "workflow_template_sha256": workflow_template_sha256,
        "workflow_compiler_version": (
            COMFY_TEMPLATE_COMPILER_VERSION if workflow_template_id else None
        ),
        "comfy_paths": comfy_paths or {},
        "source_remote_id": source_remote_id,
    }
    activation_probe = {
        "version": ACTIVATION_PROBE_VERSION,
        "kind": "chat_completion" if role == "chat" else "media_output",
        "timeout_seconds": 120 if role == "chat" else 300,
        "required": compatibility == "supported",
    }
    return ResolvedInstallPlan(
        provider="huggingface",
        remote_id=remote_id,
        revision=revision,
        role=role,
        engine=engine,
        architecture=inspection.architecture,
        family=inspection.family,
        compatibility=compatibility,
        artifacts=artifacts,
        runtime_contract=runtime_contract,
        activation_probe=activation_probe,
        failure_code=failure_code,
        failure_reason=failure_reason,
    )


def persist_install_plan(session: Session, resolved: ResolvedInstallPlan) -> InstallPlan:
    existing = session.scalar(
        select(InstallPlan).where(InstallPlan.plan_hash == resolved.plan_hash)
    )
    if existing:
        if resolved.compatibility == "supported" and existing.status not in {
            "planned",
            "downloading",
        }:
            existing.status = "planned"
            existing.failure_code = None
            existing.failure_reason = None
        return existing
    plan = InstallPlan(
        id=new_id("plan"),
        provider=resolved.provider,
        remote_id=resolved.remote_id,
        revision=resolved.revision,
        role=resolved.role,
        engine=resolved.engine,
        architecture=resolved.architecture,
        family=resolved.family,
        plan_hash=resolved.plan_hash,
        resolver_version=INSTALL_RESOLVER_VERSION,
        compatibility=resolved.compatibility,
        artifacts_json=[artifact.as_dict() for artifact in resolved.artifacts],
        runtime_contract_json=resolved.runtime_contract,
        activation_probe_json=resolved.activation_probe,
        status="planned",
        failure_code=resolved.failure_code,
        failure_reason=resolved.failure_reason,
    )
    session.add(plan)
    session.flush()
    return plan
