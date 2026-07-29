from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .adapters.contracts import ADAPTER_CONTRACT_VERSION
from .comfy_templates import COMFY_TEMPLATE_COMPILER_VERSION
from .domain import new_id
from .model_manifests import ModelManifestInspection
from .models import InstallPlan

INSTALL_RESOLVER_VERSION = "install-resolver-v5"
ACTIVATION_PROBE_VERSION = "activation-probe-v2"
LAUNCH_CONTRACT_VERSION = "worker-launch-v1"

_WORKFLOW_COMPONENT_KINDS = {
    "checkpoints": "checkpoint",
    "diffusion_models": "diffusion_model",
    "unet": "diffusion_model",
    "text_encoders": "text_encoder",
    "vae": "vae",
    "clip_vision": "clip_vision",
    "loras": "lora",
    "controlnet": "controlnet",
    "upscale_models": "upscaler",
    "embeddings": "embedding",
    "ipadapter": "ip_adapter",
}
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
    source_remote_id: str | None = None
    source_revision: str | None = None
    source_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "target_folder": self.target_folder,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "required": self.required,
            "reuse": self.reuse,
            "source_remote_id": self.source_remote_id,
            "source_revision": self.source_revision,
            "source_path": self.source_path,
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
    auxiliary_kind: str | None = None,
) -> ResolvedInstallPlan:
    metadata_by_path = {component.path: component for component in inspection.components}
    workflow_contracts = {
        path: contract
        for item in selected_files
        if workflow_template_id
        and (
            contract := _workflow_component_contract(
                path := str(item["filename"]),
                comfy_paths or {},
            )
        )
    }
    artifacts = tuple(
        PlannedArtifact(
            path=str(item["filename"]),
            kind=(
                auxiliary_kind
                or (
                    "weights"
                    if engine == "vllm"
                    and str(item["filename"]).casefold().endswith(".safetensors")
                    else workflow_contracts.get(
                        str(item["filename"]),
                        (
                            metadata_by_path[str(item["filename"])].kind,
                            metadata_by_path[str(item["filename"])].target_folder,
                        ),
                    )[0]
                )
            ),
            target_folder=(
                {
                    "lora": "loras",
                    "vae": "vae",
                    "controlnet": "controlnet",
                    "upscaler": "upscale_models",
                    "embedding": "embeddings",
                    "ip_adapter": "ipadapter",
                }[auxiliary_kind]
                if auxiliary_kind
                else workflow_contracts.get(
                    str(item["filename"]),
                    (
                        metadata_by_path[str(item["filename"])].kind,
                        metadata_by_path[str(item["filename"])].target_folder,
                    ),
                )[1]
            ),
            size_bytes=(
                int(item["size"])
                if isinstance(item.get("size"), int) and not isinstance(item["size"], bool)
                else None
            ),
            sha256=(str(item["sha256"]).lower() if isinstance(item.get("sha256"), str) else None),
            source_remote_id=(
                str(item["source_remote_id"]) if item.get("source_remote_id") else None
            ),
            source_revision=(str(item["source_revision"]) if item.get("source_revision") else None),
            source_path=(str(item["source_filename"]) if item.get("source_filename") else None),
        )
        for item in selected_files
    )
    compatibility: InstallCompatibility = "supported"
    failure_code = None
    failure_reason = None
    if auxiliary_kind:
        inspected_kinds = {metadata_by_path[str(item["filename"])].kind for item in selected_files}
        if engine != "comfyui":
            compatibility = "unsupported"
            failure_code = "unsupported_auxiliary_engine"
            failure_reason = "Auxiliary generation assets require ComfyUI."
        elif auxiliary_kind != "lora":
            compatibility = "unsupported"
            failure_code = "auxiliary_kind_not_implemented"
            failure_reason = (
                f"{auxiliary_kind.replace('_', ' ').title()} installation is reserved "
                "but is not enabled in this release."
            )
        elif inspected_kinds != {"lora"}:
            compatibility = "unsupported"
            failure_code = "auxiliary_kind_mismatch"
            failure_reason = "The selected safetensors file is not a verified LoRA."
    elif role == "chat" and engine == "llama.cpp":
        if not any(item.kind == "gguf_model" for item in artifacts):
            compatibility = "unsupported"
            failure_code = "unsupported_chat_layout"
            failure_reason = "Chat installation requires one complete GGUF model."
    elif role == "chat" and engine == "vllm":
        artifact_names = {artifact.path.rsplit("/", 1)[-1].casefold() for artifact in artifacts}
        if (
            not any(artifact.kind == "weights" for artifact in artifacts)
            or "config.json" not in artifact_names
            or "hf_quant_config.json" not in artifact_names
        ):
            compatibility = "unsupported"
            failure_code = "incomplete_modelopt_snapshot"
            failure_reason = (
                "ModelOpt installation requires complete safetensors weights, config.json, "
                "and hf_quant_config.json."
            )
    elif role == "chat":
        compatibility = "unsupported"
        failure_code = "unsupported_chat_engine"
        failure_reason = "Chat installation requires the managed llama.cpp or vLLM runtime."
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
    elif artifacts and all(item.kind == "lora" for item in artifacts):
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
        "auxiliary_kind": auxiliary_kind,
        "quantization": "modelopt" if engine == "vllm" else None,
        "model_layout": "transformers_snapshot" if engine == "vllm" else None,
    }
    activation_probe = {
        "version": ACTIVATION_PROBE_VERSION,
        "kind": (
            "auxiliary_graph"
            if auxiliary_kind
            else "chat_completion"
            if role == "chat"
            else "media_output"
        ),
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


def _workflow_component_contract(
    path: str,
    comfy_paths: dict[str, str],
) -> tuple[str, str] | None:
    parent = str(PurePosixPath(path).parent)
    parent = "." if parent == "." else parent
    folders = [
        folder
        for folder, declared_parent in comfy_paths.items()
        if declared_parent == parent and folder in _WORKFLOW_COMPONENT_KINDS
    ]
    if len(folders) != 1:
        return None
    folder = folders[0]
    return _WORKFLOW_COMPONENT_KINDS[folder], folder


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
