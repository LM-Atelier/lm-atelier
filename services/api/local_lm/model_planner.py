from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .adapters.contracts import ADAPTER_CONTRACT_VERSION
from .auxiliary_assets import AUXILIARY_ASSET_KINDS
from .comfy_templates import COMFY_TEMPLATE_COMPILER_VERSION
from .domain import new_id
from .model_manifests import ModelManifestInspection, comfy_folder_for_kind
from .models import InstallPlan, ModelComponentManifest

INSTALL_RESOLVER_VERSION = "install-resolver-v9"
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
_WORKFLOW_REFERENCE_ARTIFACT_KINDS: dict[str, frozenset[str]] = {
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
_WORKFLOW_ARTIFACT_TARGET_FOLDERS: dict[str, frozenset[str]] = {
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

InstallCompatibility = Literal[
    "supported",
    "unsupported",
    "trusted_extension_required",
]


WORKFLOW_ARTIFACT_CONTRACT_VERSION = 1

# Declared dependencies that can change how a workflow executes.
#
# Template identity is deliberately absent. It is provenance, not execution: if
# the graph, schema and execution dependencies are identical, a template revision
# does not change what runs, and including it would recreate the compiler-version
# problem at template granularity. If a template change can alter execution
# without changing any of those, the missing execution dependency belongs in this
# payload rather than provenance standing in for it. Local install identifiers
# and the compiler version are excluded for the same reason.
_EXECUTION_DEPENDENCY_KEYS = (
    "model_files",
    "custom_nodes",
    # A Registry package changes what runs exactly as a git-installed node does,
    # so it belongs in the artifact's identity. Adding the key does not disturb
    # anything already stored: the payload is built by reading only the keys a
    # revision actually carries, and no existing revision carries this one.
    "registry_packages",
    "extensions",
)


def media_workflow_contract_version(
    template_sha256: str,
    compiler_version: int | None = None,
) -> str:
    """The legacy contract key, derived from the compiler version.

    Retained so evidence recorded before the artifact contract can still be
    recognised. Pass the compiler version the revision actually recorded: using
    the current one would accept evidence from a compiler that no longer exists.
    New evidence uses `workflow_artifact_contract`.
    """

    version = COMFY_TEMPLATE_COMPILER_VERSION if compiler_version is None else compiler_version
    payload = f"{version}:{template_sha256}".encode()
    return hashlib.sha256(payload).hexdigest()


def workflow_artifact_contract(
    *,
    operation: str,
    engine: str,
    api_graph: Mapping[str, Any],
    input_schema: Mapping[str, Any],
    dependencies: Mapping[str, Any],
) -> str:
    """Identify what a compiled workflow actually executes.

    Capability evidence used to be keyed on the compiler version, so improving
    the compiler invalidated every media model's evidence even when a given
    workflow compiled to exactly the same thing. Keying on the compiled output
    means only workflows whose execution really changed need re-proving.

    The payload is canonical: sorted keys, compact separators, no NaN. List order
    is preserved because it can be semantic. Anything incidental - the UI graph,
    timestamps, revision and install identifiers, display metadata - is excluded,
    so the same input always produces the same hash.
    """

    payload = {
        "version": WORKFLOW_ARTIFACT_CONTRACT_VERSION,
        "operation": operation,
        "engine": engine,
        "api_graph": api_graph,
        "input_schema": input_schema,
        "dependencies": {
            key: dependencies[key] for key in _EXECUTION_DEPENDENCY_KEYS if key in dependencies
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


MODEL_COMPONENTS_KEY = "model_components"


def declared_model_components(dependencies: Mapping[str, Any]) -> list[dict[str, str]]:
    """The content-addressed model requirement a revision declares, if any.

    `model_install_ids` names local rows, so it means nothing on another machine -
    an exported workflow arrives pointing at UUIDs that do not exist here. This
    says which *files* the workflow needs instead, by target folder and hash, so
    it can be satisfied by whichever local install happens to hold them.

    Deliberately not part of the artifact contract: what a revision executes is
    the graph, not which local rows currently satisfy it.
    """
    declared = dependencies.get(MODEL_COMPONENTS_KEY)
    if not isinstance(declared, list):
        return []
    pairs: set[tuple[str, str]] = set()
    for item in declared:
        if not isinstance(item, Mapping):
            continue
        folder = item.get("target_folder")
        digest = item.get("sha256")
        if not isinstance(folder, str) or not folder:
            continue
        if not isinstance(digest, str) or len(digest) != 64:
            continue
        pairs.add((folder, digest.lower()))
    # Sorted and de-duplicated, so the requirement is stable to store and compare.
    return [{"target_folder": folder, "sha256": digest} for folder, digest in sorted(pairs)]


def model_components_for_install(session: Session, install_id: str) -> list[dict[str, str]]:
    """The content identity of one install, from its component manifest."""
    rows = session.scalars(
        select(ModelComponentManifest).where(ModelComponentManifest.model_install_id == install_id)
    ).all()
    return [
        {"target_folder": row.target_folder, "sha256": row.sha256.lower()}
        for row in rows
        if row.required and row.sha256 and len(row.sha256) == 64
    ]


def install_satisfies_components(
    session: Session,
    install_id: str,
    components: list[dict[str, str]],
) -> bool:
    """Whether this install holds every declared component.

    An install with no recorded manifest satisfies nothing, which is why callers
    fall back to `model_install_ids` rather than treating an empty manifest as a
    match - a missing manifest means unknown, not compatible.
    """
    if not components:
        return False
    available = {
        (item["target_folder"], item["sha256"])
        for item in model_components_for_install(session, install_id)
    }
    if not available:
        return False
    return all((item["target_folder"], item["sha256"]) in available for item in components)


def revision_accepts_install(
    session: Session,
    dependencies: Mapping[str, Any],
    model_install_id: str | None,
) -> bool:
    """Whether a revision's declared model binding admits this install.

    Content first. `model_install_ids` names local rows, so an exported or
    imported workflow arrives pointing at identifiers that do not exist on this
    machine; its component hashes still resolve against whichever local install
    holds the same files. The id list stays as the fallback, so a revision
    recorded before content binding - or an install whose component manifest was
    never written - behaves exactly as it did before.

    One implementation, because selection, setup readiness and pin validation all
    have to answer this the same way. They previously each spelled it out.
    """
    components = declared_model_components(dependencies)
    if (
        components
        and model_install_id
        and install_satisfies_components(session, model_install_id, components)
    ):
        return True
    declared = dependencies.get("model_install_ids")
    declared_installs = {str(item) for item in declared} if isinstance(declared, list) else set()
    return not declared_installs or model_install_id in declared_installs


def revision_declares_a_model(dependencies: Mapping[str, Any]) -> bool:
    """Whether a revision is bound to particular models at all."""
    if declared_model_components(dependencies):
        return True
    declared = dependencies.get("model_install_ids")
    return isinstance(declared, list) and bool(declared)


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
    # CivitAI identity: the exact version and file this artifact came from.
    # The download manager derives its URL from these server-side and never
    # consumes a catalog-supplied one, so they are part of the plan hash.
    source_version_id: str | None = None
    source_file_id: str | None = None

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
            "source_version_id": self.source_version_id,
            "source_file_id": self.source_file_id,
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


def _declared_trigger_words(selected_files: list[dict[str, Any]]) -> list[str]:
    # Provider metadata is external input. Freeze one bounded, canonical list
    # into the immutable plan so installation never trusts a later response.
    words: list[str] = []
    seen: set[str] = set()
    for item in selected_files:
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        for key_name in ("trained_words", "trigger_words"):
            declared = metadata.get(key_name)
            if not isinstance(declared, list):
                continue
            for value in declared:
                if not isinstance(value, str):
                    continue
                word = value.strip()[:200]
                key = word.casefold()
                if not word or key in seen:
                    continue
                seen.add(key)
                words.append(word)
                if len(words) == 100:
                    return words
    return words


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
    workflow_component_folders: dict[str, str] | None = None,
    source_remote_id: str | None = None,
    auxiliary_kind: str | None = None,
    workflow_reference_kind: str | None = None,
    provider: str = "huggingface",
) -> ResolvedInstallPlan:
    auxiliary_folder = None
    if auxiliary_kind:
        if auxiliary_kind not in AUXILIARY_ASSET_KINDS:
            raise ValueError("unsupported auxiliary asset kind")
        auxiliary_folder = comfy_folder_for_kind(auxiliary_kind)
        if auxiliary_folder is None:
            raise ValueError("auxiliary asset has no ComfyUI model folder")
    metadata_by_path = {component.path: component for component in inspection.components}
    workflow_contracts = {
        path: contract
        for item in selected_files
        if workflow_template_id
        and (
            contract := _workflow_component_contract(
                path := str(item["filename"]),
                comfy_paths or {},
                workflow_component_folders or {},
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
                auxiliary_folder
                if auxiliary_folder
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
            source_version_id=(
                str(item["source_version_id"]) if item.get("source_version_id") else None
            ),
            source_file_id=(str(item["source_file_id"]) if item.get("source_file_id") else None),
        )
        for item in selected_files
    )
    compatibility: InstallCompatibility = "supported"
    failure_code = None
    failure_reason = None
    if auxiliary_kind and workflow_reference_kind:
        compatibility = "unsupported"
        failure_code = "conflicting_asset_ownership"
        failure_reason = "A model asset cannot be both standalone and workflow-owned."
    elif workflow_reference_kind:
        workflow_failure = _workflow_asset_failure(
            artifacts,
            selected_files=selected_files,
            reference_kind=workflow_reference_kind,
            role=role,
            engine=engine,
            workflow_template_id=workflow_template_id,
        )
        if workflow_failure:
            compatibility = "unsupported"
            failure_code, failure_reason = workflow_failure
    elif auxiliary_kind:
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
        "trigger_words": _declared_trigger_words(selected_files),
        "engine": engine,
        "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
        "launch_contract_version": LAUNCH_CONTRACT_VERSION,
        "workflow_template_id": workflow_template_id,
        "workflow_template_sha256": workflow_template_sha256,
        "workflow_compiler_version": (
            COMFY_TEMPLATE_COMPILER_VERSION if workflow_template_id else None
        ),
        "comfy_paths": comfy_paths or {},
        "workflow_component_folders": workflow_component_folders or {},
        "source_remote_id": source_remote_id,
        "auxiliary_kind": auxiliary_kind,
        "quantization": "modelopt" if engine == "vllm" else None,
        "model_layout": "transformers_snapshot" if engine == "vllm" else None,
    }
    if workflow_reference_kind:
        runtime_contract["workflow_reference_kind"] = workflow_reference_kind
        if len(artifacts) == 1:
            workflow_artifact = artifacts[0]
            runtime_contract.update(
                {
                    "comfy_paths": {workflow_artifact.target_folder: "."},
                    "workflow_component_folders": {
                        workflow_artifact.path: workflow_artifact.target_folder
                    },
                    "workflow_asset_kind": workflow_artifact.kind,
                }
            )

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
    if workflow_reference_kind:
        activation_probe.update({"kind": "workflow_asset", "required": False})

    return ResolvedInstallPlan(
        provider=provider,
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


def _workflow_asset_failure(
    artifacts: tuple[PlannedArtifact, ...],
    *,
    selected_files: list[dict[str, Any]],
    reference_kind: str,
    role: str,
    engine: str,
    workflow_template_id: str | None,
) -> tuple[str, str] | None:
    """Validate one exact provider component as an inert workflow dependency."""

    if engine != "comfyui" or role not in {"image", "video"}:
        return (
            "unsupported_workflow_asset_runtime",
            "Workflow-owned model assets require a ComfyUI media plan.",
        )
    if workflow_template_id:
        return (
            "conflicting_workflow_contract",
            "A workflow-owned asset cannot also select a standalone workflow template.",
        )
    if len(artifacts) != 1:
        return (
            "workflow_asset_selection_required",
            "Choose one exact file for each workflow asset.",
        )
    artifact = artifacts[0]
    admitted_kinds = _WORKFLOW_REFERENCE_ARTIFACT_KINDS.get(reference_kind)
    if admitted_kinds is None or artifact.kind not in admitted_kinds:
        return (
            "workflow_asset_kind_mismatch",
            "The selected file kind does not match the workflow reference.",
        )
    if artifact.target_folder not in _WORKFLOW_ARTIFACT_TARGET_FOLDERS.get(
        artifact.kind, frozenset()
    ):
        return (
            "workflow_asset_folder_mismatch",
            "The selected file has no valid ComfyUI target folder.",
        )
    allowed_suffixes = (".gguf",) if artifact.kind == "gguf_model" else (".safetensors",)
    if not artifact.path.casefold().endswith(allowed_suffixes):
        return (
            "unsafe_workflow_asset_format",
            "Workflow-owned model assets must use a data-only weights format.",
        )
    digest = selected_files[0].get("sha256")
    if (
        not isinstance(artifact.size_bytes, int)
        or isinstance(artifact.size_bytes, bool)
        or artifact.size_bytes <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        return (
            "unverified_workflow_asset",
            "Workflow-owned model assets need an exact positive size and SHA-256.",
        )
    return None


def _workflow_component_contract(
    path: str,
    comfy_paths: dict[str, str],
    component_folders: dict[str, str],
) -> tuple[str, str] | None:
    declared_folder = component_folders.get(path)
    if declared_folder in _WORKFLOW_COMPONENT_KINDS:
        return _WORKFLOW_COMPONENT_KINDS[declared_folder], declared_folder
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
        # A plan already being fetched is left exactly as it is: the transfer
        # reads these fields, and rewriting them underneath it would change
        # what is being downloaded while it is being downloaded.
        if existing.status == "downloading":
            return existing
        # Otherwise the fresh resolution is the truth, including its failure.
        #
        # Plans are reused by hash and the failure fields are not part of that
        # hash, so two resolves of one install can disagree about why it failed.
        # A row sitting in "planned" was never rewritten, so the first reason
        # recorded was the reason reported from then on - which is how a
        # malformed request's error came back, unchanged, to well-formed
        # requests made afterwards, about an attempt nobody remembered making.
        existing.status = "planned"
        existing.compatibility = resolved.compatibility
        existing.failure_code = resolved.failure_code
        existing.failure_reason = resolved.failure_reason
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
