from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, cast
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from .auxiliary_assets import AUXILIARY_ASSET_KINDS
from .model_manifests import COMFY_MODEL_FOLDERS
from .models import (
    ComfyRegistryInstall,
    CustomNodeInstall,
    ModelAssetInstall,
    ModelComponentManifest,
    ModelInstall,
    ModelProfile,
)
from .profile_service import validate_profile_binding
from .workflow_dependencies import (
    UNORDERED_WORKFLOW_CONSTRAINT_ARRAYS,
    WORKFLOW_DEPENDENCY_RESOURCE_KINDS,
    WorkflowDependencyContract,
    WorkflowDependencyError,
    WorkflowDependencyRequirement,
    WorkflowDependencyResourceKind,
    canonical_workflow_dependency_json,
    canonicalize_workflow_dependency_collections,
    parse_workflow_dependency_contract,
    validate_portable_workflow_mapping,
    workflow_dependency_contract_payload,
    workflow_dependency_contract_sha256,
)

WORKFLOW_BINDING_VERSION = 1
MAX_WORKFLOW_BINDING_SELECTIONS = 512

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_INPUT = re.compile(r"^[0-9a-fA-F]{64}$")
_TREE_HASH = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_STABLE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,199}$")
_PACKAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_LOWERCASE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SEMANTIC_VERSION = re.compile(r"^[0-9]+[.][0-9]+[.][0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_REGISTRY_ENVIRONMENT_PATH = re.compile(r"^registry-wheels-([0-9a-f]{64})$")


class WorkflowBindingError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WorkflowBindingSelection:
    slot_name: str
    requirement_key: str
    local_kind: str
    local_id: str
    recorded_resource_identity_sha256: str | None = None
    mount: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializedWorkflowDependency:
    resource_kind: WorkflowDependencyResourceKind
    identity: dict[str, Any]


@dataclass(frozen=True)
class ResolvedWorkflowBinding:
    slot_name: str
    requirement_key: str
    resource_kind: WorkflowDependencyResourceKind
    identity: dict[str, Any]
    resource_identity_sha256: str
    mount: dict[str, Any]


@dataclass(frozen=True)
class WorkflowBindingIssue:
    code: str
    slot_name: str
    requirement_key: str | None = None


@dataclass(frozen=True)
class WorkflowActivationResolution:
    bindings: tuple[ResolvedWorkflowBinding, ...]
    issues: tuple[WorkflowBindingIssue, ...]
    missing_required_slots: tuple[str, ...]
    complete: bool
    binding_sha256: str | None


WorkflowDependencyMaterializer = Callable[
    [WorkflowDependencyRequirement, WorkflowBindingSelection],
    MaterializedWorkflowDependency | None,
]


def resolve_workflow_activation(
    contract: WorkflowDependencyContract,
    selections: Sequence[WorkflowBindingSelection],
    materialize: WorkflowDependencyMaterializer,
) -> WorkflowActivationResolution:
    """Resolve local selections into one portable, complete activation snapshot."""

    contract = _canonical_contract(contract)
    if isinstance(selections, str | bytes) or not isinstance(selections, Sequence):
        raise WorkflowBindingError(
            "invalid_dependency_bindings", "Workflow dependency selections must be an array"
        )
    if len(selections) > MAX_WORKFLOW_BINDING_SELECTIONS:
        raise WorkflowBindingError(
            "too_many_dependency_bindings", "Workflow has too many dependency selections"
        )
    slots = {slot.name: slot for slot in contract.slots}
    grouped: dict[str, list[WorkflowBindingSelection]] = {name: [] for name in slots}
    seen: set[tuple[str, str]] = set()
    mounts: dict[tuple[str, str], dict[str, Any]] = {}
    for selection in selections:
        if not isinstance(selection, WorkflowBindingSelection):
            raise WorkflowBindingError(
                "invalid_dependency_bindings", "Workflow dependency selection is invalid"
            )
        slot = slots.get(selection.slot_name)
        if slot is None:
            raise WorkflowBindingError(
                "unknown_dependency_slot",
                f"Workflow dependency slot {selection.slot_name} is not declared",
            )
        requirements = {item.key: item for item in slot.requirements}
        if selection.requirement_key not in requirements:
            raise WorkflowBindingError(
                "unknown_dependency_requirement",
                f"Workflow dependency requirement {selection.slot_name}."
                f"{selection.requirement_key} is not declared",
            )
        pair = (selection.slot_name, selection.requirement_key)
        if pair in seen:
            raise WorkflowBindingError(
                "duplicate_dependency_binding",
                f"Workflow dependency {selection.slot_name}.{selection.requirement_key} "
                "is selected more than once",
            )
        seen.add(pair)
        if selection.local_kind != slot.resource_kind:
            raise WorkflowBindingError(
                "dependency_locator_kind_mismatch",
                f"Workflow dependency {selection.slot_name} requires {slot.resource_kind}",
            )
        _local_locator(selection.local_id)
        if selection.recorded_resource_identity_sha256 is not None and not _DIGEST.fullmatch(
            selection.recorded_resource_identity_sha256
        ):
            raise WorkflowBindingError(
                "invalid_dependency_binding_digest",
                "Recorded dependency identity digest is invalid",
            )
        mounts[pair] = _portable_binding_mapping(
            selection.mount,
            label=f"workflow dependency mount {selection.slot_name}.{selection.requirement_key}",
            error_code="invalid_dependency_bindings",
        )
        grouped[selection.slot_name].append(selection)

    for slot in contract.slots:
        if slot.satisfaction == "any_of" and len(grouped[slot.name]) > 1:
            raise WorkflowBindingError(
                "ambiguous_dependency_binding",
                f"Workflow dependency slot {slot.name} accepts only one alternative",
            )

    resolved: list[ResolvedWorkflowBinding] = []
    issues: list[WorkflowBindingIssue] = []
    valid_pairs: set[tuple[str, str]] = set()
    for slot in contract.slots:
        requirements = {item.key: item for item in slot.requirements}
        for selection in sorted(grouped[slot.name], key=lambda item: item.requirement_key):
            requirement = requirements[selection.requirement_key]
            try:
                materialized = materialize(requirement, selection)
            except WorkflowBindingError as exc:
                if exc.code not in {"dependency_unavailable", "invalid_dependency_identity"}:
                    raise
                issues.append(WorkflowBindingIssue(exc.code, slot.name, selection.requirement_key))
                continue
            if materialized is None:
                issues.append(
                    WorkflowBindingIssue(
                        "dependency_unavailable", slot.name, selection.requirement_key
                    )
                )
                continue
            if materialized.resource_kind != slot.resource_kind:
                issues.append(
                    WorkflowBindingIssue(
                        "dependency_requirement_mismatch", slot.name, selection.requirement_key
                    )
                )
                continue
            try:
                identity = _portable_identity_mapping(
                    materialized.identity,
                    label=f"workflow dependency identity {slot.name}.{selection.requirement_key}",
                )
            except WorkflowBindingError as exc:
                if exc.code != "invalid_dependency_identity":
                    raise
                issues.append(WorkflowBindingIssue(exc.code, slot.name, selection.requirement_key))
                continue
            if identity.get("kind") != slot.resource_kind or not _matches_constraints(
                requirement.constraints, identity
            ):
                issues.append(
                    WorkflowBindingIssue(
                        "dependency_requirement_mismatch", slot.name, selection.requirement_key
                    )
                )
                continue
            identity_sha256 = workflow_resource_identity_sha256(slot.resource_kind, identity)
            if (
                selection.recorded_resource_identity_sha256 is not None
                and selection.recorded_resource_identity_sha256 != identity_sha256
            ):
                issues.append(
                    WorkflowBindingIssue(
                        "dependency_binding_drift", slot.name, selection.requirement_key
                    )
                )
                continue
            valid_pairs.add((slot.name, selection.requirement_key))
            resolved.append(
                ResolvedWorkflowBinding(
                    slot.name,
                    selection.requirement_key,
                    slot.resource_kind,
                    identity,
                    identity_sha256,
                    mounts[(slot.name, selection.requirement_key)],
                )
            )

    missing_required: set[str] = set()
    for slot in contract.slots:
        selected_keys = {item.requirement_key for item in grouped[slot.name]}
        valid_keys = {
            requirement_key for slot_name, requirement_key in valid_pairs if slot_name == slot.name
        }
        declared_keys = {item.key for item in slot.requirements}
        if slot.satisfaction == "any_of":
            satisfied = len(valid_keys) == 1
            if slot.required and not satisfied:
                missing_required.add(slot.name)
                issues.append(WorkflowBindingIssue("missing_required_dependency", slot.name))
            continue
        if not selected_keys and not slot.required:
            continue
        if valid_keys != declared_keys:
            if slot.required:
                missing_required.add(slot.name)
                issues.append(WorkflowBindingIssue("missing_required_dependency", slot.name))
            elif selected_keys:
                issues.append(WorkflowBindingIssue("partial_optional_dependency", slot.name))

    resolved.sort(key=lambda item: (item.slot_name, item.requirement_key))
    issues.sort(key=lambda item: (item.slot_name, item.requirement_key or "", item.code))
    complete = not issues and not missing_required
    binding_sha256 = _workflow_activation_binding_sha256(contract, resolved) if complete else None
    return WorkflowActivationResolution(
        tuple(resolved),
        tuple(issues),
        tuple(sorted(missing_required)),
        complete,
        binding_sha256,
    )


def workflow_resource_identity_sha256(
    resource_kind: WorkflowDependencyResourceKind,
    identity: dict[str, Any],
) -> str:
    portable = _portable_identity_mapping(
        identity,
        label="workflow resource identity",
    )
    if (
        resource_kind not in WORKFLOW_DEPENDENCY_RESOURCE_KINDS
        or portable.get("kind") != resource_kind
    ):
        raise WorkflowBindingError(
            "dependency_identity_kind_mismatch",
            "Workflow resource identity kind does not match its dependency kind",
        )
    payload = {
        "version": WORKFLOW_BINDING_VERSION,
        "resource_kind": resource_kind,
        "identity": portable,
    }
    return hashlib.sha256(canonical_workflow_dependency_json(payload)).hexdigest()


def _workflow_activation_binding_sha256(
    contract: WorkflowDependencyContract,
    bindings: Sequence[ResolvedWorkflowBinding],
) -> str:
    contract = _canonical_contract(contract)
    if isinstance(bindings, str | bytes) or not isinstance(bindings, Sequence):
        raise WorkflowBindingError(
            "invalid_dependency_bindings", "Resolved workflow bindings must be an array"
        )
    if len(bindings) > MAX_WORKFLOW_BINDING_SELECTIONS:
        raise WorkflowBindingError(
            "too_many_dependency_bindings", "Workflow has too many resolved bindings"
        )

    slots = {slot.name: slot for slot in contract.slots}
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, set[str]] = {slot.name: set() for slot in contract.slots}
    canonical_bindings: list[dict[str, object]] = []
    for binding in bindings:
        if not isinstance(binding, ResolvedWorkflowBinding):
            raise WorkflowBindingError(
                "invalid_dependency_bindings", "Resolved workflow binding is invalid"
            )
        slot = slots.get(binding.slot_name)
        if slot is None:
            raise WorkflowBindingError(
                "unknown_dependency_slot",
                f"Workflow dependency slot {binding.slot_name} is not declared",
            )
        requirements = {item.key: item for item in slot.requirements}
        requirement = requirements.get(binding.requirement_key)
        if requirement is None:
            raise WorkflowBindingError(
                "unknown_dependency_requirement",
                f"Workflow dependency requirement {binding.slot_name}."
                f"{binding.requirement_key} is not declared",
            )
        pair = (binding.slot_name, binding.requirement_key)
        if pair in seen:
            raise WorkflowBindingError(
                "duplicate_dependency_binding",
                f"Workflow dependency {binding.slot_name}.{binding.requirement_key} "
                "is bound more than once",
            )
        seen.add(pair)
        if binding.resource_kind != slot.resource_kind:
            raise WorkflowBindingError(
                "dependency_locator_kind_mismatch",
                f"Workflow dependency {binding.slot_name} requires {slot.resource_kind}",
            )
        identity = _portable_identity_mapping(
            binding.identity,
            label=f"workflow dependency identity {binding.slot_name}.{binding.requirement_key}",
        )
        if identity.get("kind") != slot.resource_kind or not _matches_constraints(
            requirement.constraints, identity
        ):
            raise WorkflowBindingError(
                "dependency_requirement_mismatch",
                f"Workflow dependency {binding.slot_name}.{binding.requirement_key} "
                "does not satisfy its contract",
            )
        identity_sha256 = workflow_resource_identity_sha256(slot.resource_kind, identity)
        if binding.resource_identity_sha256 != identity_sha256:
            raise WorkflowBindingError(
                "invalid_dependency_binding_digest",
                f"Workflow dependency {binding.slot_name}.{binding.requirement_key} "
                "has an invalid identity digest",
            )
        mount = _portable_binding_mapping(
            binding.mount,
            label=f"workflow dependency mount {binding.slot_name}.{binding.requirement_key}",
            error_code="invalid_dependency_bindings",
        )
        grouped[slot.name].add(binding.requirement_key)
        canonical_bindings.append(
            {
                "slot": binding.slot_name,
                "requirement": binding.requirement_key,
                "resource_kind": binding.resource_kind,
                "identity": identity,
                "mount": mount,
            }
        )

    for slot in contract.slots:
        bound = grouped[slot.name]
        declared = {item.key for item in slot.requirements}
        if slot.satisfaction == "any_of":
            if len(bound) > 1 or (slot.required and len(bound) != 1):
                raise WorkflowBindingError(
                    "incomplete_dependency_bindings",
                    f"Workflow dependency slot {slot.name} is incomplete",
                )
        elif (slot.required and bound != declared) or (
            not slot.required and bound and bound != declared
        ):
            raise WorkflowBindingError(
                "incomplete_dependency_bindings",
                f"Workflow dependency slot {slot.name} is incomplete",
            )

    ordered = sorted(
        canonical_bindings,
        key=lambda item: (str(item["slot"]), str(item["requirement"])),
    )
    bound_slots = {item["slot"] for item in ordered}
    payload = {
        "version": WORKFLOW_BINDING_VERSION,
        "dependency_contract_sha256": workflow_dependency_contract_sha256(contract),
        "bindings": ordered,
        "empty_optional_slots": sorted(
            slot.name
            for slot in contract.slots
            if not slot.required and slot.name not in bound_slots
        ),
    }
    return hashlib.sha256(canonical_workflow_dependency_json(payload)).hexdigest()


def materialize_model_install(
    session: Session,
    install: ModelInstall,
) -> MaterializedWorkflowDependency:
    if not install.active:
        raise WorkflowBindingError("dependency_unavailable", "Model install is inactive")
    comfy_paths = _comfy_paths(install) if install.engine == "comfyui" else None
    identity = {
        "kind": "model_install",
        "role": _stable_text(install.role, "model role"),
        "engine": _stable_text(install.engine, "model engine"),
        "components": _model_component_identity(session, install, comfy_paths=comfy_paths),
    }
    return _materialized("model_install", identity)


def materialize_model_profile(
    session: Session,
    profile: ModelProfile,
) -> MaterializedWorkflowDependency:
    try:
        install = validate_profile_binding(session, profile)
    except (LookupError, ValueError) as exc:
        raise WorkflowBindingError(
            "dependency_unavailable", "Model profile is not runnable"
        ) from exc
    if install is None:
        raise WorkflowBindingError("dependency_unavailable", "Model profile is not bound")
    model = materialize_model_install(session, install).identity
    identity = {
        "kind": "model_profile",
        "role": _stable_text(profile.role, "profile role"),
        "engine": _stable_text(profile.engine, "profile engine"),
        "model": {key: value for key, value in model.items() if key != "kind"},
        "load_settings": _portable_binding_mapping(
            profile.load_settings_json,
            label="model profile load settings",
            error_code="invalid_dependency_identity",
        ),
    }
    return _materialized("model_profile", identity)


def materialize_model_asset(asset: ModelAssetInstall) -> MaterializedWorkflowDependency:
    if not isinstance(asset.manifest_json, dict):
        raise WorkflowBindingError("dependency_unavailable", "Model asset is not verified")
    digest = asset.manifest_json.get("sha256")
    runtime_reference = asset.manifest_json.get("comfy_name")
    if (
        not asset.active
        or asset.verified_at is None
        or asset.kind not in AUXILIARY_ASSET_KINDS
        or not isinstance(digest, str)
        or not _DIGEST_INPUT.fullmatch(digest)
        or not isinstance(runtime_reference, str)
    ):
        raise WorkflowBindingError("dependency_unavailable", "Model asset is not verified")
    return _materialized(
        "model_asset",
        {
            "kind": "model_asset",
            "asset_kind": asset.kind,
            "runtime_reference": _runtime_reference(runtime_reference),
            "sha256": digest.lower(),
        },
    )


def materialize_custom_node(install: CustomNodeInstall) -> MaterializedWorkflowDependency:
    if not install.active or not install.trusted:
        raise WorkflowBindingError(
            "dependency_unavailable", "Custom node is not trusted and active"
        )
    source_url = _canonical_github_repository(install.source_url)
    revision = install.revision.strip()
    tree_hash = install.tree_hash.strip()
    if not _COMMIT.fullmatch(revision) or not _TREE_HASH.fullmatch(tree_hash):
        raise WorkflowBindingError("invalid_dependency_identity", "Custom node identity is invalid")
    return _materialized(
        "custom_node",
        {
            "kind": "custom_node",
            "source_url": source_url,
            "revision": revision.lower(),
            "tree_hash": tree_hash.lower(),
        },
    )


def materialize_registry_package(
    install: ComfyRegistryInstall,
) -> MaterializedWorkflowDependency:
    hashes = {
        "archive_sha256": install.archive_sha256,
        "manifest_sha256": install.manifest_sha256,
        "wheel_closure_sha256": install.wheel_closure_sha256,
        "wheel_environment_sha256": install.wheel_environment_sha256,
    }
    environment_match = (
        _REGISTRY_ENVIRONMENT_PATH.fullmatch(install.wheel_environment_path)
        if isinstance(install.wheel_environment_path, str)
        else None
    )
    if (
        not install.active
        or not install.trusted
        or not _PACKAGE_ID.fullmatch(install.package_id)
        or not (
            _SEMANTIC_VERSION.fullmatch(install.package_version)
            or _LOWERCASE_COMMIT.fullmatch(install.package_version)
        )
        or not install.registry_record_id
        or len(install.registry_record_id) > 1_000
        or environment_match is None
        or any(
            not isinstance(value, str) or not _DIGEST_INPUT.fullmatch(value)
            for value in hashes.values()
        )
        or environment_match.group(1) != str(install.wheel_closure_sha256).lower()
    ):
        raise WorkflowBindingError(
            "dependency_unavailable", "Registry package does not have a complete trusted closure"
        )
    node_types = _registry_node_types(install.node_types_json)
    return _materialized(
        "registry_package",
        {
            "kind": "registry_package",
            "package_id": install.package_id,
            "package_version": install.package_version,
            "registry_record_id": install.registry_record_id,
            "node_types": node_types,
            **{key: str(value).lower() for key, value in hashes.items()},
        },
    )


def materialize_runtime(
    *,
    engine: str,
    runtime_build: str,
    adapter_contract_version: int,
    launch_contract_version: str,
) -> MaterializedWorkflowDependency:
    if type(adapter_contract_version) is not int or adapter_contract_version < 1:
        raise WorkflowBindingError(
            "invalid_dependency_identity", "Runtime adapter contract version is invalid"
        )
    return _materialized(
        "runtime",
        {
            "kind": "runtime",
            "engine": _stable_text(engine, "runtime engine"),
            "runtime_build": _bounded_printable_text(runtime_build, "runtime build"),
            "adapter_contract_version": adapter_contract_version,
            "launch_contract_version": _stable_text(
                launch_contract_version, "runtime launch contract"
            ),
        },
    )


def _materialized(
    resource_kind: WorkflowDependencyResourceKind,
    identity: dict[str, Any],
) -> MaterializedWorkflowDependency:
    portable = _portable_identity_mapping(
        identity,
        label="workflow resource identity",
    )
    return MaterializedWorkflowDependency(resource_kind, portable)


def _canonical_contract(contract: WorkflowDependencyContract) -> WorkflowDependencyContract:
    try:
        return parse_workflow_dependency_contract(workflow_dependency_contract_payload(contract))
    except WorkflowDependencyError as exc:
        raise WorkflowBindingError(exc.code, str(exc)) from exc


def _portable_binding_mapping(
    value: object,
    *,
    label: str,
    error_code: str,
) -> dict[str, Any]:
    try:
        return validate_portable_workflow_mapping(value, label=label)
    except WorkflowDependencyError as exc:
        raise WorkflowBindingError(error_code, str(exc)) from exc


def _portable_identity_mapping(value: object, *, label: str) -> dict[str, Any]:
    portable = _portable_binding_mapping(
        value,
        label=label,
        error_code="invalid_dependency_identity",
    )
    return cast(dict[str, Any], canonicalize_workflow_dependency_collections(portable))


def _model_component_identity(
    session: Session,
    install: ModelInstall,
    *,
    comfy_paths: dict[str, str] | None,
) -> list[dict[str, str]]:
    rows = list(
        session.scalars(
            select(ModelComponentManifest).where(
                ModelComponentManifest.model_install_id == install.id,
                ModelComponentManifest.required.is_(True),
            )
        ).all()
    )
    if not rows:
        raise WorkflowBindingError(
            "dependency_unavailable", "Model install has no exact component closure"
        )
    components: list[dict[str, str]] = []
    references: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row.sha256, str) or not _DIGEST_INPUT.fullmatch(row.sha256):
            raise WorkflowBindingError(
                "invalid_dependency_identity", "Model component is not content-addressed"
            )
        target_folder = _stable_text(row.target_folder, "model target folder")
        kind = _stable_text(row.kind, "model component kind")
        runtime_reference = _component_runtime_reference(
            row.relative_path,
            target_folder=target_folder,
            comfy_paths=comfy_paths,
        )
        folded = (target_folder.casefold(), runtime_reference.casefold())
        if folded in references:
            raise WorkflowBindingError(
                "invalid_dependency_identity", "Model components have colliding runtime references"
            )
        references.add(folded)
        components.append(
            {
                "kind": kind,
                "target_folder": target_folder,
                "runtime_reference": runtime_reference,
                "sha256": row.sha256.lower(),
            }
        )
    components.sort(
        key=lambda item: (
            item["target_folder"],
            item["runtime_reference"],
            item["kind"],
            item["sha256"],
        )
    )
    return components


def _comfy_paths(install: ModelInstall) -> dict[str, str]:
    if not isinstance(install.manifest_json, dict):
        raise WorkflowBindingError(
            "dependency_unavailable", "Comfy model install has no runtime path mapping"
        )
    raw_paths = install.manifest_json.get("comfy_paths")
    if not isinstance(raw_paths, dict) or not raw_paths:
        raise WorkflowBindingError(
            "dependency_unavailable", "Comfy model install has no runtime path mapping"
        )
    paths: dict[str, str] = {}
    for key, value in raw_paths.items():
        if not isinstance(key, str) or key not in COMFY_MODEL_FOLDERS:
            raise WorkflowBindingError(
                "invalid_dependency_identity", "Comfy model path mapping is invalid"
            )
        paths[key] = _comfy_path_root(value)
    return paths


def _comfy_path_root(value: object) -> str:
    return _canonical_comfy_relative_path(value, allow_root=True)


def _component_runtime_reference(
    relative_path: object,
    *,
    target_folder: str,
    comfy_paths: dict[str, str] | None,
) -> str:
    if comfy_paths is None:
        return _runtime_reference(relative_path)
    reference = _canonical_comfy_relative_path(relative_path, allow_root=False)
    root = comfy_paths.get(target_folder)
    if root is None:
        raise WorkflowBindingError(
            "invalid_dependency_identity",
            "Comfy model component is not exposed to its target loader",
        )
    if root == ".":
        return reference
    prefix = f"{root}/"
    if not reference.startswith(prefix) or len(reference) == len(prefix):
        raise WorkflowBindingError(
            "invalid_dependency_identity",
            "Comfy model component is outside its target loader root",
        )
    return _canonical_comfy_relative_path(reference[len(prefix) :], allow_root=False)


def _canonical_comfy_relative_path(value: object, *, allow_root: bool) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 1_000
        or ":" in value
        or any(character < " " or ord(character) == 127 for character in value)
    ):
        raise WorkflowBindingError("invalid_dependency_identity", "Comfy model path is invalid")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise WorkflowBindingError("invalid_dependency_identity", "Comfy model path is invalid")
    canonical = path.as_posix()
    if canonical == "." and not allow_root:
        raise WorkflowBindingError("invalid_dependency_identity", "Comfy model path is invalid")
    return canonical


def _matches_constraints(expected: object, actual: object, *, parent_key: str = "") -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches_constraints(value, actual[key], parent_key=key)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if parent_key not in UNORDERED_WORKFLOW_CONSTRAINT_ARRAYS:
            return len(expected) == len(actual) and all(
                _matches_constraints(expected_item, actual_item, parent_key=parent_key)
                for expected_item, actual_item in zip(expected, actual, strict=True)
            )
        if len(expected) > len(actual):
            return False
        unmatched = list(actual)
        for expected_item in expected:
            match = next(
                (
                    index
                    for index, actual_item in enumerate(unmatched)
                    if _matches_constraints(expected_item, actual_item, parent_key=parent_key)
                ),
                None,
            )
            if match is None:
                return False
            unmatched.pop(match)
        return True
    return type(expected) is type(actual) and expected == actual


def _runtime_reference(value: object) -> str:
    if not isinstance(value, str):
        raise WorkflowBindingError(
            "invalid_dependency_identity", "Model component runtime reference is invalid"
        )
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or len(value) > 1_000
        or "\\" in value
        or path.is_absolute()
        or ":" in value
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(character < " " or ord(character) == 127 for character in value)
    ):
        raise WorkflowBindingError(
            "invalid_dependency_identity", "Model component runtime reference is invalid"
        )
    return value


def _canonical_github_repository(value: object) -> str:
    if not isinstance(value, str):
        raise WorkflowBindingError("invalid_dependency_identity", "Custom node source is invalid")
    parsed = urlparse(value.strip())
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or len(parts) != 2
    ):
        raise WorkflowBindingError("invalid_dependency_identity", "Custom node source is invalid")
    repository = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", parts[0]) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", repository
    ):
        raise WorkflowBindingError("invalid_dependency_identity", "Custom node source is invalid")
    return f"https://github.com/{parts[0].lower()}/{repository.lower()}.git"


def _registry_node_types(value: object) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 4_096:
        raise WorkflowBindingError(
            "invalid_dependency_identity", "Registry package node types are invalid"
        )
    node_types = [_registry_node_type(item) for item in value]
    if len(node_types) != len(set(node_types)):
        raise WorkflowBindingError(
            "invalid_dependency_identity", "Registry package node types are invalid"
        )
    return sorted(node_types)


def _registry_node_type(value: object) -> str:
    return _bounded_printable_text(value, "Registry package node type")


def _bounded_printable_text(value: object, label: str, *, maximum: int = 200) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(character < " " or ord(character) == 127 for character in value)
    ):
        raise WorkflowBindingError("invalid_dependency_identity", f"{label} is invalid")
    return value


def _stable_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not _STABLE_TEXT.fullmatch(value):
        raise WorkflowBindingError("invalid_dependency_identity", f"{label} is invalid")
    return value


def _local_locator(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 200
        or any(character < " " or ord(character) == 127 for character in value)
    ):
        raise WorkflowBindingError(
            "invalid_dependency_bindings", "Workflow dependency locator is invalid"
        )
    return value
