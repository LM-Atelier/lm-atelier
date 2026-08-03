from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

WORKFLOW_DEPENDENCY_CONTRACT_VERSION = 1
MAX_WORKFLOW_DEPENDENCY_SLOTS = 64
MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS_PER_SLOT = 64
MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS = 512
MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES = 256 * 1024
MAX_WORKFLOW_DEPENDENCY_JSON_NODES = 8_192
MAX_WORKFLOW_DEPENDENCY_JSON_DEPTH = 12
MAX_WORKFLOW_DEPENDENCY_JSON_ARRAY_ITEMS = 4_096

WorkflowDependencyResourceKind = Literal[
    "model_profile",
    "model_install",
    "model_asset",
    "custom_node",
    "registry_package",
    "runtime",
]
WORKFLOW_DEPENDENCY_RESOURCE_KINDS = frozenset(
    {
        "model_profile",
        "model_install",
        "model_asset",
        "custom_node",
        "registry_package",
        "runtime",
    }
)
WorkflowDependencySatisfaction = Literal["all_of", "any_of"]

_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,99}$")
UNORDERED_WORKFLOW_CONSTRAINT_ARRAYS = frozenset({"components", "node_types"})
_FORBIDDEN_PORTABLE_KEYS = frozenset(
    {
        "id",
        "model_profile_id",
        "profile_id",
        "model_install_id",
        "model_asset_install_id",
        "custom_node_install_id",
        "comfy_registry_install_id",
        "install_plan_id",
        "activation_id",
        "local_path",
        "installed_path",
        "wheel_environment_path",
        "source_path",
        "absolute_path",
        "relative_path",
        "credential",
        "credential_id",
        "credentials",
        "secret",
        "token",
        "access_token",
        "api_token",
        "api_key",
        "authorization",
        "cookie",
        "password",
        "passwd",
        "private_key",
        "client_secret",
        "session_token",
        "refresh_token",
        "access_key",
        "secret_key",
        "signed_url",
    }
)


class WorkflowDependencyError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _PortableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _RequirementInput(_PortableModel):
    key: str = Field(min_length=1, max_length=100)
    constraints: dict[str, Any] = Field(default_factory=dict)


class _SlotInput(_PortableModel):
    name: str = Field(min_length=1, max_length=100)
    resource_kind: WorkflowDependencyResourceKind
    required: bool
    satisfaction: WorkflowDependencySatisfaction
    requirements: list[_RequirementInput] = Field(
        min_length=1, max_length=MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS_PER_SLOT
    )


class _ContractInput(_PortableModel):
    version: Literal[1]
    slots: list[_SlotInput] = Field(max_length=MAX_WORKFLOW_DEPENDENCY_SLOTS)


@dataclass(frozen=True)
class WorkflowDependencyRequirement:
    key: str
    constraints: dict[str, Any]


@dataclass(frozen=True)
class WorkflowDependencySlotContract:
    name: str
    resource_kind: WorkflowDependencyResourceKind
    required: bool
    satisfaction: WorkflowDependencySatisfaction
    requirements: tuple[WorkflowDependencyRequirement, ...]


@dataclass(frozen=True)
class WorkflowDependencyContract:
    version: int
    slots: tuple[WorkflowDependencySlotContract, ...]


@dataclass(frozen=True)
class LegacyWorkflowDependencyView:
    """Lossless-enough inventory of legacy fields, without inferring slot semantics."""

    model_install_ids: tuple[str, ...]
    model_references: tuple[object, ...]
    custom_node_references: tuple[object, ...]
    model_components: tuple[dict[str, str], ...]
    unsupported_keys: tuple[str, ...]


def parse_workflow_dependency_contract(value: object) -> WorkflowDependencyContract:
    """Validate and canonicalize one portable, versioned dependency declaration."""

    if not isinstance(value, dict) or type(value.get("version")) is not int:
        raise WorkflowDependencyError(
            "invalid_workflow_dependencies",
            "Workflow dependencies must be a versioned object",
        )
    _preflight_dependency_contract(value)
    _bounded_canonical_json(value)
    try:
        parsed = _ContractInput.model_validate(value, strict=True)
    except ValidationError as exc:
        raise WorkflowDependencyError(
            "invalid_workflow_dependencies",
            "Workflow dependencies do not match the version 1 contract",
        ) from exc

    if len(parsed.slots) > MAX_WORKFLOW_DEPENDENCY_SLOTS:
        raise WorkflowDependencyError(
            "too_many_workflow_dependencies",
            "Workflow dependency contract declares too many slots",
        )

    slot_names: set[str] = set()
    requirement_count = 0
    slots: list[WorkflowDependencySlotContract] = []
    for source_slot in parsed.slots:
        name = _validated_name(source_slot.name, "dependency slot")
        if name in slot_names:
            raise WorkflowDependencyError(
                "duplicate_dependency_slot",
                f"Workflow dependency slot {name} is declared more than once",
            )
        slot_names.add(name)
        if not source_slot.requirements:
            raise WorkflowDependencyError(
                "invalid_workflow_dependencies",
                f"Workflow dependency slot {name} has no requirements",
            )
        if len(source_slot.requirements) > MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS_PER_SLOT:
            raise WorkflowDependencyError(
                "too_many_workflow_dependencies",
                f"Workflow dependency slot {name} declares too many requirements",
            )

        requirement_keys: set[str] = set()
        requirements: list[WorkflowDependencyRequirement] = []
        for source_requirement in source_slot.requirements:
            key = _validated_name(source_requirement.key, "dependency requirement")
            if key in requirement_keys:
                raise WorkflowDependencyError(
                    "duplicate_dependency_requirement",
                    f"Workflow dependency requirement {name}.{key} is declared more than once",
                )
            requirement_keys.add(key)
            constraints = _detached_portable_mapping(
                source_requirement.constraints,
                label=f"dependency requirement {name}.{key}",
            )
            constraints = cast(
                dict[str, Any], canonicalize_workflow_dependency_collections(constraints)
            )
            if "kind" in constraints and constraints["kind"] != source_slot.resource_kind:
                raise WorkflowDependencyError(
                    "invalid_workflow_dependencies",
                    f"Workflow dependency requirement {name}.{key} constrains an "
                    "incompatible resource kind",
                )
            requirements.append(WorkflowDependencyRequirement(key, constraints))

        requirement_count += len(requirements)
        if requirement_count > MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS:
            raise WorkflowDependencyError(
                "too_many_workflow_dependencies",
                "Workflow dependency contract declares too many requirements",
            )
        requirements.sort(key=lambda item: item.key)
        slots.append(
            WorkflowDependencySlotContract(
                name,
                source_slot.resource_kind,
                source_slot.required,
                source_slot.satisfaction,
                tuple(requirements),
            )
        )

    slots.sort(key=lambda item: item.name)
    contract = WorkflowDependencyContract(WORKFLOW_DEPENDENCY_CONTRACT_VERSION, tuple(slots))
    _bounded_canonical_json(workflow_dependency_contract_payload(contract))
    return contract


def workflow_dependency_contract_payload(
    contract: WorkflowDependencyContract,
) -> dict[str, object]:
    return {
        "version": contract.version,
        "slots": [
            workflow_dependency_slot_payload(slot)
            for slot in sorted(contract.slots, key=lambda item: item.name)
        ],
    }


def workflow_dependency_slot_payload(slot: WorkflowDependencySlotContract) -> dict[str, object]:
    return {
        "name": slot.name,
        "resource_kind": slot.resource_kind,
        "required": slot.required,
        "satisfaction": slot.satisfaction,
        "requirements": [
            {"key": requirement.key, "constraints": requirement.constraints}
            for requirement in sorted(slot.requirements, key=lambda item: item.key)
        ],
    }


def workflow_dependency_contract_sha256(contract: WorkflowDependencyContract) -> str:
    canonical = parse_workflow_dependency_contract(workflow_dependency_contract_payload(contract))
    return hashlib.sha256(
        _bounded_canonical_json(workflow_dependency_contract_payload(canonical))
    ).hexdigest()


def workflow_dependency_slot_sha256(slot: WorkflowDependencySlotContract) -> str:
    canonical = parse_workflow_dependency_contract(
        {
            "version": WORKFLOW_DEPENDENCY_CONTRACT_VERSION,
            "slots": [workflow_dependency_slot_payload(slot)],
        }
    ).slots[0]
    payload = {
        "version": WORKFLOW_DEPENDENCY_CONTRACT_VERSION,
        "slot": workflow_dependency_slot_payload(canonical),
    }
    return hashlib.sha256(_bounded_canonical_json(payload)).hexdigest()


def legacy_workflow_dependency_view(value: object) -> LegacyWorkflowDependencyView:
    """Read known legacy fields while keeping their ambiguous unions explicit."""

    if not isinstance(value, dict):
        return LegacyWorkflowDependencyView((), (), (), (), ())
    local_ids = _legacy_strings(value.get("model_install_ids"))
    model_references = _legacy_values(value.get("models"))
    custom_node_references = _legacy_values(value.get("custom_nodes"))
    components: set[tuple[str, str]] = set()
    raw_components = value.get("model_components")
    if isinstance(raw_components, list):
        for item in raw_components:
            if not isinstance(item, dict):
                continue
            folder = item.get("target_folder")
            digest = item.get("sha256")
            if (
                isinstance(folder, str)
                and folder
                and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-fA-F]{64}", digest)
            ):
                components.add((folder, digest.lower()))
    known = {"model_install_ids", "models", "custom_nodes", "model_components"}
    return LegacyWorkflowDependencyView(
        local_ids,
        model_references,
        custom_node_references,
        tuple({"target_folder": folder, "sha256": digest} for folder, digest in sorted(components)),
        tuple(sorted(str(key) for key in value if key not in known)),
    )


def validate_portable_workflow_mapping(value: object, *, label: str) -> dict[str, Any]:
    """Validate a detached JSON mapping for identities and execution mounts."""

    if not isinstance(value, dict):
        raise WorkflowDependencyError(
            "invalid_portable_dependency_data", f"{label} must be an object"
        )
    return _detached_portable_mapping(value, label=label)


def canonical_workflow_dependency_json(value: object) -> bytes:
    """Encode already-validated workflow dependency data deterministically."""

    return _bounded_canonical_json(value)


def _validated_name(value: str, label: str) -> str:
    if not _NAME.fullmatch(value):
        raise WorkflowDependencyError(
            "invalid_workflow_dependencies",
            f"Workflow {label} must use a stable lowercase key",
        )
    return value


def _detached_portable_mapping(value: dict[str, Any], *, label: str) -> dict[str, Any]:
    budget = [0]
    checked = _portable_json(value, label=label, depth=0, budget=budget)
    encoded = _bounded_canonical_json(checked)
    return cast(dict[str, Any], json.loads(encoded.decode("utf-8")))


def _portable_json(
    value: object,
    *,
    label: str,
    depth: int,
    budget: list[int],
) -> object:
    budget[0] += 1
    if budget[0] > MAX_WORKFLOW_DEPENDENCY_JSON_NODES:
        raise WorkflowDependencyError(
            "dependency_data_too_large", f"{label} contains too many JSON values"
        )
    if depth > MAX_WORKFLOW_DEPENDENCY_JSON_DEPTH:
        raise WorkflowDependencyError(
            "dependency_data_too_deep", f"{label} exceeds the nesting limit"
        )
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkflowDependencyError(
                "invalid_portable_dependency_data", f"{label} contains a non-finite number"
            )
        return 0.0 if value == 0 else value
    if isinstance(value, str):
        if len(value) > 4_000 or any(
            character < " " or ord(character) == 127 for character in value
        ):
            raise WorkflowDependencyError(
                "invalid_portable_dependency_data", f"{label} contains invalid text"
            )
        if _looks_like_local_path(value):
            raise WorkflowDependencyError(
                "nonportable_dependency_data", f"{label} contains a local path"
            )
        if _looks_like_secret_url(value):
            raise WorkflowDependencyError(
                "nonportable_dependency_data", f"{label} contains credentials in a URL"
            )
        if (
            re.match(r"^[A-Za-z]:[\\\\/]", value)
            or value.startswith("\\\\")
            or value.startswith("/")
        ):
            raise WorkflowDependencyError(
                "nonportable_dependency_data", f"{label} contains an absolute local path"
            )
        return value
    if isinstance(value, list):
        if len(value) > MAX_WORKFLOW_DEPENDENCY_JSON_ARRAY_ITEMS:
            raise WorkflowDependencyError(
                "dependency_data_too_large", f"{label} contains an oversized array"
            )
        return [_portable_json(item, label=label, depth=depth + 1, budget=budget) for item in value]
    if isinstance(value, dict):
        if len(value) > 512:
            raise WorkflowDependencyError(
                "dependency_data_too_large", f"{label} contains an oversized object"
            )
        result: dict[str, object] = {}
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key) > 200
                or any(character < " " or ord(character) == 127 for character in key)
            ):
                raise WorkflowDependencyError(
                    "invalid_portable_dependency_data", f"{label} contains an invalid field"
                )
            if _is_forbidden_portable_key(key):
                raise WorkflowDependencyError(
                    "nonportable_dependency_data",
                    f"{label} contains the local or secret field {key}",
                )
            result[key] = _portable_json(
                item,
                label=label,
                depth=depth + 1,
                budget=budget,
            )
        return result
    raise WorkflowDependencyError(
        "invalid_portable_dependency_data", f"{label} contains a non-JSON value"
    )


def _looks_like_local_path(value: str) -> bool:
    normalized = value.replace(chr(92), "/")
    if _looks_like_shell_environment_path(normalized):
        return True
    return (
        bool(re.match(r"^[A-Za-z]:", value))
        or value.startswith(("/", chr(92)))
        or bool(re.match(r"^~[^/]*/", normalized))
        or bool(re.match(r"^%[^%]+%/", normalized))
        or bool(re.match(r"^\\$(?:\\{[^}]+\\}|[A-Za-z_][A-Za-z0-9_]*)/", normalized))
        or value.casefold().startswith("file:")
        or ".." in normalized.split("/")
    )


def _looks_like_shell_environment_path(value: str) -> bool:
    if not value.startswith("$") or "/" not in value:
        return False
    prefix = value.split("/", 1)[0]
    name = prefix[2:-1] if prefix.startswith("${") and prefix.endswith("}") else prefix[1:]
    if name.casefold().startswith("env:"):
        name = name[4:]
    return bool(name) and all(character.isalnum() or character == "_" for character in name)


def _is_forbidden_portable_key(key: str) -> bool:
    folded = key.casefold()
    if folded in _FORBIDDEN_PORTABLE_KEYS:
        return True
    compact = re.sub(r"[^a-z0-9]", "", folded)
    return compact.endswith(
        (
            "token",
            "tokens",
            "secret",
            "secrets",
            "password",
            "passwords",
            "passwd",
            "passwds",
            "credential",
            "credentials",
            "privatekey",
            "privatekeys",
            "apikey",
            "apikeys",
            "accesskey",
            "accesskeys",
            "secretkey",
            "secretkeys",
            "signedurl",
            "signedurls",
        )
    )


def _looks_like_secret_url(value: str) -> bool:
    if "://" not in value and "?" not in value and "#" not in value:
        return False
    try:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            return True
        parameter_keys = (
            key
            for encoded in (parsed.query, parsed.fragment)
            for key, _ in parse_qsl(encoded, keep_blank_values=True)
        )
    except ValueError:
        return True
    return any(
        _is_forbidden_portable_key(key)
        or re.sub(r"[^a-z0-9]", "", key.casefold()) in {"key", "sig", "signature", "xamzcredential"}
        for key in parameter_keys
    )


def _bounded_canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowDependencyError(
            "invalid_portable_dependency_data", "Workflow dependency data is not valid JSON"
        ) from exc
    if len(encoded) > MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES:
        raise WorkflowDependencyError(
            "dependency_data_too_large", "Workflow dependency data exceeds the size limit"
        )
    return encoded


def canonicalize_workflow_dependency_collections(value: object, *, parent_key: str = "") -> object:
    if isinstance(value, dict):
        return {
            key: canonicalize_workflow_dependency_collections(item, parent_key=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        items = [
            canonicalize_workflow_dependency_collections(item, parent_key=parent_key)
            for item in value
        ]
        if parent_key in UNORDERED_WORKFLOW_CONSTRAINT_ARRAYS:
            items.sort(key=_bounded_canonical_json)
        return items
    return value


def _preflight_dependency_contract(value: dict[object, object]) -> None:
    slots = value.get("slots")
    if not isinstance(slots, list):
        return
    if len(slots) > MAX_WORKFLOW_DEPENDENCY_SLOTS:
        raise WorkflowDependencyError(
            "too_many_workflow_dependencies",
            "Workflow dependency contract declares too many slots",
        )
    requirement_count = 0
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        requirements = slot.get("requirements")
        if not isinstance(requirements, list):
            continue
        if len(requirements) > MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS_PER_SLOT:
            raise WorkflowDependencyError(
                "too_many_workflow_dependencies",
                "Workflow dependency slot declares too many requirements",
            )
        requirement_count += len(requirements)
        if requirement_count > MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS:
            raise WorkflowDependencyError(
                "too_many_workflow_dependencies",
                "Workflow dependency contract declares too many requirements",
            )


def _legacy_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(sorted({item for item in value if isinstance(item, str) and item}))


def _legacy_values(value: object) -> tuple[object, ...]:
    if not isinstance(value, list):
        return ()
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        copied = json.loads(encoded)
    except (TypeError, ValueError):
        return ()
    return tuple(copied)
