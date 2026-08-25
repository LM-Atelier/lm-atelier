"""Strict portable Prompt Library bundles and mutation-free import previews."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
import time
from dataclasses import dataclass
from typing import NoReturn, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from .domain import Operation
from .lora_constraints import MAX_LORA_STRENGTH
from .model_planner import workflow_artifact_contract
from .models import (
    ModelAssetInstall,
    PromptTemplateDefinition,
    PromptTemplateRevision,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from .prompt_library import prompt_template_workflow_revision_is_ready
from .prompt_templates import (
    PromptTemplateContract,
    PromptTemplateError,
    PromptTemplateLora,
    PromptTemplateLoraPolicy,
    PromptTemplateLoraPolicyMode,
    PromptTemplateResourceMode,
    parse_prompt_template_contract,
    prompt_template_contract_payload,
    prompt_template_contract_sha256,
)
from .workflow_bindings import WorkflowBindingError, materialize_model_asset
from .workflow_dependencies import (
    WorkflowDependencyError,
    parse_workflow_dependency_contract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_sha256,
)

PORTABLE_BUNDLE_KIND = "lm-atelier-prompt-template"
PORTABLE_BUNDLE_VERSION = 1
PORTABLE_DESCRIPTOR_VERSION = 1
PORTABLE_AUTHORITY_RULE = "prompt-template-import-authority-v1"
PORTABLE_RECEIPT_TTL_SECONDS = 10 * 60
MAX_PORTABLE_SUGGESTIONS = 20
MAX_PORTABLE_CANDIDATE_SCAN = 80
MAX_PORTABLE_LORA_SCAN = 4_096
MAX_PORTABLE_BUNDLE_BYTES = 524_288
MAX_PORTABLE_JSON_DEPTH = 20
MAX_PORTABLE_JSON_NODES = 5_000
MAX_PORTABLE_JSON_STRING_CHARACTERS = 131_072
MAX_PORTABLE_JSON_OBJECT_MEMBERS = 4_096
MAX_PORTABLE_JSON_LIST_ITEMS = 4_096
MAX_PORTABLE_JSON_NUMBER_CHARACTERS = 128

PORTABLE_INVALID = "Prompt template bundle is invalid."
PORTABLE_NOT_FOUND = "Prompt template does not exist."
PORTABLE_CONFLICT = "Prompt template conflicts with its stored revision."
PORTABLE_RECEIPT_INVALID = "Prompt template import preview has expired or is invalid."

_BUNDLE_CONTEXT = b"lm-atelier-prompt-template-bundle-v1\0"
_REQUIREMENTS_CONTEXT = b"lm-atelier-prompt-template-requirements-v1\0"
_RECEIPT_CONTEXT = b"lm-atelier-prompt-template-import-receipt-v1\0"
_CANDIDATE_RECEIPT_CONTEXT = b"lm-atelier-prompt-template-candidate-receipt-v1\0"
_WORKFLOW_AUTHORITY_CONTEXT = b"lm-atelier-prompt-template-workflow-authority-v1\0"
_LORA_AUTHORITY_CONTEXT = b"lm-atelier-prompt-template-lora-authority-v1\0"
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_BINDING_KEY = re.compile(r"workflow_[1-9][0-9]{0,2}", re.ASCII)
_LOCAL_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}", re.ASCII)
_HEX_FLOAT = re.compile(r"-?0x(?:0\.0|[01]\.[0-9a-f]{13})p[+-](?:0|[1-9][0-9]{0,3})", re.ASCII)


class PromptTemplatePortabilityError(ValueError):
    """One non-echoing portable bundle boundary failure."""

    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class PortablePromptTemplateBundle:
    payload: dict[str, object]
    bundle_sha256: str
    workflow_requirements: tuple[dict[str, object], ...]
    lora_requirements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PortablePromptTemplatePreview:
    bundle: PortablePromptTemplateBundle
    requirements: tuple[dict[str, object], ...]
    receipt: str
    expires_at: int


def _invalid(code: str = "prompt-template-bundle-invalid") -> NoReturn:
    raise PromptTemplatePortabilityError(code, PORTABLE_INVALID, status_code=422)


def _conflict() -> NoReturn:
    raise PromptTemplatePortabilityError(
        "prompt-template-export-conflict", PORTABLE_CONFLICT, status_code=409
    )


def _receipt_invalid() -> NoReturn:
    raise PromptTemplatePortabilityError(
        "prompt-template-import-receipt-invalid",
        PORTABLE_RECEIPT_INVALID,
        status_code=409,
    )


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, UnicodeError, ValueError) as exc:
        raise PromptTemplatePortabilityError(
            "prompt-template-bundle-invalid", PORTABLE_INVALID, status_code=422
        ) from exc


def _digest(context: bytes, value: object) -> str:
    return hashlib.sha256(context + _canonical_json(value)).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _invalid("prompt-template-bundle-duplicate-key")
        result[key] = value
    return result


def _parse_json(raw: object) -> dict[str, object]:
    if type(raw) is bytes:
        encoded = raw
        try:
            text = raw.decode("utf-8")
        except UnicodeError as exc:
            raise PromptTemplatePortabilityError(
                "prompt-template-bundle-text-invalid", PORTABLE_INVALID, status_code=422
            ) from exc
    elif type(raw) is str:
        text = raw
        try:
            encoded = raw.encode("utf-8")
        except UnicodeError as exc:
            raise PromptTemplatePortabilityError(
                "prompt-template-bundle-text-invalid", PORTABLE_INVALID, status_code=422
            ) from exc
    else:
        _invalid()
    if not encoded or len(encoded) > MAX_PORTABLE_BUNDLE_BYTES or text.startswith("\ufeff"):
        _invalid("prompt-template-bundle-size-invalid")

    def reject_constant(_value: str) -> NoReturn:
        _invalid("prompt-template-bundle-number-invalid")

    def bounded_int(token: str) -> int:
        if len(token) > MAX_PORTABLE_JSON_NUMBER_CHARACTERS:
            _invalid("prompt-template-bundle-number-invalid")
        try:
            return int(token)
        except ValueError as exc:
            raise PromptTemplatePortabilityError(
                "prompt-template-bundle-number-invalid", PORTABLE_INVALID, status_code=422
            ) from exc

    def bounded_float(token: str) -> float:
        if len(token) > MAX_PORTABLE_JSON_NUMBER_CHARACTERS:
            _invalid("prompt-template-bundle-number-invalid")
        try:
            value = float(token)
        except (OverflowError, ValueError) as exc:
            raise PromptTemplatePortabilityError(
                "prompt-template-bundle-number-invalid", PORTABLE_INVALID, status_code=422
            ) from exc
        if not math.isfinite(value):
            _invalid("prompt-template-bundle-number-invalid")
        return value

    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=bounded_int,
            parse_float=bounded_float,
            parse_constant=reject_constant,
        )
    except PromptTemplatePortabilityError:
        raise
    except (RecursionError, UnicodeError, ValueError) as exc:
        raise PromptTemplatePortabilityError(
            "prompt-template-bundle-invalid", PORTABLE_INVALID, status_code=422
        ) from exc
    if type(value) is not dict:
        _invalid()
    root = cast(dict[str, object], value)
    nodes = 0
    string_characters = 0
    object_members = 0
    list_items = 0
    pending: list[tuple[object, int]] = [(root, 1)]
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAX_PORTABLE_JSON_NODES:
            _invalid("prompt-template-bundle-size-invalid")
        if depth > MAX_PORTABLE_JSON_DEPTH:
            _invalid("prompt-template-bundle-depth-invalid")
        if type(current) is dict:
            object_members += len(current)
            if object_members > MAX_PORTABLE_JSON_OBJECT_MEMBERS:
                _invalid("prompt-template-bundle-size-invalid")
            for key, child in current.items():
                string_characters += len(key)
                pending.append((child, depth + 1))
        elif type(current) is list:
            list_items += len(current)
            if list_items > MAX_PORTABLE_JSON_LIST_ITEMS:
                _invalid("prompt-template-bundle-size-invalid")
            pending.extend((child, depth + 1) for child in current)
        elif type(current) is str:
            string_characters += len(current)
        if string_characters > MAX_PORTABLE_JSON_STRING_CHARACTERS:
            _invalid("prompt-template-bundle-size-invalid")
    return root


def _exact(value: object, keys: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        _invalid()
    return cast(dict[str, object], value)


def _strength_from_portable(value: object) -> float:
    if type(value) is not str or _HEX_FLOAT.fullmatch(value) is None:
        _invalid("prompt-template-bundle-strength-invalid")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise PromptTemplatePortabilityError(
            "prompt-template-bundle-strength-invalid", PORTABLE_INVALID, status_code=422
        ) from exc
    if (
        not math.isfinite(parsed)
        or parsed == 0.0
        and value.startswith("-")
        or not -MAX_LORA_STRENGTH <= parsed <= MAX_LORA_STRENGTH
        or parsed.hex() != value
    ):
        _invalid("prompt-template-bundle-strength-invalid")
    return parsed


def _strength_to_portable(value: object) -> str:
    if type(value) not in {int, float}:
        _invalid("prompt-template-bundle-strength-invalid")
    parsed = float(cast(int | float, value))
    if not math.isfinite(parsed) or not -MAX_LORA_STRENGTH <= parsed <= MAX_LORA_STRENGTH:
        _invalid("prompt-template-bundle-strength-invalid")
    return (0.0 if parsed == 0.0 else parsed).hex()


def _portable_lora_policy_to_local(value: object) -> dict[str, object]:
    if type(value) is not dict:
        _invalid()
    mode = value.get("mode")
    if mode in {"inherited_auto", "none"}:
        return dict(_exact(value, frozenset({"mode"})))
    stacks_value: object
    if mode == "fixed":
        item = _exact(value, frozenset({"mode", "stack"}))
        stacks_value = [item["stack"]]
    elif mode == "pool":
        item = _exact(value, frozenset({"mode", "strategy", "stacks"}))
        stacks_value = item["stacks"]
    else:
        _invalid()
    if type(stacks_value) is not list:
        _invalid()
    stacks = cast(list[object], stacks_value)
    converted: list[list[dict[str, object]]] = []
    for raw_stack in stacks:
        if type(raw_stack) is not list:
            _invalid()
        stack: list[dict[str, object]] = []
        for raw_lora in raw_stack:
            lora = _exact(
                raw_lora,
                frozenset({"sha256", "model_strength", "clip_strength"}),
            )
            stack.append(
                {
                    "sha256": lora["sha256"],
                    "model_strength": _strength_from_portable(lora["model_strength"]),
                    "clip_strength": _strength_from_portable(lora["clip_strength"]),
                }
            )
        converted.append(stack)
    if mode == "fixed":
        return {"mode": mode, "stack": converted[0] if converted else []}
    return {"mode": mode, "strategy": item["strategy"], "stacks": converted}


def _portable_policy_to_local(value: object) -> tuple[dict[str, object], tuple[str, ...]]:
    if type(value) is not dict:
        _invalid()
    mode = value.get("mode")
    if mode == "inherited":
        return dict(_exact(value, frozenset({"mode"}))), ()
    if mode == "fixed":
        item = _exact(
            value,
            frozenset({"mode", "workflow_binding_key", "lora_policy"}),
        )
        key = item["workflow_binding_key"]
        if type(key) is not str or _BINDING_KEY.fullmatch(key) is None:
            _invalid()
        return (
            {
                "mode": mode,
                "workflow_revision_id": key,
                "lora_policy": _portable_lora_policy_to_local(item["lora_policy"]),
            },
            (key,),
        )
    item = _exact(value, frozenset({"mode", "strategy", "options"}))
    if mode != "pool" or type(item["options"]) is not list:
        _invalid()
    options: list[dict[str, object]] = []
    keys: list[str] = []
    for raw_option in item["options"]:
        option = _exact(
            raw_option,
            frozenset({"workflow_binding_key", "lora_policy"}),
        )
        key = option["workflow_binding_key"]
        if type(key) is not str or _BINDING_KEY.fullmatch(key) is None:
            _invalid()
        keys.append(key)
        options.append(
            {
                "workflow_revision_id": key,
                "lora_policy": _portable_lora_policy_to_local(option["lora_policy"]),
            }
        )
    return {"mode": mode, "strategy": item["strategy"], "options": options}, tuple(keys)


def _portable_lora_policy_payload(policy: PromptTemplateLoraPolicy) -> dict[str, object]:
    if policy.mode in {
        PromptTemplateLoraPolicyMode.INHERITED_AUTO,
        PromptTemplateLoraPolicyMode.NONE,
    }:
        return {"mode": policy.mode.value}

    def stack_payload(stack: tuple[PromptTemplateLora, ...]) -> list[dict[str, object]]:
        return [
            {
                "sha256": item.sha256,
                "model_strength": _strength_to_portable(item.model_strength),
                "clip_strength": _strength_to_portable(item.clip_strength),
            }
            for item in stack
        ]

    if policy.mode is PromptTemplateLoraPolicyMode.FIXED:
        return {
            "mode": policy.mode.value,
            "stack": stack_payload(policy.stack),
        }
    if policy.mode is not PromptTemplateLoraPolicyMode.POOL or policy.strategy is None:
        _invalid()
    return {
        "mode": policy.mode.value,
        "strategy": policy.strategy.value,
        "stacks": [stack_payload(stack) for stack in policy.stacks],
    }


def _portable_contract_payload(
    contract: PromptTemplateContract,
    *,
    revision_key: dict[str, str],
) -> dict[str, object]:
    local = prompt_template_contract_payload(contract)
    policy = contract.resource_policy
    if policy.mode is PromptTemplateResourceMode.INHERITED:
        portable_policy: dict[str, object] = {"mode": "inherited"}
    elif policy.mode is PromptTemplateResourceMode.FIXED:
        if policy.workflow_revision_id is None or policy.lora_policy is None:
            _invalid()
        portable_policy = {
            "mode": "fixed",
            "workflow_binding_key": revision_key[policy.workflow_revision_id],
            "lora_policy": _portable_lora_policy_payload(policy.lora_policy),
        }
    else:
        if policy.strategy is None:
            _invalid()
        portable_policy = {
            "mode": "pool",
            "strategy": policy.strategy.value,
            "options": [
                {
                    "workflow_binding_key": revision_key[option.workflow_revision_id],
                    "lora_policy": _portable_lora_policy_payload(option.lora_policy),
                }
                for option in policy.options
            ],
        }
    return {
        "schema_version": local["schema_version"],
        "operation": local["operation"],
        "body": local["body"],
        "slots": local["slots"],
        "resource_policy": portable_policy,
    }


def _contract_revision_ids(contract: PromptTemplateContract) -> tuple[str, ...]:
    policy = contract.resource_policy
    if policy.mode is PromptTemplateResourceMode.INHERITED:
        return ()
    if policy.mode is PromptTemplateResourceMode.FIXED:
        if policy.workflow_revision_id is None:
            _invalid()
        return (policy.workflow_revision_id,)
    return tuple(option.workflow_revision_id for option in policy.options)


def _workflow_descriptor(
    session: Session,
    revision: WorkflowRevision,
) -> tuple[WorkflowDefinition, dict[str, object]] | None:
    definition = session.get(WorkflowDefinition, revision.workflow_id)
    if definition is None or definition.operation != Operation.TEXT_TO_IMAGE.value:
        return None
    try:
        artifact_sha256 = workflow_artifact_contract(
            operation=definition.operation,
            engine=revision.engine,
            api_graph=revision.api_graph_json,
            input_schema=revision.input_schema_json,
            dependencies=revision.dependencies_json,
        )
    except (RecursionError, TypeError, UnicodeError, ValueError):
        return None
    if revision.artifact_sha256 != artifact_sha256:
        return None
    rows = list(
        session.scalars(
            select(WorkflowDependencySlot)
            .where(WorkflowDependencySlot.workflow_revision_id == revision.id)
            .order_by(WorkflowDependencySlot.ordinal)
        ).all()
    )
    if revision.dependency_contract_sha256 is None:
        if rows:
            return None
    else:
        if [row.ordinal for row in rows] != list(range(len(rows))):
            return None
        try:
            contract = parse_workflow_dependency_contract(
                {
                    "version": 1,
                    "slots": [
                        {
                            "name": row.name,
                            "resource_kind": row.resource_kind,
                            "required": row.required,
                            "satisfaction": row.satisfaction,
                            "requirements": row.requirements_json,
                        }
                        for row in rows
                    ],
                }
            )
        except WorkflowDependencyError:
            return None
        slots = {slot.name: slot for slot in contract.slots}
        if any(
            row.name not in slots
            or row.contract_sha256 != workflow_dependency_slot_sha256(slots[row.name])
            for row in rows
        ):
            return None
        if revision.dependency_contract_sha256 != workflow_dependency_contract_sha256(contract):
            return None
    return (
        definition,
        {
            "descriptor_version": PORTABLE_DESCRIPTOR_VERSION,
            "operation": definition.operation,
            "artifact_sha256": artifact_sha256,
            "dependency_contract_sha256": revision.dependency_contract_sha256,
        },
    )


def _workflow_suggestion_authority(
    session: Session,
    revision: WorkflowRevision,
    *,
    expected_engine: str,
) -> tuple[WorkflowDefinition, dict[str, object], str] | None:
    descriptor_result = _workflow_descriptor(session, revision)
    if descriptor_result is None:
        return None
    definition, descriptor = descriptor_result
    if definition.current_revision_id != revision.id:
        return None
    family_payload: dict[str, object]
    if definition.family_id is None:
        family_payload = {"kind": "legacy-current"}
    else:
        family = session.get(WorkflowFamily, definition.family_id)
        preference = session.scalar(
            select(WorkflowPreference).where(
                WorkflowPreference.workflow_family_id == definition.family_id,
                WorkflowPreference.selector_capability == "image",
                WorkflowPreference.enabled.is_(True),
            )
        )
        if family is None or family.archived or not family.enabled or preference is None:
            return None
        family_payload = {
            "kind": "enabled-family-current",
            "family_id": family.id,
            "preference_id": preference.id,
        }
    if not prompt_template_workflow_revision_is_ready(
        session, revision, expected_engine=expected_engine
    ):
        return None
    activation_payload: dict[str, object] | None = None
    if revision.dependency_contract_sha256 is not None:
        activation = session.scalar(
            select(WorkflowActivation).where(
                WorkflowActivation.workflow_revision_id == revision.id,
                WorkflowActivation.is_active.is_(True),
                WorkflowActivation.state == "ready",
                WorkflowActivation.invalidated_at.is_(None),
            )
        )
        launch_sha256 = (
            activation.details_json.get("launch_sha256")
            if activation is not None and type(activation.details_json) is dict
            else None
        )
        if (
            activation is None
            or type(launch_sha256) is not str
            or _SHA256.fullmatch(launch_sha256) is None
        ):
            return None
        activation_payload = {
            "activation_id": activation.id,
            "dependency_contract_sha256": activation.dependency_contract_sha256,
            "binding_sha256": activation.binding_sha256,
            "launch_sha256": launch_sha256,
        }
    authority = {
        "version": 1,
        "local_ref": revision.id,
        "descriptor": descriptor,
        "engine": revision.engine,
        "trusted": revision.trusted,
        "family": family_payload,
        "activation": activation_payload,
    }
    return definition, descriptor, _digest(_WORKFLOW_AUTHORITY_CONTEXT, authority)


def _lora_digests(contract: PromptTemplateContract) -> tuple[str, ...]:
    policy = contract.resource_policy
    policies: tuple[PromptTemplateLoraPolicy, ...]
    if policy.mode is PromptTemplateResourceMode.INHERITED:
        policies = ()
    elif policy.mode is PromptTemplateResourceMode.FIXED:
        policies = (policy.lora_policy,) if policy.lora_policy is not None else ()
    else:
        policies = tuple(option.lora_policy for option in policy.options)
    found: set[str] = set()
    for lora_policy in policies:
        if lora_policy.mode is PromptTemplateLoraPolicyMode.FIXED:
            found.update(item.sha256 for item in lora_policy.stack)
        elif lora_policy.mode is PromptTemplateLoraPolicyMode.POOL:
            found.update(item.sha256 for stack in lora_policy.stacks for item in stack)
    return tuple(sorted(found))


def parse_portable_prompt_template_bundle(raw: object) -> PortablePromptTemplateBundle:
    """Parse one JSON string without allowing portable keys into the live codec."""

    root = _exact(
        _parse_json(raw),
        frozenset({"kind", "bundle_version", "template", "workflows", "bundle_sha256"}),
    )
    if root["kind"] != PORTABLE_BUNDLE_KIND:
        _invalid()
    if type(root["bundle_version"]) is not int or root["bundle_version"] != 1:
        _invalid()
    claimed_digest = root["bundle_sha256"]
    if type(claimed_digest) is not str or _SHA256.fullmatch(claimed_digest) is None:
        _invalid()
    unsigned = {key: value for key, value in root.items() if key != "bundle_sha256"}
    computed_digest = _digest(_BUNDLE_CONTEXT, unsigned)
    if not hmac.compare_digest(claimed_digest, computed_digest):
        _invalid("prompt-template-bundle-digest-invalid")

    template = _exact(root["template"], frozenset({"name", "description", "contract"}))
    name = template["name"]
    description = template["description"]
    if (
        type(name) is not str
        or not name.strip()
        or name != name.strip()
        or len(name) > 200
        or "\0" in name
        or type(description) is not str
        or description != description.strip()
        or len(description) > 4_000
        or "\0" in description
    ):
        _invalid()
    portable_contract = _exact(
        template["contract"],
        frozenset({"schema_version", "operation", "body", "slots", "resource_policy"}),
    )
    local_policy, traversal_keys = _portable_policy_to_local(portable_contract["resource_policy"])
    try:
        contract = parse_prompt_template_contract(
            {
                "schema_version": portable_contract["schema_version"],
                "operation": portable_contract["operation"],
                "body": portable_contract["body"],
                "slots": portable_contract["slots"],
                "resource_policy": local_policy,
            }
        )
    except PromptTemplateError as exc:
        raise PromptTemplatePortabilityError(
            "prompt-template-bundle-invalid", PORTABLE_INVALID, status_code=422
        ) from exc

    raw_workflows = root["workflows"]
    if type(raw_workflows) is not list or len(raw_workflows) > 16:
        _invalid()
    requirements: list[dict[str, object]] = []
    for raw_requirement in raw_workflows:
        requirement = _exact(raw_requirement, frozenset({"key", "descriptor"}))
        key = requirement["key"]
        descriptor = _exact(
            requirement["descriptor"],
            frozenset(
                {
                    "descriptor_version",
                    "operation",
                    "artifact_sha256",
                    "dependency_contract_sha256",
                }
            ),
        )
        dependency_digest = descriptor["dependency_contract_sha256"]
        if (
            type(key) is not str
            or _BINDING_KEY.fullmatch(key) is None
            or type(descriptor["descriptor_version"]) is not int
            or descriptor["descriptor_version"] != PORTABLE_DESCRIPTOR_VERSION
            or descriptor["operation"] != Operation.TEXT_TO_IMAGE.value
            or type(descriptor["artifact_sha256"]) is not str
            or _SHA256.fullmatch(descriptor["artifact_sha256"]) is None
            or (
                dependency_digest is not None
                and (
                    type(dependency_digest) is not str
                    or _SHA256.fullmatch(dependency_digest) is None
                )
            )
        ):
            _invalid()
        requirements.append({"key": key, "descriptor": dict(descriptor)})
    first_traversal: list[str] = []
    for key in traversal_keys:
        if key not in first_traversal:
            first_traversal.append(key)
    canonical_keys = [f"workflow_{index}" for index in range(1, len(first_traversal) + 1)]
    if (
        first_traversal != canonical_keys
        or [item["key"] for item in requirements] != canonical_keys
    ):
        _invalid("prompt-template-bundle-bindings-invalid")

    normalized_contract = _portable_contract_payload(
        contract,
        revision_key={key: key for key in first_traversal},
    )
    normalized_unsigned = {
        "kind": PORTABLE_BUNDLE_KIND,
        "bundle_version": PORTABLE_BUNDLE_VERSION,
        "template": {
            "name": name,
            "description": description,
            "contract": normalized_contract,
        },
        "workflows": requirements,
    }
    if _digest(_BUNDLE_CONTEXT, normalized_unsigned) != claimed_digest:
        _invalid("prompt-template-bundle-canonical-invalid")
    payload = {**normalized_unsigned, "bundle_sha256": claimed_digest}
    return PortablePromptTemplateBundle(
        payload=payload,
        bundle_sha256=claimed_digest,
        workflow_requirements=tuple(requirements),
        lora_requirements=_lora_digests(contract),
    )


def export_prompt_template_bundle(
    session: Session,
    *,
    definition_id: str,
    revision_id: str,
) -> PortablePromptTemplateBundle:
    """Export one exact historical revision without machine-local identities."""

    definition = session.get(PromptTemplateDefinition, definition_id)
    revision = session.get(PromptTemplateRevision, revision_id)
    if definition is None or revision is None or revision.prompt_template_id != definition_id:
        raise PromptTemplatePortabilityError(
            "prompt-template-revision-not-found", PORTABLE_NOT_FOUND, status_code=404
        )
    try:
        contract = parse_prompt_template_contract(revision.contract_json)
        if (
            revision.schema_version != contract.schema_version
            or revision.contract_sha256 != prompt_template_contract_sha256(contract)
        ):
            _conflict()
    except PromptTemplateError as exc:
        raise PromptTemplatePortabilityError(
            "prompt-template-export-conflict", PORTABLE_CONFLICT, status_code=409
        ) from exc

    revision_key: dict[str, str] = {}
    requirements: list[dict[str, object]] = []
    for source_revision_id in _contract_revision_ids(contract):
        if source_revision_id in revision_key:
            continue
        source_revision = session.get(WorkflowRevision, source_revision_id)
        descriptor_result = (
            _workflow_descriptor(session, source_revision) if source_revision is not None else None
        )
        if descriptor_result is None:
            _conflict()
        _source_definition, descriptor = descriptor_result
        key = f"workflow_{len(revision_key) + 1}"
        revision_key[source_revision_id] = key
        requirements.append({"key": key, "descriptor": descriptor})
    unsigned = {
        "kind": PORTABLE_BUNDLE_KIND,
        "bundle_version": PORTABLE_BUNDLE_VERSION,
        "template": {
            "name": definition.name,
            "description": definition.description,
            "contract": _portable_contract_payload(contract, revision_key=revision_key),
        },
        "workflows": requirements,
    }
    bundle_sha256 = _digest(_BUNDLE_CONTEXT, unsigned)
    payload = {**unsigned, "bundle_sha256": bundle_sha256}
    try:
        return parse_portable_prompt_template_bundle(_canonical_json(payload))
    except PromptTemplatePortabilityError:
        _conflict()


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str, *, maximum: int) -> bytes:
    if not value or len(value) > maximum or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        _receipt_invalid()
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise PromptTemplatePortabilityError(
            "prompt-template-import-receipt-invalid", PORTABLE_RECEIPT_INVALID, status_code=409
        ) from exc
    if len(decoded) > maximum:
        _receipt_invalid()
    if not hmac.compare_digest(_b64encode(decoded), value):
        _receipt_invalid()
    return decoded


def _requirements_payload(bundle: PortablePromptTemplateBundle) -> dict[str, object]:
    return {
        "authority_rule": PORTABLE_AUTHORITY_RULE,
        "workflows": list(bundle.workflow_requirements),
        "lora_sha256": list(bundle.lora_requirements),
    }


def _issue_receipt(
    bundle: PortablePromptTemplateBundle,
    *,
    signing_key: bytes,
    issued_at: int,
) -> tuple[str, int]:
    if type(signing_key) is not bytes or len(signing_key) != hashlib.sha256().digest_size:
        _receipt_invalid()
    if type(issued_at) is not int or issued_at < 0:
        _receipt_invalid()
    expires_at = issued_at + PORTABLE_RECEIPT_TTL_SECONDS
    payload = {
        "version": 1,
        "bundle_sha256": bundle.bundle_sha256,
        "requirements_sha256": _digest(_REQUIREMENTS_CONTEXT, _requirements_payload(bundle)),
        "authority_rule": PORTABLE_AUTHORITY_RULE,
        "issued_at": issued_at,
        "expires_at": expires_at,
    }
    encoded = _canonical_json(payload)
    signature = hmac.new(signing_key, _RECEIPT_CONTEXT + encoded, hashlib.sha256).digest()
    return f"{_b64encode(encoded)}.{_b64encode(signature)}", expires_at


def _issue_candidate_receipt(
    payload: dict[str, object],
    *,
    signing_key: bytes,
) -> str:
    if type(signing_key) is not bytes or len(signing_key) != hashlib.sha256().digest_size:
        _receipt_invalid()
    encoded = _canonical_json({"version": 1, **payload})
    signature = hmac.new(
        signing_key,
        _CANDIDATE_RECEIPT_CONTEXT + encoded,
        hashlib.sha256,
    ).digest()
    return f"{_b64encode(encoded)}.{_b64encode(signature)}"


def verify_prompt_template_import_receipt(
    receipt: object,
    bundle: PortablePromptTemplateBundle,
    *,
    signing_key: bytes,
    now: int | None = None,
) -> int:
    """Verify the root preview receipt and return its immutable expiry."""

    return _verify_import_receipt_expires_at(
        receipt,
        bundle,
        signing_key=signing_key,
        now=now,
    )


def _verify_import_receipt_expires_at(
    receipt: object,
    bundle: PortablePromptTemplateBundle,
    *,
    signing_key: bytes,
    now: int | None = None,
) -> int:
    """Verify a root receipt and return its immutable expiry boundary."""

    if type(receipt) is not str or len(receipt) > 2_048 or receipt.count(".") != 1:
        _receipt_invalid()
    if type(signing_key) is not bytes or len(signing_key) != hashlib.sha256().digest_size:
        _receipt_invalid()
    encoded_payload, encoded_signature = receipt.split(".", 1)
    payload_bytes = _b64decode(encoded_payload, maximum=1_200)
    signature = _b64decode(encoded_signature, maximum=64)
    expected = hmac.new(signing_key, _RECEIPT_CONTEXT + payload_bytes, hashlib.sha256).digest()
    if len(signature) != len(expected) or not hmac.compare_digest(signature, expected):
        _receipt_invalid()
    try:
        value = json.loads(payload_bytes, object_pairs_hook=_unique_object)
    except (PromptTemplatePortabilityError, RecursionError, UnicodeError, ValueError) as exc:
        raise PromptTemplatePortabilityError(
            "prompt-template-import-receipt-invalid",
            PORTABLE_RECEIPT_INVALID,
            status_code=409,
        ) from exc
    if type(value) is not dict or set(value) != {
        "version",
        "bundle_sha256",
        "requirements_sha256",
        "authority_rule",
        "issued_at",
        "expires_at",
    }:
        _receipt_invalid()
    current = int(time.time()) if now is None else now
    if type(current) is not int or current < 0:
        _receipt_invalid()
    requirements_sha256 = _digest(_REQUIREMENTS_CONTEXT, _requirements_payload(bundle))
    if (
        type(value["version"]) is not int
        or value["version"] != 1
        or value["bundle_sha256"] != bundle.bundle_sha256
        or value["requirements_sha256"] != requirements_sha256
        or value["authority_rule"] != PORTABLE_AUTHORITY_RULE
        or type(value["issued_at"]) is not int
        or type(value["expires_at"]) is not int
        or value["expires_at"] != value["issued_at"] + PORTABLE_RECEIPT_TTL_SECONDS
        or current < value["issued_at"]
        or current >= value["expires_at"]
    ):
        _receipt_invalid()
    return value["expires_at"]


def verify_prompt_template_candidate_receipt(
    receipt: object,
    expected_payload: dict[str, object],
    *,
    signing_key: bytes,
    now: int | None = None,
) -> None:
    """Verify one displayed candidate or digest authority without trusting its token."""

    kind = expected_payload.get("kind")
    expected_keys = (
        frozenset(
            {
                "kind",
                "bundle_sha256",
                "binding_key",
                "local_ref",
                "authority_sha256",
                "expires_at",
            }
        )
        if kind == "workflow"
        else frozenset({"kind", "bundle_sha256", "sha256", "authority_sha256", "expires_at"})
        if kind == "lora"
        else frozenset()
    )
    if (
        not expected_keys
        or set(expected_payload) != expected_keys
        or type(expected_payload.get("bundle_sha256")) is not str
        or _SHA256.fullmatch(cast(str, expected_payload["bundle_sha256"])) is None
        or type(expected_payload.get("authority_sha256")) is not str
        or _SHA256.fullmatch(cast(str, expected_payload["authority_sha256"])) is None
        or (
            kind == "workflow"
            and (
                type(expected_payload.get("binding_key")) is not str
                or _BINDING_KEY.fullmatch(cast(str, expected_payload["binding_key"])) is None
                or type(expected_payload.get("local_ref")) is not str
                or _LOCAL_REF.fullmatch(cast(str, expected_payload["local_ref"])) is None
            )
        )
        or (
            kind == "lora"
            and (
                type(expected_payload.get("sha256")) is not str
                or _SHA256.fullmatch(cast(str, expected_payload["sha256"])) is None
            )
        )
    ):
        _receipt_invalid()
    if type(receipt) is not str or len(receipt) > 2_048 or receipt.count(".") != 1:
        _receipt_invalid()
    if type(signing_key) is not bytes or len(signing_key) != hashlib.sha256().digest_size:
        _receipt_invalid()
    encoded_payload, encoded_signature = receipt.split(".", 1)
    payload_bytes = _b64decode(encoded_payload, maximum=1_200)
    signature = _b64decode(encoded_signature, maximum=64)
    expected_signature = hmac.new(
        signing_key,
        _CANDIDATE_RECEIPT_CONTEXT + payload_bytes,
        hashlib.sha256,
    ).digest()
    if len(signature) != len(expected_signature) or not hmac.compare_digest(
        signature, expected_signature
    ):
        _receipt_invalid()
    try:
        value = json.loads(payload_bytes, object_pairs_hook=_unique_object)
    except (PromptTemplatePortabilityError, RecursionError, UnicodeError, ValueError) as exc:
        raise PromptTemplatePortabilityError(
            "prompt-template-import-receipt-invalid",
            PORTABLE_RECEIPT_INVALID,
            status_code=409,
        ) from exc
    expected = {"version": 1, **expected_payload}
    current = int(time.time()) if now is None else now
    expires_at = expected_payload.get("expires_at")
    if (
        type(current) is not int
        or current < 0
        or type(expires_at) is not int
        or current >= expires_at
        or type(value) is not dict
        or value != expected
    ):
        _receipt_invalid()


def _verified_lora_digests(session: Session, expected: frozenset[str]) -> set[str]:
    matched: set[str] = set()
    if not expected:
        return matched
    candidates = session.scalars(
        select(ModelAssetInstall)
        .where(
            ModelAssetInstall.kind == "lora",
            ModelAssetInstall.active.is_(True),
            ModelAssetInstall.verified_at.is_not(None),
            ModelAssetInstall.manifest_json["sha256"].as_string().in_(sorted(expected)),
        )
        .order_by(ModelAssetInstall.id)
        .limit(MAX_PORTABLE_LORA_SCAN)
    ).all()
    for candidate in candidates:
        if type(candidate.manifest_json) is not dict:
            continue
        digest = candidate.manifest_json.get("sha256")
        if type(digest) is not str or digest not in expected:
            continue
        try:
            materialized = materialize_model_asset(candidate)
        except WorkflowBindingError:
            continue
        if materialized.identity.get("sha256") == digest:
            matched.add(digest)
    return matched


def bind_portable_prompt_template_contract(
    bundle: PortablePromptTemplateBundle,
    workflow_bindings: dict[str, str],
) -> PromptTemplateContract:
    """Replace portable keys with exact local refs and re-enter the live codec."""

    required = tuple(str(item["key"]) for item in bundle.workflow_requirements)
    if set(workflow_bindings) != set(required) or any(
        type(key) is not str
        or _BINDING_KEY.fullmatch(key) is None
        or type(value) is not str
        or _LOCAL_REF.fullmatch(value) is None
        for key, value in workflow_bindings.items()
    ):
        _invalid("prompt-template-bundle-bindings-invalid")
    template = cast(dict[str, object], bundle.payload["template"])
    portable_contract = cast(dict[str, object], template["contract"])
    local_policy, traversal = _portable_policy_to_local(portable_contract["resource_policy"])
    if tuple(dict.fromkeys(traversal)) != required:
        _invalid("prompt-template-bundle-bindings-invalid")
    if local_policy["mode"] == "fixed":
        key = cast(str, local_policy["workflow_revision_id"])
        local_policy["workflow_revision_id"] = workflow_bindings[key]
    elif local_policy["mode"] == "pool":
        for option in cast(list[dict[str, object]], local_policy["options"]):
            key = cast(str, option["workflow_revision_id"])
            option["workflow_revision_id"] = workflow_bindings[key]
    try:
        return parse_prompt_template_contract(
            {
                "schema_version": portable_contract["schema_version"],
                "operation": portable_contract["operation"],
                "body": portable_contract["body"],
                "slots": portable_contract["slots"],
                "resource_policy": local_policy,
            }
        )
    except PromptTemplateError as exc:
        raise PromptTemplatePortabilityError(
            "prompt-template-bundle-invalid", PORTABLE_INVALID, status_code=422
        ) from exc


def prompt_template_import_workflow_authority(
    session: Session,
    *,
    local_ref: str,
    expected_engine: str,
) -> tuple[dict[str, object], str] | None:
    """Return the fresh descriptor and authority digest for one local workflow."""

    revision = session.get(WorkflowRevision, local_ref)
    if revision is None:
        return None
    result = _workflow_suggestion_authority(
        session,
        revision,
        expected_engine=expected_engine,
    )
    if result is None:
        return None
    _definition, descriptor, authority_sha256 = result
    return descriptor, authority_sha256


def prompt_template_import_lora_authority(
    session: Session,
    *,
    sha256: str,
) -> str | None:
    """Return fresh verified-active LoRA authority, never an unavailable receipt."""

    if _SHA256.fullmatch(sha256) is None:
        return None
    if sha256 not in _verified_lora_digests(session, frozenset({sha256})):
        return None
    return _digest(
        _LORA_AUTHORITY_CONTEXT,
        {"version": 1, "sha256": sha256, "verified_active_exists": True},
    )


def resolve_prompt_template_import_candidate(
    session: Session,
    raw_bundle: object,
    *,
    preview_receipt: object,
    binding_key: object,
    local_ref: object,
    expected_engine: str,
    signing_key: bytes,
    now: int | None = None,
) -> dict[str, object]:
    """Authorize one exact current local workflow without widening preview bounds."""

    bundle = parse_portable_prompt_template_bundle(raw_bundle)
    expires_at = _verify_import_receipt_expires_at(
        preview_receipt,
        bundle,
        signing_key=signing_key,
        now=now,
    )
    if (
        type(binding_key) is not str
        or _BINDING_KEY.fullmatch(binding_key) is None
        or type(local_ref) is not str
        or _LOCAL_REF.fullmatch(local_ref) is None
    ):
        _receipt_invalid()
    requirement = next(
        (item for item in bundle.workflow_requirements if item["key"] == binding_key),
        None,
    )
    if requirement is None:
        _receipt_invalid()
    revision = session.get(WorkflowRevision, local_ref)
    if revision is None:
        _receipt_invalid()
    authority_result = _workflow_suggestion_authority(
        session,
        revision,
        expected_engine=expected_engine,
    )
    expected_descriptor = cast(dict[str, object], requirement["descriptor"])
    if authority_result is None or authority_result[1] != expected_descriptor:
        _receipt_invalid()
    definition, _descriptor, authority_sha256 = authority_result
    candidate_payload = {
        "kind": "workflow",
        "bundle_sha256": bundle.bundle_sha256,
        "binding_key": binding_key,
        "local_ref": local_ref,
        "authority_sha256": authority_sha256,
        "expires_at": expires_at,
    }
    return {
        "local_ref": local_ref,
        "label": definition.name,
        "authority_sha256": authority_sha256,
        "candidate_receipt": _issue_candidate_receipt(
            candidate_payload,
            signing_key=signing_key,
        ),
    }


def preview_prompt_template_import(
    session: Session,
    raw_bundle: object,
    *,
    expected_engine: str,
    signing_key: bytes,
    now: int | None = None,
) -> PortablePromptTemplatePreview:
    """Return bounded local suggestions without mutating persistent state."""

    bundle = parse_portable_prompt_template_bundle(raw_bundle)
    issued_at = int(time.time()) if now is None else now
    if type(issued_at) is not int or issued_at < 0:
        _receipt_invalid()
    receipt, expires_at = _issue_receipt(bundle, signing_key=signing_key, issued_at=issued_at)
    requirements: list[dict[str, object]] = []
    for requirement in bundle.workflow_requirements:
        descriptor = cast(dict[str, object], requirement["descriptor"])
        suggestions: list[dict[str, object]] = []
        dependency_digest = descriptor["dependency_contract_sha256"]
        dependency_filter = (
            WorkflowRevision.dependency_contract_sha256.is_(None)
            if dependency_digest is None
            else WorkflowRevision.dependency_contract_sha256 == dependency_digest
        )
        revisions = session.scalars(
            select(WorkflowRevision)
            .join(WorkflowDefinition, WorkflowDefinition.id == WorkflowRevision.workflow_id)
            .where(
                WorkflowDefinition.operation == descriptor["operation"],
                WorkflowRevision.artifact_sha256 == descriptor["artifact_sha256"],
                dependency_filter,
            )
            .order_by(WorkflowRevision.id)
            .limit(MAX_PORTABLE_CANDIDATE_SCAN)
        ).all()
        for revision in revisions:
            authority_result = _workflow_suggestion_authority(
                session,
                revision,
                expected_engine=expected_engine,
            )
            if authority_result is None or authority_result[1] != descriptor:
                continue
            definition, _candidate_descriptor, authority_sha256 = authority_result
            suggestions.append(
                {
                    "local_ref": revision.id,
                    "label": definition.name,
                    "authority_sha256": authority_sha256,
                    "candidate_receipt": _issue_candidate_receipt(
                        {
                            "kind": "workflow",
                            "bundle_sha256": bundle.bundle_sha256,
                            "binding_key": requirement["key"],
                            "local_ref": revision.id,
                            "authority_sha256": authority_sha256,
                            "expires_at": expires_at,
                        },
                        signing_key=signing_key,
                    ),
                }
            )
            if len(suggestions) == MAX_PORTABLE_SUGGESTIONS:
                break
        requirements.append(
            {
                "kind": "workflow",
                "binding_key": requirement["key"],
                "descriptor": dict(descriptor),
                "suggestions": suggestions,
            }
        )
    available_loras = _verified_lora_digests(session, frozenset(bundle.lora_requirements))
    for digest in bundle.lora_requirements:
        available = digest in available_loras
        authority_sha256 = _digest(
            _LORA_AUTHORITY_CONTEXT,
            {"version": 1, "sha256": digest, "verified_active_exists": available},
        )
        requirements.append(
            {
                "kind": "lora",
                "sha256": digest,
                "available": available,
                "authority_sha256": authority_sha256,
                "confirmation_receipt": _issue_candidate_receipt(
                    {
                        "kind": "lora",
                        "bundle_sha256": bundle.bundle_sha256,
                        "sha256": digest,
                        "authority_sha256": authority_sha256,
                        "expires_at": expires_at,
                    },
                    signing_key=signing_key,
                ),
            }
        )
    return PortablePromptTemplatePreview(
        bundle=bundle,
        requirements=tuple(requirements),
        receipt=receipt,
        expires_at=expires_at,
    )
