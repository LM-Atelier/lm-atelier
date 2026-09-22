"""Bind retained source reviews to the exact installed dependency closure."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Sequence
from typing import NoReturn

from .comfy_registry_dependencies import ComfyRegistryDependencyError
from .comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies
from .comfy_registry_mixed_wheel_closure import (
    ComfyRegistryMixedWheelClosure,
    mixed_wheel_closure_payload,
)
from .comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from .comfy_registry_runtime import (
    MAX_COMFY_RUNTIME_DISTRIBUTIONS,
    ComfyRegistryRuntimeDistribution,
    ComfyRegistryRuntimeError,
    comfy_registry_runtime_distribution_payload,
)
from .comfy_registry_wheel_closure import MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS
from .comfy_registry_wheel_inputs_v1 import (
    ComfyRegistryWheelInputError,
    ComfyRegistryWheelInputManifest,
    parse_wheel_input_manifest,
    wheel_input_manifest_payload,
)


def reviewed_closure_binding_payload(closure: ComfyRegistryMixedWheelClosure) -> dict[str, object]:
    """Retain a complete closure's input identities without wheel bytes or metadata."""
    witness = mixed_wheel_closure_payload(closure)
    if not closure.complete:
        _fail()
    return {
        "version": 1,
        "closure": witness,
        "inputs": wheel_input_manifest_payload(closure.manifest),
    }


def parse_reviewed_closure_binding(
    value: object,
    *,
    expected_closure_sha256: str,
    declarations: Sequence[str],
) -> ComfyRegistryWheelInputManifest:
    """Verify retained evidence against the installed hash and original declarations."""
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "closure", "inputs"}
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        _fail()
    witness = _witness(value["closure"])
    digest = hashlib.sha256(
        json.dumps(
            witness, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if not _is_digest(expected_closure_sha256) or not hmac.compare_digest(
        digest, expected_closure_sha256
    ):
        _fail("reviewed_closure_hash_mismatch")
    manifest = parse_wheel_input_manifest(value["inputs"])
    if witness["manifest_sha256"] != manifest.manifest_sha256:
        _fail("reviewed_closure_input_mismatch")
    try:
        plan = plan_comfy_registry_mixed_dependencies(declarations)
    except ComfyRegistryDependencyError as exc:
        raise ComfyRegistryWheelInputError("reviewed_closure_declaration_mismatch") from exc
    if (
        manifest.declaration_sha256 != plan.declaration_sha256
        or manifest.remote.declaration_sha256 != plan.remote.declaration_sha256
    ):
        _fail("reviewed_closure_declaration_mismatch")
    return manifest


def validate_current_reviewed_closure_binding(
    value: object,
    *,
    expected_closure_sha256: str,
    declarations: Sequence[str],
    reviewed_inputs: ComfyRegistryReviewedInputContext,
) -> ComfyRegistryWheelInputManifest:
    """Recheck the installed closure's exact source reviews and interpreter target."""
    manifest = parse_reviewed_closure_binding(
        value, expected_closure_sha256=expected_closure_sha256, declarations=declarations
    )
    reviewed_inputs.validate(manifest)
    return manifest


def _witness(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "version",
        "kind",
        "manifest_sha256",
        "metadata_plan_sha256",
        "runtime_distributions",
        "manifest_history",
        "round_number",
        "pending_projects",
        "complete",
    }:
        _fail()
    history = value["manifest_history"]
    if (
        type(value["version"]) is not int
        or value["version"] != 1
        or value["kind"] != "mixed-wheel-inputs"
        or not _is_digest(value["manifest_sha256"])
        or not _is_digest(value["metadata_plan_sha256"])
        or value["complete"] is not True
        or value["pending_projects"] != []
        or not isinstance(history, list)
        or not 1 <= len(history) <= MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS + 1
        or any(not _is_digest(item) for item in history)
        or len(set(history)) != len(history)
        or history[-1] != value["manifest_sha256"]
        or type(value["round_number"]) is not int
        or value["round_number"] != len(history) - 1
    ):
        _fail()
    records = value["runtime_distributions"]
    if not isinstance(records, list) or len(records) > MAX_COMFY_RUNTIME_DISTRIBUTIONS:
        _fail()
    runtime: list[ComfyRegistryRuntimeDistribution] = []
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"name", "version"}
            or not isinstance(record["name"], str)
            or not isinstance(record["version"], str)
        ):
            _fail()
        runtime.append(ComfyRegistryRuntimeDistribution(record["name"], record["version"]))
    try:
        canonical = comfy_registry_runtime_distribution_payload(runtime)
    except ComfyRegistryRuntimeError as exc:
        raise ComfyRegistryWheelInputError("invalid_reviewed_closure_binding") from exc
    if canonical != records:
        _fail()
    return dict(value)


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fail(code: str = "invalid_reviewed_closure_binding") -> NoReturn:
    raise ComfyRegistryWheelInputError(code)
