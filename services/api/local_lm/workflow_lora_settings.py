"""Transport workflow-native LoRA edits beside ordinary generation settings.

The reserved value is a durable, public, multi-target envelope.  This module
only parses, canonicalizes, and overlays that value.  It does not decide which
target is active, derive graph authority, or apply a graph mutation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from .settings_registry import WORKFLOW_LORA_OVERRIDES_SETTING_KEY
from .workflow_lora_overrides import (
    MAX_WORKFLOW_LORA_OVERRIDE_TARGETS,
    MAX_WORKFLOW_LORA_OVERRIDES_TOTAL,
    WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION,
    WorkflowLoraFieldChange,
    WorkflowLoraOverrideField,
    WorkflowLoraOverrideResolution,
    WorkflowLoraOverrides,
    WorkflowLoraOverrideTarget,
    WorkflowLoraOverrideTargetWitness,
    WorkflowLoraSlotOverride,
    parse_workflow_lora_overrides,
    workflow_lora_override_resolution_payload,
    workflow_lora_overrides_payload,
)

WORKFLOW_LORA_OVERRIDES_SETTING_ROLES = frozenset({"image", "video"})
MAX_WORKFLOW_LORA_SETTING_LAYERS = 16


class WorkflowLoraSettingsError(ValueError):
    """A typed refusal to transport a workflow-native LoRA setting."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class _MutableSlotOverlay:
    loader_contract: str
    loader_authority_sha256: str
    changes: dict[str, bool | float]


@dataclass(frozen=True, slots=True)
class DetachedWorkflowLoraSetting:
    """One settings object with its reserved value removed but not yet parsed.

    ``present`` distinguishes an absent key from any value stored under it, so
    a caller can decide precedence before paying for, or failing on, a parse.
    """

    present: bool
    raw: object
    ordinary: dict[str, Any]


def detach_workflow_lora_overrides_setting(
    settings: Mapping[str, Any] | None,
    *,
    role: str,
) -> DetachedWorkflowLoraSetting:
    """Remove the reserved value from one settings object without parsing it.

    The outer object must still be a plain JSON object with string keys, and
    only image and video settings may carry the reserved key at all.
    """

    if settings is None:
        return DetachedWorkflowLoraSetting(present=False, raw=None, ordinary={})
    if type(settings) is not dict:
        raise WorkflowLoraSettingsError(
            "invalid_workflow_lora_settings",
            "Workflow LoRA settings must be a JSON object",
        )
    raw = cast(dict[object, object], settings)
    if any(type(key) is not str for key in raw):
        raise WorkflowLoraSettingsError(
            "invalid_workflow_lora_settings",
            "Workflow LoRA setting keys must be strings",
        )
    exact = cast(dict[str, Any], raw)
    ordinary = {
        key: value for key, value in exact.items() if key != WORKFLOW_LORA_OVERRIDES_SETTING_KEY
    }
    if WORKFLOW_LORA_OVERRIDES_SETTING_KEY not in exact:
        return DetachedWorkflowLoraSetting(present=False, raw=None, ordinary=ordinary)
    if type(role) is not str or role not in WORKFLOW_LORA_OVERRIDES_SETTING_ROLES:
        raise WorkflowLoraSettingsError(
            "unsupported_workflow_lora_setting_role",
            "Workflow LoRA overrides are supported only for image and video requests",
        )
    return DetachedWorkflowLoraSetting(
        present=True,
        raw=exact[WORKFLOW_LORA_OVERRIDES_SETTING_KEY],
        ordinary=ordinary,
    )


def split_workflow_lora_overrides_setting(
    settings: Mapping[str, Any] | None,
    *,
    role: str,
) -> tuple[WorkflowLoraOverrides | None, dict[str, Any]]:
    """Remove and parse the reserved value before generic schema validation.

    The first result is ``None`` only when the key was absent.  An explicit
    empty envelope remains a typed empty envelope, preserving the difference
    between inheritance and reset.  Only image/video request transports may
    carry this special key; ordinary settings can still pass through for any
    role.
    """

    detached = detach_workflow_lora_overrides_setting(settings, role=role)
    if not detached.present:
        return None, detached.ordinary
    return parse_workflow_lora_overrides(detached.raw), detached.ordinary


def workflow_lora_overrides_setting_value(
    value: WorkflowLoraOverrides,
) -> dict[str, object]:
    """Return the detached canonical JSON value stored under the reserved key."""

    return workflow_lora_overrides_payload(value)


def overlay_workflow_lora_overrides(
    *layers: WorkflowLoraOverrides | None,
) -> WorkflowLoraOverrides:
    """Materialize lower-to-higher durable envelopes one field at a time.

    Targets not mentioned by a higher layer remain intact; this includes
    targets that may be inactive for some future dispatch.  An explicit empty
    envelope is the canonical reset, while ``None`` means inherit.  Matching
    target/slot evidence must agree exactly before fields can be combined.
    """

    if len(layers) > MAX_WORKFLOW_LORA_SETTING_LAYERS:
        raise WorkflowLoraSettingsError(
            "too_many_workflow_lora_setting_layers",
            f"Workflow LoRA settings may overlay at most {MAX_WORKFLOW_LORA_SETTING_LAYERS} layers",
        )
    targets: dict[
        WorkflowLoraOverrideTargetWitness,
        dict[str, _MutableSlotOverlay],
    ] = {}
    for layer in layers:
        if layer is None:
            continue
        canonical = parse_workflow_lora_overrides(workflow_lora_overrides_payload(layer))
        if not canonical.targets:
            targets.clear()
            continue
        for target in canonical.targets:
            slots = targets.setdefault(target.witness, {})
            for override in target.overrides:
                current = slots.get(override.slot_id)
                if current is None:
                    current = _MutableSlotOverlay(
                        loader_contract=override.loader_contract,
                        loader_authority_sha256=override.loader_authority_sha256,
                        changes={},
                    )
                    slots[override.slot_id] = current
                elif (
                    current.loader_contract != override.loader_contract
                    or current.loader_authority_sha256 != override.loader_authority_sha256
                ):
                    raise WorkflowLoraSettingsError(
                        "conflicting_workflow_lora_override_slot_evidence",
                        "Workflow LoRA setting layers disagree about one slot's public authority",
                    )
                for change in override.changes:
                    current.changes[change.field] = change.value
        if (
            len(targets) > MAX_WORKFLOW_LORA_OVERRIDE_TARGETS
            or sum(len(slots) for slots in targets.values()) > MAX_WORKFLOW_LORA_OVERRIDES_TOTAL
        ):
            raise WorkflowLoraSettingsError(
                "workflow_lora_setting_overlay_too_large",
                "Workflow LoRA setting layers exceed the durable envelope bounds",
            )

    materialized = WorkflowLoraOverrides(
        version=WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION,
        targets=tuple(
            WorkflowLoraOverrideTarget(
                witness=witness,
                overrides=tuple(
                    WorkflowLoraSlotOverride(
                        slot_id=slot_id,
                        loader_contract=slot.loader_contract,
                        loader_authority_sha256=slot.loader_authority_sha256,
                        changes=tuple(
                            WorkflowLoraFieldChange(
                                field=cast(WorkflowLoraOverrideField, field),
                                value=value,
                            )
                            for field, value in slot.changes.items()
                        ),
                    )
                    for slot_id, slot in slots.items()
                ),
            )
            for witness, slots in targets.items()
        ),
    )
    return parse_workflow_lora_overrides(workflow_lora_overrides_payload(materialized))


def workflow_lora_override_resolution_as_overrides(
    value: WorkflowLoraOverrideResolution,
) -> WorkflowLoraOverrides:
    """Reduce one effective plan to zero or one current-target envelope.

    Winning origins belong in the separately persisted public resolution
    payload.  The returned setting value contains only effective editable
    fields and never carries inactive targets into a run/work-step snapshot.
    """

    payload = workflow_lora_override_resolution_payload(value)
    raw_overrides = cast(list[dict[str, object]], payload["overrides"])
    if not raw_overrides:
        return WorkflowLoraOverrides(
            version=WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION,
            targets=(),
        )
    target = cast(dict[str, object], payload["target"])
    setting_target: dict[str, object] = {
        **target,
        "overrides": [
            {
                "slot_id": override["slot_id"],
                "loader_contract": override["loader_contract"],
                "loader_authority_sha256": override["loader_authority_sha256"],
                "changes": {
                    field: cast(dict[str, object], change)["value"]
                    for field, change in cast(
                        dict[str, object],
                        override["changes"],
                    ).items()
                },
            }
            for override in raw_overrides
        ],
    }
    return parse_workflow_lora_overrides(
        {
            "version": WORKFLOW_LORA_OVERRIDE_CONTRACT_VERSION,
            "targets": [setting_target],
        }
    )
