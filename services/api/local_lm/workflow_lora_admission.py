"""Atomically admit workflow-native and Added LoRAs for one queued media step.

The settings hierarchy carries a reserved public envelope beside ordinary
generation settings.  Admission removes that envelope from each of the eight
request-setting layers before generic validation, preserves field origins,
binds the effective plan to the exact activation snapshot selected for the
step, and composes native edits before the existing Added/Auto stack.

Only the canonical current-target envelope and public integrity receipt leave
this module.  Private graph targets and effective graph bytes remain in-process
authority and are re-derived again at dispatch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy.orm import Session

from .models import WorkflowRevision
from .workflow_lora_composition import (
    WorkflowLoraComposition,
    WorkflowLoraCompositionError,
    compose_workflow_lora_graph,
)
from .workflow_lora_execution import (
    WorkflowLoraExecutionError,
    build_workflow_lora_execution_authority,
    workflow_lora_replay_payload,
)
from .workflow_lora_graph import WorkflowLoraGraphError
from .workflow_lora_overrides import (
    WORKFLOW_LORA_OVERRIDE_ORIGINS,
    WorkflowLoraOverrideError,
    WorkflowLoraOverrideLayer,
    WorkflowLoraOverrideOrigin,
    WorkflowLoraOverrideResolution,
    parse_workflow_lora_overrides,
    resolve_workflow_lora_override_layers,
)
from .workflow_lora_settings import (
    WorkflowLoraSettingsError,
    detach_workflow_lora_overrides_setting,
    split_workflow_lora_overrides_setting,
    workflow_lora_override_resolution_as_overrides,
    workflow_lora_overrides_setting_value,
)

WORKFLOW_LORA_ADMISSION_INVALID_CODE = "workflow-lora-settings-invalid"
WORKFLOW_LORA_ADMISSION_INVALID_MESSAGE = "The workflow LoRA settings are invalid."
WORKFLOW_LORA_ADMISSION_CONFLICT_CODE = "workflow-lora-settings-stale"
WORKFLOW_LORA_ADMISSION_CONFLICT_MESSAGE = (
    "The workflow LoRA settings no longer match the selected workflow."
)
WORKFLOW_LORA_REPLAY_CONFLICT_CODE = "workflow-lora-replay-stale"
WORKFLOW_LORA_REPLAY_CONFLICT_MESSAGE = (
    "The queued workflow LoRA configuration could not be verified."
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ORIGINS = WORKFLOW_LORA_OVERRIDE_ORIGINS
_STALE_OVERRIDE_CODES = frozenset(
    {"stale_workflow_lora_override_slot", "stale_workflow_lora_override_target"}
)


class WorkflowLoraAdmissionError(ValueError):
    """A fixed public refusal for user-invalid or stale admission evidence."""

    def __init__(self, *, conflict: bool, replay: bool = False) -> None:
        self.conflict = conflict
        self.code = (
            WORKFLOW_LORA_REPLAY_CONFLICT_CODE
            if replay
            else (
                WORKFLOW_LORA_ADMISSION_CONFLICT_CODE
                if conflict
                else WORKFLOW_LORA_ADMISSION_INVALID_CODE
            )
        )
        super().__init__(
            WORKFLOW_LORA_REPLAY_CONFLICT_MESSAGE
            if replay
            else WORKFLOW_LORA_ADMISSION_CONFLICT_MESSAGE
            if conflict
            else WORKFLOW_LORA_ADMISSION_INVALID_MESSAGE
        )


@dataclass(frozen=True, slots=True)
class WorkflowLoraAdmissionLayers:
    """The eight native layers after removing their reserved values.

    ``override_layers`` holds only the envelopes above the highest valid reset,
    in precedence order. Envelopes below that reset were never parsed, so a
    reset can hide stale or corrupt lower state without reviving it.
    """

    role: str
    override_layers: tuple[WorkflowLoraOverrideLayer, ...] = field(repr=False)
    _ordinary: tuple[dict[str, Any], ...] = field(repr=False)

    def ordinary(self, origin: str) -> dict[str, Any]:
        """Return the caller-owned ordinary settings for one closed origin."""

        if origin not in _ORIGINS:
            raise WorkflowLoraAdmissionError(conflict=False)
        return self._ordinary[_ORIGINS.index(origin)]

    def relevant_to(self, revision: WorkflowRevision | None) -> bool:
        """Whether any surviving target names exactly this workflow revision.

        Targets for another workflow or revision stay inactive evidence and do
        not make native admission necessary; neither does an empty result.
        """

        if revision is None:
            return False
        return any(
            target.witness.workflow_definition_id == revision.workflow_id
            and target.witness.workflow_revision_id == revision.id
            for layer in self.override_layers
            for target in layer.overrides.targets
        )


@dataclass(frozen=True, slots=True)
class WorkflowLoraAdmission:
    """Typed in-process composition plus its public effective native plan."""

    resolution: WorkflowLoraOverrideResolution = field(repr=False)
    composition: WorkflowLoraComposition = field(repr=False)


def split_workflow_lora_admission_layers(
    *,
    role: str,
    profile_request: Mapping[str, Any] | None,
    default_preset: Mapping[str, Any] | None,
    project_preset: Mapping[str, Any] | None,
    project: Mapping[str, Any] | None,
    chat_preset: Mapping[str, Any] | None,
    chat: Mapping[str, Any] | None,
    turn_preset: Mapping[str, Any] | None,
    turn: Mapping[str, Any] | None,
) -> WorkflowLoraAdmissionLayers:
    """Split exactly the eight native layers, never the profile-load layer.

    Every layer's ordinary settings are detached first. Reserved envelopes are
    then parsed from the highest precedence down, stopping at the first valid
    empty reset. A failure in the turn's own settings is invalid input; a
    failure in any stored layer is stale saved state.
    """

    raw_layers = (
        profile_request,
        default_preset,
        project_preset,
        project,
        chat_preset,
        chat,
        turn_preset,
        turn,
    )
    ordinary: list[dict[str, Any]] = []
    present: list[tuple[WorkflowLoraOverrideOrigin, object]] = []
    for origin, raw in zip(_ORIGINS, raw_layers, strict=True):
        try:
            detached = detach_workflow_lora_overrides_setting(raw, role=role)
        except WorkflowLoraSettingsError as exc:
            raise WorkflowLoraAdmissionError(conflict=origin != "turn") from exc
        ordinary.append(detached.ordinary)
        if detached.present:
            present.append((origin, detached.raw))

    surviving: list[WorkflowLoraOverrideLayer] = []
    for origin, raw_envelope in reversed(present):
        try:
            parsed = parse_workflow_lora_overrides(raw_envelope)
        except WorkflowLoraOverrideError as exc:
            raise WorkflowLoraAdmissionError(conflict=origin != "turn") from exc
        if not parsed.targets:
            break
        surviving.append(WorkflowLoraOverrideLayer(origin, parsed))
    surviving.reverse()
    return WorkflowLoraAdmissionLayers(
        role=role,
        override_layers=tuple(surviving),
        _ordinary=tuple(ordinary),
    )


def split_inherited_workflow_lora_setting(
    settings: Mapping[str, Any] | None,
    *,
    role: str,
) -> tuple[dict[str, Any], dict[str, object] | None]:
    """Detach one persisted envelope so ordinary compatibility filtering cannot drop it."""

    try:
        overrides, ordinary = split_workflow_lora_overrides_setting(settings, role=role)
        canonical = (
            workflow_lora_overrides_setting_value(overrides) if overrides is not None else None
        )
        return ordinary, canonical
    except (WorkflowLoraOverrideError, WorkflowLoraSettingsError) as exc:
        raise WorkflowLoraAdmissionError(conflict=True) from exc


def admit_workflow_lora_composition(
    session: Session,
    revision: WorkflowRevision,
    *,
    activation_snapshot: object,
    layers: WorkflowLoraAdmissionLayers,
    added_loras: object,
) -> WorkflowLoraAdmission:
    """Bind, resolve, and compose one native-present media settings hierarchy."""

    try:
        if type(layers) is not WorkflowLoraAdmissionLayers or not layers.relevant_to(revision):
            raise WorkflowLoraAdmissionError(conflict=False)
        activation = _activation_snapshot(activation_snapshot)
        authority = build_workflow_lora_execution_authority(
            session,
            revision,
            activation_id=activation["id"],
        )
        if any(
            getattr(authority, authority_field) != activation[snapshot_field]
            for authority_field, snapshot_field in (
                ("activation_id", "id"),
                ("resolver_version", "resolver_version"),
                ("dependency_contract_sha256", "dependency_contract_sha256"),
                ("binding_sha256", "binding_sha256"),
                ("launch_sha256", "launch_sha256"),
            )
        ):
            raise WorkflowLoraAdmissionError(conflict=True)
        for layer in layers.override_layers:
            # Resolved alone first, so a refusal names the layer it came from:
            # the turn's own request is invalid input, a stored layer is stale.
            try:
                resolve_workflow_lora_override_layers(
                    catalog=authority.catalog,
                    layers=(layer,),
                )
            except WorkflowLoraOverrideError as exc:
                raise WorkflowLoraAdmissionError(
                    conflict=layer.origin != "turn" or exc.code in _STALE_OVERRIDE_CODES
                ) from exc
        resolution = resolve_workflow_lora_override_layers(
            catalog=authority.catalog,
            layers=layers.override_layers,
        )
        composition = compose_workflow_lora_graph(
            session,
            revision,
            added_loras=added_loras,
            workflow_activation_id=activation["id"],
            override_catalog=authority.catalog,
            slot_extraction=authority.slot_extraction,
            override_resolution=resolution,
        )
        return WorkflowLoraAdmission(resolution=resolution, composition=composition)
    except WorkflowLoraAdmissionError:
        raise
    except WorkflowLoraSettingsError as exc:
        raise WorkflowLoraAdmissionError(conflict=False) from exc
    except WorkflowLoraOverrideError as exc:
        raise WorkflowLoraAdmissionError(conflict=exc.code in _STALE_OVERRIDE_CODES) from exc
    except WorkflowLoraCompositionError as exc:
        raise WorkflowLoraAdmissionError(
            conflict=exc.code != "invalid_added_lora_composition"
        ) from exc
    except (WorkflowLoraExecutionError, WorkflowLoraGraphError) as exc:
        raise WorkflowLoraAdmissionError(conflict=True) from exc


def workflow_lora_admission_setting_value(
    value: WorkflowLoraAdmission,
) -> dict[str, object]:
    """Return the canonical zero/one current-target envelope for persistence."""

    if type(value) is not WorkflowLoraAdmission:
        raise WorkflowLoraAdmissionError(conflict=True)
    try:
        effective = workflow_lora_override_resolution_as_overrides(value.resolution)
        return workflow_lora_overrides_setting_value(effective)
    except (WorkflowLoraOverrideError, WorkflowLoraSettingsError) as exc:
        raise WorkflowLoraAdmissionError(conflict=True) from exc


def workflow_lora_admission_provenance(
    value: WorkflowLoraAdmission,
) -> dict[str, object]:
    """Return the versioned public receipt used for exact dispatch replay."""

    if type(value) is not WorkflowLoraAdmission:
        raise WorkflowLoraAdmissionError(conflict=True)
    try:
        return workflow_lora_replay_payload(value.composition, value.resolution)
    except (WorkflowLoraExecutionError, WorkflowLoraOverrideError) as exc:
        raise WorkflowLoraAdmissionError(conflict=True) from exc


def _activation_snapshot(value: object) -> dict[str, str]:
    required = {
        "id",
        "resolver_version",
        "dependency_contract_sha256",
        "binding_sha256",
        "launch_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != required
        or any(type(value.get(key)) is not str or not value[key] for key in required)
        or any(
            _DIGEST.fullmatch(value[key]) is None
            for key in {
                "dependency_contract_sha256",
                "binding_sha256",
                "launch_sha256",
            }
        )
    ):
        raise WorkflowLoraAdmissionError(conflict=True)
    return cast(dict[str, str], value)
