"""Rebuild workflow-native LoRA execution authority and replay receipts.

Projection may use the current activation to describe editable controls.  A
queued run is different: it must name its recorded activation and rebuild all
private graph targets from persisted evidence.  The public replay receipt is
only an integrity claim.  It contains no graph bytes, node locators, or local
paths and cannot authorize mutation without this server-private pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy.orm import Session

from .comfy_package_widgets import POWER_LORA_LOADER
from .models import WorkflowRevision
from .workflow_activations import WorkflowActivationError
from .workflow_lora_activation import load_recorded_workflow_lora_activation_evidence
from .workflow_lora_composition import (
    WORKFLOW_LORA_COMPOSITION_VERSION,
    WorkflowLoraComposition,
    workflow_lora_composition_added_settings,
    workflow_lora_composition_graph,
    workflow_lora_composition_payload,
    workflow_lora_composition_sha256,
)
from .workflow_lora_overrides import (
    WorkflowLoraOverrideCatalog,
    WorkflowLoraOverrideError,
    WorkflowLoraOverrideResolution,
    parse_workflow_lora_override_resolution,
    workflow_lora_override_resolution_payload,
    workflow_lora_override_resolution_sha256,
)
from .workflow_lora_slots import (
    WorkflowLoraCoreEvidence,
    WorkflowLoraPackageEvidence,
    WorkflowLoraSlotError,
    WorkflowLoraSlotExtractionWithPrivateTargets,
    extract_workflow_lora_slots,
    extract_workflow_lora_slots_with_private_targets,
)
from .workflow_loras import (
    WorkflowLoraEvidenceGap,
    _core_evidence,
    _expanded_ui_graph,
    _load_contract,
    _loader_types,
    _override_target,
    _package_evidence,
)

WORKFLOW_LORA_REPLAY_VERSION = 1
MAX_WORKFLOW_LORA_REPLAY_CANONICAL_BYTES = 512 * 1024
MAX_WORKFLOW_LORA_REPLAY_JSON_NODES = 65_536
MAX_WORKFLOW_LORA_REPLAY_JSON_DEPTH = 128

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CORE_LOADERS = frozenset({"LoraLoader", "LoraLoaderModelOnly"})
_COMPOSITION_FIELDS = frozenset(
    {
        "version",
        "source_api_graph_sha256",
        "workflow_effective_graph_sha256",
        "effective_graph_sha256",
        "workflow_override_resolution_sha256",
        "workflow_graph_resolution_sha256",
        "workflow_native",
        "added_transform_version",
        "added_settings",
        "added_provenance",
    }
)


class WorkflowLoraExecutionError(ValueError):
    """A typed refusal to derive or replay workflow-native LoRA execution."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def detach_workflow_lora_execution_json_object(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    """Copy one untrusted run snapshot through an exact inert JSON boundary."""

    try:
        encoded = _canonical_json(value, label=label)
        detached: Any = json.loads(encoded)
        if type(detached) is not dict:
            raise ValueError(f"{label} is not an object")
        return cast(dict[str, Any], detached)
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraExecutionError(
            "invalid_workflow_lora_execution_json",
            f"The {label} snapshot is malformed",
        ) from exc


@dataclass(frozen=True, slots=True)
class WorkflowLoraExecutionAuthority:
    """Exact recorded-activation authority with private targets kept hidden."""

    activation_id: str
    resolver_version: str
    dependency_contract_sha256: str
    binding_sha256: str
    launch_sha256: str
    activation_witness_sha256: str
    catalog: WorkflowLoraOverrideCatalog
    slot_extraction: WorkflowLoraSlotExtractionWithPrivateTargets = field(repr=False)


@dataclass(frozen=True, slots=True)
class WorkflowLoraReplayReceipt:
    """Detached persisted receipt; never standalone graph-edit authority."""

    version: int
    override_resolution_sha256: str
    composition_sha256: str
    override_resolution: WorkflowLoraOverrideResolution = field(repr=False)
    _composition_json: bytes = field(repr=False)

    @property
    def composition_payload(self) -> dict[str, object]:
        """Return a fresh exact-JSON copy of the recorded public receipt."""

        value: Any = json.loads(self._composition_json)
        if type(value) is not dict:
            raise WorkflowLoraExecutionError(
                "invalid_workflow_lora_replay_receipt",
                "Workflow LoRA replay evidence is malformed",
            )
        return cast(dict[str, object], value)


def build_workflow_lora_execution_authority(
    session: Session,
    revision: WorkflowRevision,
    *,
    activation_id: str,
) -> WorkflowLoraExecutionAuthority:
    """Derive private edit targets from one supplied recorded activation only."""

    try:
        if (
            type(revision) is not WorkflowRevision
            or type(revision.id) is not str
            or not revision.id
            or type(revision.workflow_id) is not str
            or not revision.workflow_id
            or type(revision.api_graph_json) is not dict
            or type(revision.dependency_contract_sha256) is not str
            or _DIGEST.fullmatch(revision.dependency_contract_sha256) is None
        ):
            raise WorkflowLoraExecutionError(
                "invalid_workflow_lora_execution_revision",
                "The workflow revision cannot authorize workflow LoRA execution",
            )
        activation = load_recorded_workflow_lora_activation_evidence(
            session,
            revision.id,
            activation_id,
        )
        gaps: set[WorkflowLoraEvidenceGap] = set()
        contract_snapshot = _load_contract(session, revision, gaps)
        if (
            not contract_snapshot.verified
            or gaps
            or activation.dependency_contract_sha256 != revision.dependency_contract_sha256
        ):
            raise WorkflowLoraExecutionError(
                "invalid_workflow_lora_execution_contract",
                "The recorded workflow dependency contract cannot authorize LoRA execution",
            )

        resolution = activation.resolution
        detected = extract_workflow_lora_slots(
            revision_scope=revision.id,
            api_graph=revision.api_graph_json,
            dependency_contract=contract_snapshot.contract,
            activation=resolution,
        )
        core_evidence: tuple[WorkflowLoraCoreEvidence, ...] = ()
        package_evidence: tuple[WorkflowLoraPackageEvidence, ...] = ()
        loader_types = _loader_types(revision.api_graph_json)
        if loader_types:
            ui_graph = (
                _expanded_ui_graph(revision.ui_graph_json) if revision.engine == "comfyui" else None
            )
            if ui_graph is not None:
                if loader_types & _CORE_LOADERS:
                    core_evidence = _core_evidence(
                        revision.api_graph_json,
                        ui_graph,
                        resolution.bindings,
                        detected.api_graph_sha256,
                        gaps,
                    )
                if POWER_LORA_LOADER in loader_types:
                    package_evidence = _package_evidence(
                        revision.api_graph_json,
                        ui_graph,
                        resolution.bindings,
                        detected.api_graph_sha256,
                        gaps,
                    )

        extraction = extract_workflow_lora_slots_with_private_targets(
            revision_scope=revision.id,
            api_graph=revision.api_graph_json,
            dependency_contract=contract_snapshot.contract,
            activation=resolution,
            core_evidence=core_evidence,
            package_evidence=package_evidence,
        )
        target = _override_target(session, revision, activation, extraction)
        if target is None:
            raise WorkflowLoraExecutionError(
                "workflow_lora_execution_authority_unavailable",
                "The recorded workflow activation cannot authorize LoRA execution",
            )
        catalog = WorkflowLoraOverrideCatalog(target, extraction.public.slots)
        return WorkflowLoraExecutionAuthority(
            activation_id=activation.activation_id,
            resolver_version=activation.resolver_version,
            dependency_contract_sha256=activation.dependency_contract_sha256,
            binding_sha256=activation.binding_sha256,
            launch_sha256=activation.launch_sha256,
            activation_witness_sha256=activation.activation_witness_sha256,
            catalog=catalog,
            slot_extraction=extraction,
        )
    except WorkflowLoraExecutionError:
        raise
    except (WorkflowActivationError, WorkflowLoraOverrideError, WorkflowLoraSlotError) as exc:
        raise WorkflowLoraExecutionError(
            "workflow_lora_execution_authority_unavailable",
            "The recorded workflow activation cannot authorize LoRA execution",
        ) from exc


def workflow_lora_replay_payload(
    composition: WorkflowLoraComposition,
    override_resolution: WorkflowLoraOverrideResolution,
) -> dict[str, object]:
    """Encode public-safe replay integrity without graph bytes or private locators."""

    composition_payload = workflow_lora_composition_payload(composition)
    resolution_payload = workflow_lora_override_resolution_payload(override_resolution)
    native = composition_payload.get("workflow_native")
    if (
        type(native) is not dict
        or _canonical_json(native.get("override_resolution"), label="override resolution")
        != _canonical_json(resolution_payload, label="override resolution")
        or composition_payload.get("workflow_override_resolution_sha256")
        != workflow_lora_override_resolution_sha256(override_resolution)
    ):
        raise WorkflowLoraExecutionError(
            "invalid_workflow_lora_replay_receipt",
            "Workflow LoRA replay evidence is malformed",
        )
    payload: dict[str, object] = {
        "version": WORKFLOW_LORA_REPLAY_VERSION,
        "override_resolution": resolution_payload,
        "override_resolution_sha256": workflow_lora_override_resolution_sha256(override_resolution),
        "composition": composition_payload,
        "composition_sha256": workflow_lora_composition_sha256(composition),
    }
    parsed = parse_workflow_lora_replay_receipt(payload)
    return {
        "version": parsed.version,
        "override_resolution": workflow_lora_override_resolution_payload(
            parsed.override_resolution
        ),
        "override_resolution_sha256": parsed.override_resolution_sha256,
        "composition": parsed.composition_payload,
        "composition_sha256": parsed.composition_sha256,
    }


def parse_workflow_lora_replay_receipt(value: object) -> WorkflowLoraReplayReceipt:
    """Strictly parse a frozen full resolution plus public composition receipt."""

    try:
        encoded = _canonical_json(value, label="replay receipt")
        root_value: Any = json.loads(encoded)
        if type(root_value) is not dict:
            raise ValueError("receipt is not an object")
        root = cast(dict[str, object], root_value)
        if set(root) != {
            "version",
            "override_resolution",
            "override_resolution_sha256",
            "composition",
            "composition_sha256",
        }:
            raise ValueError("receipt keys are invalid")
        if type(root.get("version")) is not int or root["version"] != WORKFLOW_LORA_REPLAY_VERSION:
            raise ValueError("receipt version is invalid")
        digest = root.get("composition_sha256")
        if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
            raise ValueError("receipt digest is invalid")

        raw_resolution = root.get("override_resolution")
        resolution = parse_workflow_lora_override_resolution(raw_resolution)
        canonical_resolution = workflow_lora_override_resolution_payload(resolution)
        if _canonical_json(raw_resolution, label="override resolution") != _canonical_json(
            canonical_resolution,
            label="override resolution",
        ):
            raise ValueError("override resolution is not canonical")
        resolution_digest = root.get("override_resolution_sha256")
        if (
            type(resolution_digest) is not str
            or _DIGEST.fullmatch(resolution_digest) is None
            or resolution_digest != workflow_lora_override_resolution_sha256(resolution)
        ):
            raise ValueError("override resolution digest changed")

        raw_composition = root.get("composition")
        composition_json = _canonical_json(raw_composition, label="composition receipt")
        composition_value: Any = json.loads(composition_json)
        if type(composition_value) is not dict:
            raise ValueError("composition receipt is not an object")
        composition = cast(dict[str, object], composition_value)
        _validate_composition_receipt(composition, canonical_resolution)
        if hashlib.sha256(composition_json).hexdigest() != digest:
            raise ValueError("composition receipt digest changed")
        return WorkflowLoraReplayReceipt(
            version=WORKFLOW_LORA_REPLAY_VERSION,
            override_resolution_sha256=resolution_digest,
            composition_sha256=digest,
            override_resolution=resolution,
            _composition_json=composition_json,
        )
    except WorkflowLoraExecutionError:
        raise
    except (
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
        WorkflowLoraOverrideError,
    ) as exc:
        raise WorkflowLoraExecutionError(
            "invalid_workflow_lora_replay_receipt",
            "Workflow LoRA replay evidence is malformed",
        ) from exc


def verify_workflow_lora_replay_composition(
    receipt: WorkflowLoraReplayReceipt,
    composition: WorkflowLoraComposition,
    *,
    stored_added_loras: object,
) -> dict[str, Any]:
    """Require both the full public payload and its digest before graph dispatch."""

    try:
        if type(receipt) is not WorkflowLoraReplayReceipt:
            raise TypeError("replay receipt is not typed")
        normalized_added = workflow_lora_composition_added_settings(composition)
        if _canonical_json(
            stored_added_loras,
            label="stored Added settings",
        ) != _canonical_json(normalized_added, label="normalized Added settings"):
            raise ValueError("stored Added settings are not canonical")
        recomposed_payload = workflow_lora_composition_payload(composition)
        recomposed_json = _canonical_json(recomposed_payload, label="recomposed receipt")
        if (
            recomposed_json != receipt._composition_json
            or workflow_lora_composition_sha256(composition) != receipt.composition_sha256
        ):
            raise ValueError("recomposed receipt changed")
        return workflow_lora_composition_graph(composition)
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowLoraExecutionError(
            "stale_workflow_lora_replay_composition",
            "The queued workflow LoRA composition no longer matches its receipt",
        ) from exc


def _validate_composition_receipt(
    composition: dict[str, object],
    resolution_payload: dict[str, object],
) -> None:
    if set(composition) != _COMPOSITION_FIELDS:
        raise ValueError("composition receipt keys are invalid")
    if (
        type(composition.get("version")) is not int
        or composition["version"] != WORKFLOW_LORA_COMPOSITION_VERSION
        or not _is_digest(composition.get("source_api_graph_sha256"))
        or not _is_digest(composition.get("workflow_effective_graph_sha256"))
        or not _is_digest(composition.get("effective_graph_sha256"))
        or not _is_digest(composition.get("workflow_override_resolution_sha256"))
        or not _is_digest(composition.get("workflow_graph_resolution_sha256"))
        or type(composition.get("added_transform_version")) is not str
        or not composition["added_transform_version"]
        or type(composition.get("added_settings")) is not list
        or type(composition.get("added_provenance")) is not list
    ):
        raise ValueError("composition receipt fields are invalid")
    native = composition.get("workflow_native")
    if type(native) is not dict or set(native) != {
        "override_resolution",
        "graph_resolution",
    }:
        raise ValueError("composition native receipt is invalid")
    if type(native.get("graph_resolution")) is not dict:
        raise ValueError("composition graph receipt is invalid")
    if _canonical_json(native.get("override_resolution"), label="native resolution") != (
        _canonical_json(resolution_payload, label="override resolution")
    ):
        raise ValueError("composition resolution changed")
    if composition[
        "workflow_override_resolution_sha256"
    ] != workflow_lora_override_resolution_sha256(
        parse_workflow_lora_override_resolution(resolution_payload)
    ):
        raise ValueError("composition resolution digest changed")


def _canonical_json(value: object, *, label: str) -> bytes:
    nodes = 0
    seen_containers: set[int] = set()

    def detach(item: object, depth: int) -> object:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_WORKFLOW_LORA_REPLAY_JSON_NODES:
            raise ValueError(f"{label} has too many values")
        if depth > MAX_WORKFLOW_LORA_REPLAY_JSON_DEPTH:
            raise ValueError(f"{label} is too deeply nested")
        if item is None or type(item) in {str, bool}:
            return item
        if type(item) is int:
            if not -(2**63) <= item < 2**63:
                raise ValueError(f"{label} has an invalid integer")
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{label} has a non-finite number")
            return item
        if type(item) not in {dict, list}:
            raise TypeError(f"{label} must use exact JSON built-ins")
        identity = id(item)
        if identity in seen_containers:
            raise ValueError(f"{label} contains a shared or cyclic container")
        seen_containers.add(identity)
        if type(item) is list:
            return [detach(child, depth + 1) for child in cast(list[object], item)]
        raw = cast(dict[object, object], item)
        if any(type(key) is not str for key in raw):
            raise TypeError(f"{label} has a non-string key")
        return {cast(str, key): detach(child, depth + 1) for key, child in raw.items()}

    detached = detach(value, 0)
    try:
        encoded = json.dumps(
            detached,
            # This must byte-match workflow_lora_composition._canonical_json:
            # public receipt digests are already defined with escaped Unicode.
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if len(encoded) > MAX_WORKFLOW_LORA_REPLAY_CANONICAL_BYTES:
        raise ValueError(f"{label} is too large")
    return encoded


def _is_digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None
