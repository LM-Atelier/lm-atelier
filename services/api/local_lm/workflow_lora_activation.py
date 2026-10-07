"""Load exact persisted workflow activation evidence for private LoRA execution.

This module is deliberately not part of the API projection.  Admission may
name the one current activation; replay and dispatch must name the activation
recorded with the run.  Both modes rebuild the typed dependency resolution
from persisted rows and fail closed instead of substituting another snapshot.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    WorkflowActivation,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowRevision,
)

# This private reader must accept exactly the resolver identities that the
# activation writer accepts; keeping one validator avoids a second authority.
from .workflow_activations import WorkflowActivationError, _resolver_version
from .workflow_bindings import (
    MaterializedWorkflowDependency,
    ResolvedWorkflowBinding,
    WorkflowActivationResolution,
    WorkflowBindingError,
    WorkflowBindingSelection,
    resolve_workflow_activation,
    workflow_resource_identity_sha256,
)
from .workflow_dependencies import (
    MAX_WORKFLOW_DEPENDENCY_JSON_NODES,
    WORKFLOW_DEPENDENCY_RESOURCE_KINDS,
    WorkflowDependencyContract,
    WorkflowDependencyError,
    WorkflowDependencyRequirement,
    WorkflowDependencyResourceKind,
    canonical_workflow_dependency_json,
    parse_workflow_dependency_contract,
    validate_portable_workflow_mapping,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_sha256,
)

WORKFLOW_LORA_ACTIVATION_WITNESS_VERSION = 1

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_LOCATOR_FIELDS: tuple[tuple[WorkflowDependencyResourceKind, str], ...] = (
    ("model_profile", "model_profile_id"),
    ("model_install", "model_install_id"),
    ("model_asset", "model_asset_install_id"),
    ("custom_node", "custom_node_install_id"),
    ("registry_package", "comfy_registry_install_id"),
    ("runtime", "runtime_key"),
)


@dataclass(frozen=True, slots=True)
class WorkflowLoraActivationEvidence:
    """Detached, server-private authority for one exact persisted activation."""

    activation_id: str
    resolver_version: str
    dependency_contract_sha256: str
    binding_sha256: str
    launch_sha256: str
    activation_witness_sha256: str
    _resolution_json: bytes = field(repr=False)

    @property
    def resolution(self) -> WorkflowActivationResolution:
        """Return a freshly reconstructed resolution without retaining mutable JSON."""

        return _workflow_activation_resolution_from_json(
            self._resolution_json,
            expected_binding_sha256=self.binding_sha256,
        )


def load_current_workflow_lora_activation_evidence(
    session: Session,
    workflow_revision_id: str,
) -> WorkflowLoraActivationEvidence:
    """Load the sole current activation for preview or admission."""

    revision = _revision(session, workflow_revision_id)
    with session.no_autoflush:
        rows = list(
            session.scalars(
                select(WorkflowActivation).where(
                    WorkflowActivation.workflow_revision_id == revision.id,
                    WorkflowActivation.is_active.is_(True),
                )
            ).all()
        )
    if not rows:
        raise WorkflowActivationError(
            "workflow_activation_unavailable",
            "The workflow revision has no current activation",
        )
    if len(rows) != 1:
        raise WorkflowActivationError(
            "workflow_activation_ambiguous",
            "The workflow revision has more than one current activation",
        )
    return _load_evidence(session, revision, rows[0])


def load_recorded_workflow_lora_activation_evidence(
    session: Session,
    workflow_revision_id: str,
    activation_id: str,
) -> WorkflowLoraActivationEvidence:
    """Load exactly the activation recorded by a queued run, never a fallback."""

    revision = _revision(session, workflow_revision_id)
    if type(activation_id) is not str or not activation_id or len(activation_id) > 40:
        raise WorkflowActivationError(
            "workflow_activation_unavailable",
            "The recorded workflow activation is unavailable",
        )
    with session.no_autoflush:
        activation = session.get(WorkflowActivation, activation_id)
    if activation is None:
        raise WorkflowActivationError(
            "workflow_activation_unavailable",
            "The recorded workflow activation is unavailable",
        )
    if activation.workflow_revision_id != revision.id:
        raise WorkflowActivationError(
            "workflow_activation_revision_mismatch",
            "The recorded workflow activation belongs to another revision",
        )
    return _load_evidence(session, revision, activation)


def _revision(session: Session, revision_id: str) -> WorkflowRevision:
    if type(revision_id) is not str or not revision_id or len(revision_id) > 40:
        raise WorkflowActivationError(
            "workflow_revision_unavailable",
            "The workflow revision is unavailable",
        )
    with session.no_autoflush:
        revision = session.get(WorkflowRevision, revision_id)
    if revision is None:
        raise WorkflowActivationError(
            "workflow_revision_unavailable",
            "The workflow revision is unavailable",
        )
    return revision


def _load_evidence(
    session: Session,
    revision: WorkflowRevision,
    activation: WorkflowActivation,
) -> WorkflowLoraActivationEvidence:
    contract, slots_by_id = _dependency_contract(session, revision)
    resolver_version, binding_sha256, launch_sha256 = _activation_identity(
        revision,
        activation,
    )
    resolution = _activation_resolution(
        session,
        revision,
        activation,
        contract,
        slots_by_id,
    )
    if (
        not resolution.complete
        or resolution.issues
        or resolution.missing_required_slots
        or resolution.binding_sha256 is None
    ):
        raise WorkflowActivationError(
            "workflow_activation_incomplete",
            "The recorded workflow activation dependency snapshot is incomplete",
            issues=resolution.issues,
        )
    if resolution.binding_sha256 != binding_sha256:
        raise WorkflowActivationError(
            "dependency_binding_drift",
            "The recorded workflow activation binding identity has changed",
        )
    witness = {
        "version": WORKFLOW_LORA_ACTIVATION_WITNESS_VERSION,
        "workflow_revision_id": revision.id,
        "activation_id": activation.id,
        "resolver_version": resolver_version,
        "dependency_contract_sha256": revision.dependency_contract_sha256,
        "binding_sha256": binding_sha256,
        "launch_sha256": launch_sha256,
    }
    return WorkflowLoraActivationEvidence(
        activation_id=activation.id,
        resolver_version=resolver_version,
        dependency_contract_sha256=cast(str, revision.dependency_contract_sha256),
        binding_sha256=binding_sha256,
        launch_sha256=launch_sha256,
        activation_witness_sha256=hashlib.sha256(
            canonical_workflow_dependency_json(witness)
        ).hexdigest(),
        _resolution_json=_workflow_activation_resolution_json(resolution),
    )


def _workflow_activation_resolution_json(
    resolution: WorkflowActivationResolution,
) -> bytes:
    """Seal one complete resolution as the evidence object's immutable authority."""

    if (
        not resolution.complete
        or resolution.issues
        or resolution.missing_required_slots
        or resolution.binding_sha256 is None
    ):
        raise WorkflowActivationError(
            "workflow_activation_incomplete",
            "The recorded workflow activation dependency snapshot is incomplete",
            issues=resolution.issues,
        )
    return canonical_workflow_dependency_json(
        {
            "version": 1,
            "bindings": [
                {
                    "slot_name": binding.slot_name,
                    "requirement_key": binding.requirement_key,
                    "resource_kind": binding.resource_kind,
                    "identity": binding.identity,
                    "resource_identity_sha256": binding.resource_identity_sha256,
                    "mount": binding.mount,
                }
                for binding in resolution.bindings
            ],
            "issues": [],
            "missing_required_slots": [],
            "complete": True,
            "binding_sha256": resolution.binding_sha256,
        }
    )


def _workflow_activation_resolution_from_json(
    value: bytes,
    *,
    expected_binding_sha256: str,
) -> WorkflowActivationResolution:
    """Rebuild a fresh mutable compatibility graph from immutable private bytes."""

    try:
        if type(value) is not bytes:
            raise TypeError("workflow activation evidence must be immutable bytes")
        payload: Any = json.loads(value)
        if (
            type(payload) is not dict
            or set(payload)
            != {
                "version",
                "bindings",
                "issues",
                "missing_required_slots",
                "complete",
                "binding_sha256",
            }
            or payload["version"] != 1
            or type(payload["version"]) is not int
            or type(payload["bindings"]) is not list
            or payload["issues"] != []
            or type(payload["issues"]) is not list
            or payload["missing_required_slots"] != []
            or type(payload["missing_required_slots"]) is not list
            or payload["complete"] is not True
            or type(payload["binding_sha256"]) is not str
            or payload["binding_sha256"] != expected_binding_sha256
        ):
            raise ValueError("invalid canonical workflow activation resolution")
        bindings: list[ResolvedWorkflowBinding] = []
        seen: set[tuple[str, str]] = set()
        for item in payload["bindings"]:
            if (
                type(item) is not dict
                or set(item)
                != {
                    "slot_name",
                    "requirement_key",
                    "resource_kind",
                    "identity",
                    "resource_identity_sha256",
                    "mount",
                }
                or type(item["slot_name"]) is not str
                or not item["slot_name"]
                or type(item["requirement_key"]) is not str
                or not item["requirement_key"]
                or type(item["resource_kind"]) is not str
                or item["resource_kind"] not in WORKFLOW_DEPENDENCY_RESOURCE_KINDS
                or type(item["resource_identity_sha256"]) is not str
                or _DIGEST.fullmatch(item["resource_identity_sha256"]) is None
            ):
                raise ValueError("invalid canonical workflow activation binding")
            pair = (item["slot_name"], item["requirement_key"])
            if pair in seen:
                raise ValueError("duplicate canonical workflow activation binding")
            seen.add(pair)
            resource_kind = cast(WorkflowDependencyResourceKind, item["resource_kind"])
            identity = validate_portable_workflow_mapping(
                item["identity"],
                label="workflow activation evidence identity",
            )
            mount = validate_portable_workflow_mapping(
                item["mount"],
                label="workflow activation evidence mount",
            )
            if (
                workflow_resource_identity_sha256(resource_kind, identity)
                != item["resource_identity_sha256"]
            ):
                raise ValueError("canonical workflow activation identity drift")
            bindings.append(
                ResolvedWorkflowBinding(
                    item["slot_name"],
                    item["requirement_key"],
                    resource_kind,
                    identity,
                    item["resource_identity_sha256"],
                    mount,
                )
            )
        return WorkflowActivationResolution(
            tuple(bindings),
            (),
            (),
            True,
            expected_binding_sha256,
        )
    except (
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
        WorkflowBindingError,
        WorkflowDependencyError,
    ) as exc:
        raise WorkflowActivationError(
            "invalid_activation_snapshot",
            "The workflow activation evidence is invalid",
        ) from exc


def _dependency_contract(
    session: Session,
    revision: WorkflowRevision,
) -> tuple[WorkflowDependencyContract, dict[str, WorkflowDependencySlot]]:
    if (
        type(revision.dependency_contract_sha256) is not str
        or _DIGEST.fullmatch(revision.dependency_contract_sha256) is None
    ):
        raise WorkflowActivationError(
            "invalid_workflow_dependency_snapshot",
            "The workflow revision has no valid dependency contract identity",
        )
    with session.no_autoflush:
        rows = list(
            session.scalars(
                select(WorkflowDependencySlot)
                .where(WorkflowDependencySlot.workflow_revision_id == revision.id)
                .order_by(WorkflowDependencySlot.ordinal)
            ).all()
        )
    if any(
        row.workflow_revision_id != revision.id
        or type(row.ordinal) is not int
        or type(row.requirements_json) is not list
        or not _plain_json_value(row.requirements_json)
        for row in rows
    ) or [row.ordinal for row in rows] != list(range(len(rows))):
        raise WorkflowActivationError(
            "invalid_workflow_dependency_snapshot",
            "The workflow dependency slot snapshot is malformed or unordered",
        )
    if any(
        type(requirement) is not dict or type(requirement.get("constraints")) is not dict
        for row in rows
        for requirement in row.requirements_json
    ):
        raise WorkflowActivationError(
            "invalid_workflow_dependency_snapshot",
            "The workflow dependency slot requirements are malformed",
        )
    payload = {
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
    try:
        contract = parse_workflow_dependency_contract(payload)
        contract_slots = {slot.name: slot for slot in contract.slots}
        if any(
            type(row.contract_sha256) is not str
            or _DIGEST.fullmatch(row.contract_sha256) is None
            or row.name not in contract_slots
            or row.contract_sha256 != workflow_dependency_slot_sha256(contract_slots[row.name])
            for row in rows
        ):
            raise WorkflowActivationError(
                "workflow_contract_drift",
                "A workflow dependency slot no longer matches its recorded identity",
            )
        if workflow_dependency_contract_sha256(contract) != revision.dependency_contract_sha256:
            raise WorkflowActivationError(
                "workflow_contract_drift",
                "The workflow dependency contract no longer matches its recorded identity",
            )
    except WorkflowActivationError:
        raise
    except (KeyError, RecursionError, TypeError, ValueError, WorkflowDependencyError) as exc:
        raise WorkflowActivationError(
            "invalid_workflow_dependency_snapshot",
            "The workflow dependency slot snapshot is invalid",
        ) from exc
    return contract, {row.id: row for row in rows}


def _activation_identity(
    revision: WorkflowRevision,
    activation: WorkflowActivation,
) -> tuple[str, str, str]:
    if activation.workflow_revision_id != revision.id:
        raise WorkflowActivationError(
            "workflow_activation_revision_mismatch",
            "The workflow activation belongs to another revision",
        )
    if activation.state != "ready":
        raise WorkflowActivationError(
            "workflow_activation_not_ready",
            "The workflow activation is not ready",
        )
    if (
        activation.invalidated_at is not None
        or activation.invalidation_code is not None
        or activation.invalidation_reason is not None
    ):
        raise WorkflowActivationError(
            "workflow_activation_invalidated",
            "The workflow activation has been invalidated",
        )
    try:
        resolver_version = _resolver_version(activation.resolver_version)
    except WorkflowActivationError as exc:
        raise WorkflowActivationError(
            "invalid_activation_snapshot",
            "The workflow activation resolver identity is invalid",
        ) from exc
    if (
        type(activation.dependency_contract_sha256) is not str
        or activation.dependency_contract_sha256 != revision.dependency_contract_sha256
    ):
        raise WorkflowActivationError(
            "workflow_contract_drift",
            "The workflow activation dependency contract identity has changed",
        )
    if (
        type(activation.binding_sha256) is not str
        or _DIGEST.fullmatch(activation.binding_sha256) is None
    ):
        raise WorkflowActivationError(
            "invalid_activation_snapshot",
            "The workflow activation binding identity is invalid",
        )
    if type(activation.details_json) is not dict or set(activation.details_json) != {
        "launch_sha256"
    }:
        raise WorkflowActivationError(
            "invalid_activation_snapshot",
            "The workflow activation launch identity is invalid",
        )
    launch_sha256 = activation.details_json.get("launch_sha256")
    if type(launch_sha256) is not str or _DIGEST.fullmatch(launch_sha256) is None:
        raise WorkflowActivationError(
            "invalid_activation_snapshot",
            "The workflow activation launch identity is invalid",
        )
    return resolver_version, activation.binding_sha256, launch_sha256


def _activation_resolution(
    session: Session,
    revision: WorkflowRevision,
    activation: WorkflowActivation,
    contract: WorkflowDependencyContract,
    slots_by_id: dict[str, WorkflowDependencySlot],
) -> WorkflowActivationResolution:
    with session.no_autoflush:
        rows = list(
            session.scalars(
                select(WorkflowDependencyBinding)
                .where(WorkflowDependencyBinding.workflow_activation_id == activation.id)
                .order_by(
                    WorkflowDependencyBinding.workflow_dependency_slot_id,
                    WorkflowDependencyBinding.requirement_key,
                    WorkflowDependencyBinding.id,
                )
            ).all()
        )
    selections: list[WorkflowBindingSelection] = []
    identities: dict[tuple[str, str], MaterializedWorkflowDependency] = {}
    try:
        for row in rows:
            slot = slots_by_id.get(row.workflow_dependency_slot_id)
            if (
                slot is None
                or row.workflow_revision_id != revision.id
                or row.workflow_activation_id != activation.id
                or type(row.resource_identity_json) is not dict
                or type(row.mount_json) is not dict
                or not _plain_json_value(row.resource_identity_json)
                or not _plain_json_value(row.mount_json)
            ):
                raise WorkflowActivationError(
                    "invalid_activation_snapshot",
                    "A workflow activation binding row is malformed",
                )
            local_kind, local_id = _binding_locator(row)
            identity = validate_portable_workflow_mapping(
                row.resource_identity_json,
                label="workflow activation binding identity",
            )
            mount = validate_portable_workflow_mapping(
                row.mount_json,
                label="workflow activation binding mount",
            )
            selection = WorkflowBindingSelection(
                slot.name,
                row.requirement_key,
                local_kind,
                local_id,
                row.resource_identity_sha256,
                mount,
            )
            selections.append(selection)
            identities[(slot.name, row.requirement_key)] = MaterializedWorkflowDependency(
                cast(WorkflowDependencyResourceKind, slot.resource_kind),
                identity,
            )

        def materialize(
            _requirement: WorkflowDependencyRequirement,
            selection: WorkflowBindingSelection,
        ) -> MaterializedWorkflowDependency | None:
            return identities.get((selection.slot_name, selection.requirement_key))

        return resolve_workflow_activation(contract, selections, materialize)
    except WorkflowActivationError:
        raise
    except (
        RecursionError,
        TypeError,
        ValueError,
        WorkflowBindingError,
        WorkflowDependencyError,
    ) as exc:
        code = exc.code if isinstance(exc, WorkflowBindingError) else "invalid_activation_snapshot"
        raise WorkflowActivationError(
            code,
            "The workflow activation binding snapshot is invalid",
        ) from exc


def _binding_locator(
    row: WorkflowDependencyBinding,
) -> tuple[WorkflowDependencyResourceKind, str]:
    present = [
        (kind, value)
        for kind, field in _LOCATOR_FIELDS
        if (value := getattr(row, field)) is not None
    ]
    if len(present) != 1:
        raise WorkflowActivationError(
            "invalid_activation_snapshot",
            "A workflow activation binding locator is invalid",
        )
    kind, value = present[0]
    if (
        type(value) is not str
        or not value
        or len(value) > 200
        or any(character < " " or ord(character) == 127 for character in value)
    ):
        raise WorkflowActivationError(
            "invalid_activation_snapshot",
            "A workflow activation binding locator is invalid",
        )
    return kind, value


def _plain_json_value(value: object) -> bool:
    """Reject executable container subclasses before canonical validators run."""

    stack = [value]
    seen_containers: set[int] = set()
    visited = 0
    while stack:
        current = stack.pop()
        visited += 1
        if visited > MAX_WORKFLOW_DEPENDENCY_JSON_NODES:
            return False
        if type(current) is dict:
            identity = id(current)
            if identity in seen_containers or any(type(key) is not str for key in current):
                return False
            seen_containers.add(identity)
            stack.extend(current.values())
        elif type(current) is list:
            identity = id(current)
            if identity in seen_containers:
                return False
            seen_containers.add(identity)
            stack.extend(current)
        elif current is None or type(current) in {str, int, float, bool}:
            continue
        else:
            return False
    return True
