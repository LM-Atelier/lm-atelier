"""Build the same bounded dependency declaration during preview and completion."""

from __future__ import annotations

from typing import Any, Literal

from .workflow_dependencies import (
    MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES,
    MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS,
    MAX_WORKFLOW_DEPENDENCY_SLOTS,
    WorkflowDependencyContract,
    WorkflowDependencyError,
    canonical_workflow_dependency_json,
    parse_workflow_dependency_contract,
    validate_portable_workflow_mapping,
    workflow_dependency_contract_payload,
    workflow_dependency_slot_payload,
)

WorkflowSourceResourceKind = Literal["model_asset", "model_install", "registry_package", "runtime"]


class WorkflowSourceDependencyBuilder:
    def __init__(self, original: WorkflowDependencyContract) -> None:
        self._slots = [dict(workflow_dependency_slot_payload(slot)) for slot in original.slots]
        self._names = {slot.name for slot in original.slots}
        self._requirements = sum(len(slot.requirements) for slot in original.slots)
        self._bytes = len(canonical_workflow_dependency_json(self._declaration()))

    def _declaration(self) -> dict[str, Any]:
        return {"version": 1, "slots": self._slots}

    def include(self, kind: WorkflowSourceResourceKind, constraints: dict[str, Any]) -> None:
        if (
            len(self._slots) >= MAX_WORKFLOW_DEPENDENCY_SLOTS
            or self._requirements >= MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS
        ):
            raise WorkflowDependencyError(
                "too_many_workflow_dependencies",
                "The installed dependency declaration is too large.",
            )
        label = {
            "model_asset": "media_asset",
            "model_install": "model_files",
            "registry_package": "extension",
            "runtime": "media_runtime",
        }[kind]
        index = 1
        name = f"{label}_{index}"
        while name in self._names:
            index += 1
            name = f"{label}_{index}"
        checked = validate_portable_workflow_mapping(
            constraints, label="Installed dependency identity"
        )
        slot: dict[str, Any] = {
            "name": name,
            "resource_kind": kind,
            "required": True,
            "satisfaction": "all_of",
            "requirements": [{"key": "installed", "constraints": checked}],
        }
        size = self._bytes + len(canonical_workflow_dependency_json(slot)) + bool(self._slots)
        if size > MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES:
            raise WorkflowDependencyError(
                "dependency_data_too_large", "The installed dependency declaration is too large."
            )
        self._slots.append(slot)
        self._names.add(name)
        self._requirements += 1
        self._bytes = size

    def payload(self) -> dict[str, Any]:
        return workflow_dependency_contract_payload(
            parse_workflow_dependency_contract(self._declaration())
        )
