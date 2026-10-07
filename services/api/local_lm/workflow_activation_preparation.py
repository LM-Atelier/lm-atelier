"""Prepare installed dependency choices without changing trust or activation state."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    ComfyRegistryInstall,
    CustomNodeInstall,
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
)
from .schemas import ApiModel
from .workflow_activation_requests import (
    WorkflowActivationSelectionIn,
    activation_subject,
)
from .workflow_activations import (
    WorkflowActivationError,
    WorkflowRuntimeMaterializer,
    resolve_workflow_dependencies,
)
from .workflow_bindings import WorkflowBindingSelection
from .workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyResourceKind,
    parse_workflow_dependency_contract,
)


class WorkflowActivationChoice(ApiModel):
    name: str
    selection: WorkflowActivationSelectionIn


class WorkflowActivationSlotChoices(ApiModel):
    name: str
    resource_kind: WorkflowDependencyResourceKind
    required: bool
    satisfaction: Literal["all_of", "any_of"]
    requirement_keys: list[str]
    choices: list[WorkflowActivationChoice]


class WorkflowActivationPreparationIssue(ApiModel):
    code: Literal["missing_required_dependency", "ambiguous_dependency_binding"]
    slot_name: str


class WorkflowActivationPreparation(ApiModel):
    workflow_revision_id: str
    workflow_artifact_sha256: str
    dependency_contract_sha256: str
    state: Literal["prepared", "needs_attention"]
    selections: list[WorkflowActivationSelectionIn] | None
    slots: list[WorkflowActivationSlotChoices]
    issues: list[WorkflowActivationPreparationIssue]


class WorkflowDependencyChoices(ApiModel):
    state: Literal["prepared", "needs_attention"]
    selections: list[WorkflowActivationSelectionIn] | None
    slots: list[WorkflowActivationSlotChoices]
    issues: list[WorkflowActivationPreparationIssue]


def _installed_candidates(
    session: Session, kind: WorkflowDependencyResourceKind
) -> tuple[tuple[str, str], ...]:
    # These are locators and labels only. The shared activation resolver decides
    # whether each resource still has a valid identity matching the declaration.
    if kind == "runtime":
        return (("comfyui", "ComfyUI"),)
    if kind == "model_install":
        return tuple(
            session.execute(
                select(ModelInstall.id, ModelInstall.name).order_by(ModelInstall.id)
            ).tuples()
        )
    if kind == "model_profile":
        return tuple(
            session.execute(
                select(ModelProfile.id, ModelProfile.name).order_by(ModelProfile.id)
            ).tuples()
        )
    if kind == "model_asset":
        return tuple(
            session.execute(
                select(ModelAssetInstall.id, ModelAssetInstall.name).order_by(ModelAssetInstall.id)
            ).tuples()
        )
    if kind == "custom_node":
        return tuple(
            session.execute(
                select(CustomNodeInstall.id, CustomNodeInstall.name).order_by(CustomNodeInstall.id)
            ).tuples()
        )
    return tuple(
        (row.id, f"{row.package_id} {row.package_version}")
        for row in session.scalars(select(ComfyRegistryInstall).order_by(ComfyRegistryInstall.id))
    )


def prepare_workflow_activation(
    session: Session,
    workflow_id: str,
    revision_id: str,
    *,
    runtime_materializer: WorkflowRuntimeMaterializer | None = None,
    accepted_resources: frozenset[tuple[str, str]] = frozenset(),
) -> WorkflowActivationPreparation:
    """Offer matching installed resources; final activation still verifies them.

    Optional slots start disabled unless an accepted install supplies them.
    Verified accepted resources take precedence over other compatible resources.
    All-of slots still need one unique match per requirement; any-of slots need
    exactly one choice across alternatives. Remaining ambiguity requires a choice.
    """
    with session.no_autoflush:
        subject = activation_subject(session, workflow_id, revision_id)
        contract = parse_workflow_dependency_contract({"version": 1, "slots": subject.slots})
        choices = prepare_workflow_dependency_choices(
            session,
            contract,
            runtime_materializer=runtime_materializer,
            accepted_resources=accepted_resources,
        )
        if activation_subject(session, workflow_id, revision_id) != subject:
            raise WorkflowActivationError(
                "workflow_contract_drift", "Workflow content changed during preparation"
            )
        return WorkflowActivationPreparation(
            workflow_revision_id=subject.workflow_revision_id,
            workflow_artifact_sha256=subject.workflow_artifact_sha256,
            dependency_contract_sha256=subject.dependency_contract_sha256,
            state=choices.state,
            selections=choices.selections,
            slots=choices.slots,
            issues=choices.issues,
        )


def prepare_workflow_dependency_choices(
    session: Session,
    contract: WorkflowDependencyContract,
    *,
    runtime_materializer: WorkflowRuntimeMaterializer | None = None,
    accepted_resources: frozenset[tuple[str, str]] = frozenset(),
) -> WorkflowDependencyChoices:
    """Resolve declared choices without granting permission to run a graph or its code."""
    with session.no_autoflush:
        inventory = {
            kind: _installed_candidates(session, kind)
            for kind in sorted({slot.resource_kind for slot in contract.slots})
        }
        slots: list[WorkflowActivationSlotChoices] = []
        selected: list[WorkflowActivationSelectionIn] = []
        issues: list[WorkflowActivationPreparationIssue] = []
        for slot in contract.slots:
            choices: list[WorkflowActivationChoice] = []
            by_requirement: dict[str, list[WorkflowActivationChoice]] = {}
            for requirement in slot.requirements:
                # A one-requirement probe uses exactly the same constraint and
                # identity validation as activation. It makes no claim that the
                # complete slot, launch scope, or physical files are ready.
                probe = WorkflowDependencyContract(
                    version=contract.version,
                    slots=(replace(slot, required=True, requirements=(requirement,)),),
                )
                matches: list[WorkflowActivationChoice] = []
                for local_id, name in inventory[slot.resource_kind]:
                    selection = WorkflowBindingSelection(
                        slot.name, requirement.key, slot.resource_kind, local_id
                    )
                    resolution = resolve_workflow_dependencies(
                        session, probe, [selection], runtime_materializer=runtime_materializer
                    )
                    if not resolution.complete or len(resolution.bindings) != 1:
                        continue
                    binding = resolution.bindings[0]
                    matches.append(
                        WorkflowActivationChoice(
                            name=name,
                            selection=WorkflowActivationSelectionIn(
                                slot_name=slot.name,
                                requirement_key=requirement.key,
                                local_kind=slot.resource_kind,
                                local_id=local_id,
                                recorded_resource_identity_sha256=binding.resource_identity_sha256,
                            ),
                        )
                    )
                by_requirement[requirement.key] = matches
                choices.extend(matches)
            slots.append(
                WorkflowActivationSlotChoices(
                    name=slot.name,
                    resource_kind=slot.resource_kind,
                    required=slot.required,
                    satisfaction=slot.satisfaction,
                    requirement_keys=[requirement.key for requirement in slot.requirements],
                    choices=choices,
                )
            )
            supplied = [
                choice
                for choice in choices
                if (choice.selection.local_kind, choice.selection.local_id) in accepted_resources
            ]
            if not slot.required and not supplied:
                continue
            groups = [choices] if slot.satisfaction == "any_of" else list(by_requirement.values())
            groups = [
                [choice for choice in group if choice in supplied] or group for group in groups
            ]
            if any(not group for group in groups):
                issues.append(
                    WorkflowActivationPreparationIssue(
                        code="missing_required_dependency", slot_name=slot.name
                    )
                )
            if any(len(group) > 1 for group in groups):
                issues.append(
                    WorkflowActivationPreparationIssue(
                        code="ambiguous_dependency_binding", slot_name=slot.name
                    )
                )
            if all(len(group) == 1 for group in groups):
                selected.extend(group[0].selection for group in groups)
        session.expire_all()
        if not issues:
            combined = resolve_workflow_dependencies(
                session,
                contract,
                [choice.binding() for choice in selected],
                runtime_materializer=runtime_materializer,
            )
            if not combined.complete:
                raise WorkflowActivationError(
                    "workflow_activation_incomplete",
                    "Workflow dependencies changed during preparation",
                )
        return WorkflowDependencyChoices(
            state="needs_attention" if issues else "prepared",
            selections=None if issues else selected,
            slots=slots,
            issues=issues,
        )
