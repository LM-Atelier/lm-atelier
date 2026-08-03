from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from .domain import JobStatus, RunStatus
from .models import (
    ChatWorkflowSelection,
    ComfyRegistryInstall,
    CustomNodeInstall,
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
    Project,
    ProjectWorkflowSelection,
    Run,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
    WorkStep,
)


@dataclass(frozen=True)
class WorkflowDependencyImpact:
    resource_kind: str
    resource_id: str
    resource_name: str
    binding_count: int
    revision_count: int
    current_revision: bool
    shared: bool
    other_workflow_count: int
    other_family_ids: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowFamilyRemovalImpact:
    family_id: str
    revision_count: int
    current_revision_count: int
    chat_selection_count: int
    project_selection_count: int
    project_revision_pin_count: int
    active_run_count: int
    queued_step_count: int
    historical_run_count: int
    active_activation_count: int
    default_for: tuple[str, ...]
    dependencies: tuple[WorkflowDependencyImpact, ...]

    @property
    def archive_blocked(self) -> bool:
        return bool(self.chat_selection_count or self.project_selection_count or self.default_for)


@dataclass(frozen=True)
class WorkflowResourceConsumer:
    workflow_id: str
    workflow_name: str
    workflow_family_id: str | None
    workflow_family_name: str | None
    revision_ids: tuple[str, ...]
    binding_count: int
    current_revision: bool


@dataclass
class _DependencyGroup:
    resource_kind: str
    resource_id: str
    binding_ids: set[str]
    revision_ids: set[str]
    current_revision: bool
    other_workflow_ids: set[str]
    other_family_ids: set[str]


@dataclass(frozen=True)
class _BindingConsumerRow:
    binding: WorkflowDependencyBinding
    resource_kind: str
    revision_id: str
    workflow_id: str
    workflow_name: str
    current_revision_id: str | None
    family_id: str | None
    family_name: str | None


def workflow_family_removal_impact(
    session: Session,
    family: WorkflowFamily,
) -> WorkflowFamilyRemovalImpact:
    revision_rows = session.execute(
        select(
            WorkflowRevision.id,
            WorkflowDefinition.id.label("workflow_id"),
            WorkflowDefinition.current_revision_id,
        )
        .join(
            WorkflowDefinition,
            WorkflowDefinition.id == WorkflowRevision.workflow_id,
        )
        .where(WorkflowDefinition.family_id == family.id)
    ).all()
    revision_ids = {row.id for row in revision_rows}
    definition_ids = {row.workflow_id for row in revision_rows}
    current_revision_ids = {
        row.current_revision_id
        for row in revision_rows
        if row.current_revision_id is not None and row.current_revision_id == row.id
    }

    chat_selection_count = _count(
        session,
        select(func.count(ChatWorkflowSelection.id)).where(
            ChatWorkflowSelection.workflow_family_id == family.id
        ),
    )
    project_selection_count = _count(
        session,
        select(func.count(ProjectWorkflowSelection.id)).where(
            ProjectWorkflowSelection.workflow_family_id == family.id
        ),
    )
    active_run_count = (
        _count(
            session,
            select(func.count(Run.id)).where(
                Run.workflow_revision_id.in_(revision_ids),
                Run.status.in_(
                    (
                        RunStatus.PENDING.value,
                        RunStatus.ROUTING.value,
                        RunStatus.QUEUED.value,
                        RunStatus.RUNNING.value,
                    )
                ),
            ),
        )
        if revision_ids
        else 0
    )
    queued_step_count = (
        _count(
            session,
            select(func.count(WorkStep.id)).where(
                WorkStep.workflow_revision_id.in_(revision_ids),
                WorkStep.status.in_(
                    (
                        JobStatus.QUEUED.value,
                        JobStatus.RUNNING.value,
                        JobStatus.PAUSED.value,
                    )
                ),
            ),
        )
        if revision_ids
        else 0
    )
    historical_run_count = (
        _count(
            session,
            select(func.count(Run.id)).where(Run.workflow_revision_id.in_(revision_ids)),
        )
        if revision_ids
        else 0
    )
    active_activation_count = (
        _count(
            session,
            select(func.count(WorkflowActivation.id)).where(
                WorkflowActivation.workflow_revision_id.in_(revision_ids),
                WorkflowActivation.is_active.is_(True),
            ),
        )
        if revision_ids
        else 0
    )
    default_for = tuple(
        session.scalars(
            select(WorkflowPreference.selector_capability)
            .where(
                WorkflowPreference.workflow_family_id == family.id,
                WorkflowPreference.is_default.is_(True),
            )
            .order_by(WorkflowPreference.selector_capability)
        ).all()
    )
    return WorkflowFamilyRemovalImpact(
        family_id=family.id,
        revision_count=len(revision_ids),
        current_revision_count=len(current_revision_ids),
        chat_selection_count=chat_selection_count,
        project_selection_count=project_selection_count,
        project_revision_pin_count=_project_revision_pin_count(session, revision_ids),
        active_run_count=active_run_count,
        queued_step_count=queued_step_count,
        historical_run_count=historical_run_count,
        active_activation_count=active_activation_count,
        default_for=default_for,
        dependencies=_dependency_impacts(
            session,
            revision_ids=revision_ids,
            definition_ids=definition_ids,
            current_revision_ids=current_revision_ids,
        ),
    )


def workflow_family_selector_reference_count(
    session: Session,
    family_id: str,
    selector_capability: str,
) -> int:
    chat_count = _count(
        session,
        select(func.count(ChatWorkflowSelection.id)).where(
            ChatWorkflowSelection.workflow_family_id == family_id,
            ChatWorkflowSelection.selector_capability == selector_capability,
        ),
    )
    project_count = _count(
        session,
        select(func.count(ProjectWorkflowSelection.id)).where(
            ProjectWorkflowSelection.workflow_family_id == family_id,
            ProjectWorkflowSelection.selector_capability == selector_capability,
        ),
    )
    return chat_count + project_count


def workflow_resource_consumers(
    session: Session,
    resource_kind: str,
    resource_id: str,
) -> tuple[WorkflowResourceConsumer, ...]:
    grouped: dict[
        tuple[str, str, str | None, str | None],
        tuple[set[str], set[str], bool],
    ] = {}
    for row in _binding_consumer_rows(session):
        if row.resource_kind != resource_kind:
            continue
        if _binding_resource_id(row.binding) != resource_id:
            continue
        key = (
            row.workflow_id,
            row.workflow_name,
            row.family_id,
            row.family_name,
        )
        revision_ids, binding_ids, current_revision = grouped.get(
            key,
            (set(), set(), False),
        )
        revision_ids.add(row.revision_id)
        binding_ids.add(row.binding.id)
        grouped[key] = (
            revision_ids,
            binding_ids,
            current_revision or row.current_revision_id == row.revision_id,
        )
    return tuple(
        WorkflowResourceConsumer(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            workflow_family_id=family_id,
            workflow_family_name=family_name,
            revision_ids=tuple(sorted(revision_ids)),
            binding_count=len(binding_ids),
            current_revision=current_revision,
        )
        for (
            workflow_id,
            workflow_name,
            family_id,
            family_name,
        ), (
            revision_ids,
            binding_ids,
            current_revision,
        ) in sorted(grouped.items(), key=lambda item: item[0][0])
    )


def workflow_resource_name(session: Session, resource_kind: str, resource_id: str) -> str:
    return _resource_name(session, resource_kind, resource_id)


def _project_revision_pin_count(session: Session, revision_ids: set[str]) -> int:
    if not revision_ids:
        return 0
    project_ids = set(
        session.scalars(
            select(ProjectWorkflowSelection.project_id).where(
                ProjectWorkflowSelection.workflow_revision_id.in_(revision_ids)
            )
        ).all()
    )
    project_ids.update(
        session.scalars(
            select(Project.id).where(
                or_(
                    Project.image_workflow_revision_id.in_(revision_ids),
                    Project.video_workflow_revision_id.in_(revision_ids),
                )
            )
        ).all()
    )
    return len(project_ids)


def _count(session: Session, statement: Select[tuple[int]]) -> int:
    return int(session.scalar(statement) or 0)


def _dependency_impacts(
    session: Session,
    *,
    revision_ids: set[str],
    definition_ids: set[str],
    current_revision_ids: set[str],
) -> tuple[WorkflowDependencyImpact, ...]:
    if not revision_ids:
        return ()
    rows = _binding_consumer_rows(session)
    consumers: dict[tuple[str, str], list[tuple[str, str | None]]] = defaultdict(list)
    own_bindings: list[tuple[WorkflowDependencyBinding, str, str]] = []
    for row in rows:
        resource_id = _binding_resource_id(row.binding)
        if resource_id is None:
            continue
        key = (row.resource_kind, resource_id)
        consumers[key].append((row.workflow_id, row.family_id))
        if row.revision_id in revision_ids:
            own_bindings.append((row.binding, row.resource_kind, row.revision_id))

    groups: dict[tuple[str, str], _DependencyGroup] = {}
    for binding, resource_kind, revision_id in own_bindings:
        resource_id = _binding_resource_id(binding)
        if resource_id is None:  # pragma: no cover - filtered above
            continue
        key = (resource_kind, resource_id)
        group = groups.get(key)
        if group is None:
            group = _DependencyGroup(
                resource_kind=resource_kind,
                resource_id=resource_id,
                binding_ids=set(),
                revision_ids=set(),
                current_revision=False,
                other_workflow_ids=set(),
                other_family_ids=set(),
            )
            groups[key] = group
        group.binding_ids.add(binding.id)
        group.revision_ids.add(revision_id)
        group.current_revision = group.current_revision or revision_id in current_revision_ids
        for workflow_id, family_id in consumers[key]:
            if workflow_id in definition_ids:
                continue
            group.other_workflow_ids.add(workflow_id)
            if family_id is not None:
                group.other_family_ids.add(family_id)

    return tuple(
        WorkflowDependencyImpact(
            resource_kind=group.resource_kind,
            resource_id=group.resource_id,
            resource_name=_resource_name(session, group.resource_kind, group.resource_id),
            binding_count=len(group.binding_ids),
            revision_count=len(group.revision_ids),
            current_revision=group.current_revision,
            shared=bool(group.other_workflow_ids),
            other_workflow_count=len(group.other_workflow_ids),
            other_family_ids=tuple(sorted(group.other_family_ids)),
        )
        for group in sorted(
            groups.values(),
            key=lambda item: (item.resource_kind, item.resource_id),
        )
    )


def _binding_consumer_rows(session: Session) -> tuple[_BindingConsumerRow, ...]:
    rows = session.execute(
        select(
            WorkflowDependencyBinding,
            WorkflowDependencySlot.resource_kind,
            WorkflowRevision.id.label("revision_id"),
            WorkflowDefinition.id.label("workflow_id"),
            WorkflowDefinition.name.label("workflow_name"),
            WorkflowDefinition.current_revision_id,
            WorkflowDefinition.family_id,
            WorkflowFamily.name.label("family_name"),
        )
        .join(
            WorkflowDependencySlot,
            (WorkflowDependencySlot.id == WorkflowDependencyBinding.workflow_dependency_slot_id)
            & (
                WorkflowDependencySlot.workflow_revision_id
                == WorkflowDependencyBinding.workflow_revision_id
            ),
        )
        .join(
            WorkflowRevision,
            WorkflowRevision.id == WorkflowDependencyBinding.workflow_revision_id,
        )
        .join(
            WorkflowDefinition,
            WorkflowDefinition.id == WorkflowRevision.workflow_id,
        )
        .outerjoin(
            WorkflowFamily,
            WorkflowFamily.id == WorkflowDefinition.family_id,
        )
    ).all()
    return tuple(
        _BindingConsumerRow(
            binding=binding,
            resource_kind=resource_kind,
            revision_id=revision_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            current_revision_id=current_revision_id,
            family_id=family_id,
            family_name=family_name,
        )
        for (
            binding,
            resource_kind,
            revision_id,
            workflow_id,
            workflow_name,
            current_revision_id,
            family_id,
            family_name,
        ) in rows
    )


def _binding_resource_id(binding: WorkflowDependencyBinding) -> str | None:
    return next(
        (
            value
            for value in (
                binding.model_profile_id,
                binding.model_install_id,
                binding.model_asset_install_id,
                binding.custom_node_install_id,
                binding.comfy_registry_install_id,
                binding.runtime_key,
            )
            if value is not None
        ),
        None,
    )


def _resource_name(session: Session, resource_kind: str, resource_id: str) -> str:
    model_and_field = {
        "model_profile": (ModelProfile, "name"),
        "model_install": (ModelInstall, "name"),
        "model_asset": (ModelAssetInstall, "name"),
        "custom_node": (CustomNodeInstall, "name"),
        "registry_package": (ComfyRegistryInstall, "package_id"),
    }.get(resource_kind)
    if model_and_field is None:
        return resource_id
    model, field = model_and_field
    resource = session.get(model, resource_id)
    value = getattr(resource, field, None) if resource is not None else None
    return value if isinstance(value, str) and value else resource_id
