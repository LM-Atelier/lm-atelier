"""Current-revision dependency summaries for browsing workflow families."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from sqlalchemy import case, func, or_, select, tuple_
from sqlalchemy.orm import Session

from .model_planner import declared_model_components
from .models import (
    ComfyRegistryInstall,
    CustomNodeInstall,
    ModelAssetInstall,
    ModelComponentManifest,
    ModelInstall,
    ModelProfile,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowProfileCompatibility,
    WorkflowRevision,
)


@dataclass(frozen=True)
class WorkflowFamilyDependencySummary:
    dependency_count: int
    names: tuple[str, ...]


def _stored_slot_summaries(
    session: Session,
    family_ids: Sequence[str],
) -> tuple[dict[str, WorkflowFamilyDependencySummary], set[str]]:
    """Read slot names and active bound resource names in one scalar query.

    Summaries describe stored dependencies, not a fresh validation of installed
    content. Historical revisions and retired activations do not contribute.
    """
    if not family_ids:
        return {}, set()
    definition = WorkflowDefinition
    revision = WorkflowRevision
    slot = WorkflowDependencySlot
    activation = WorkflowActivation
    binding = WorkflowDependencyBinding
    query = (
        select(
            definition.family_id,
            slot.id,
            revision.id,
            slot.name,
            ModelProfile.name,
            ModelInstall.name,
            ModelAssetInstall.name,
            CustomNodeInstall.name,
            ComfyRegistryInstall.package_id,
            case((slot.resource_kind == "runtime", binding.runtime_key)),
        )
        .select_from(definition)
        .join(
            revision,
            (revision.id == definition.current_revision_id)
            & (revision.workflow_id == definition.id),
        )
        .join(slot, slot.workflow_revision_id == revision.id)
        .outerjoin(
            activation,
            (activation.workflow_revision_id == revision.id)
            & activation.is_active.is_(True)
            & (activation.state == "ready")
            & activation.invalidated_at.is_(None),
        )
        .outerjoin(
            binding,
            (binding.workflow_activation_id == activation.id)
            & (binding.workflow_revision_id == revision.id)
            & (binding.workflow_dependency_slot_id == slot.id),
        )
        .outerjoin(
            ModelProfile,
            (ModelProfile.id == binding.model_profile_id) & (slot.resource_kind == "model_profile"),
        )
        .outerjoin(
            ModelInstall,
            (ModelInstall.id == binding.model_install_id) & (slot.resource_kind == "model_install"),
        )
        .outerjoin(
            ModelAssetInstall,
            (ModelAssetInstall.id == binding.model_asset_install_id)
            & (slot.resource_kind == "model_asset"),
        )
        .outerjoin(
            CustomNodeInstall,
            (CustomNodeInstall.id == binding.custom_node_install_id)
            & (slot.resource_kind == "custom_node"),
        )
        .outerjoin(
            ComfyRegistryInstall,
            (ComfyRegistryInstall.id == binding.comfy_registry_install_id)
            & (slot.resource_kind == "registry_package"),
        )
        .where(definition.family_id.in_(family_ids))
        .execution_options(autoflush=False)
    )
    slots: dict[str, set[str]] = {family_id: set() for family_id in family_ids}
    names: dict[str, set[str]] = {family_id: set() for family_id in family_ids}
    slotted_revisions: set[str] = set()
    for row in session.execute(query):
        family_id, slot_id, revision_id, *labels = row
        slotted_revisions.add(revision_id)
        slots[family_id].add(slot_id)
        names[family_id].update(label for label in labels if isinstance(label, str) and label)
    summaries = {
        family_id: WorkflowFamilyDependencySummary(
            dependency_count=len(slots[family_id]),
            names=tuple(sorted(names[family_id], key=lambda name: (name.casefold(), name))),
        )
        for family_id in family_ids
    }
    return summaries, slotted_revisions


def _reference_label(value: str) -> str:
    """Use a filename for locator-shaped declarations, without URL credentials."""
    if "://" in value:
        try:
            value = urlsplit(value).path
        except ValueError:
            return ""
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].strip()


def _declared_references(value: object) -> dict[tuple[str, str], str | None]:
    result: dict[tuple[str, str], str | None] = {}
    if not isinstance(value, dict):
        return result
    identifiers = value.get("model_install_ids")
    for identifier in identifiers if isinstance(identifiers, list) else []:
        if isinstance(identifier, str) and identifier:
            result[("model_install", identifier)] = None
    for field, kind, keys in [
        ("models", "model_install", ("id", "name", "path")),
        ("custom_nodes", "custom_node", ("id", "name", "source_url")),
        ("registry_packages", "registry_package", ("package_id",)),
    ]:
        declarations = value.get(field)
        for declaration in declarations if isinstance(declarations, list) else []:
            identifier = declaration
            identity_field = ""
            if isinstance(declaration, dict):
                for key in keys:
                    if declaration.get(key):
                        identifier = declaration[key]
                        identity_field = key
                        break
            if not isinstance(identifier, str) or not identifier:
                continue
            label = (
                None
                if identity_field == "id"
                else (
                    identifier
                    if identity_field in {"name", "package_id"}
                    else _reference_label(identifier)
                )
            )
            result[(kind, identifier)] = label
    return result


def _installed_reference_names(
    session: Session,
    references: set[tuple[str, str]],
) -> dict[tuple[str, str], set[tuple[str, str]]]:
    result: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for kind, id_column, name_column, locator in [
        ("model_install", ModelInstall.id, ModelInstall.name, ModelInstall.local_path),
        ("custom_node", CustomNodeInstall.id, CustomNodeInstall.name, CustomNodeInstall.source_url),
    ]:
        identifiers = sorted(identifier for ref_kind, identifier in references if ref_kind == kind)
        for offset in range(0, len(identifiers), 200):
            batch = identifiers[offset : offset + 200]
            rows = session.execute(
                select(id_column, name_column, locator)
                .where(or_(id_column.in_(batch), name_column.in_(batch), locator.in_(batch)))
                .execution_options(autoflush=False)
            )
            for row_id, name, location in rows:
                for identifier in (row_id, name, location):
                    if (kind, identifier) in references:
                        result.setdefault((kind, identifier), set()).add((row_id, name))
    return result


def _installed_component_names(
    session: Session,
    components: set[tuple[str, str]],
) -> dict[tuple[str, str], set[tuple[str, str]]]:
    result: dict[tuple[str, str], set[tuple[str, str]]] = {}
    ordered = sorted(components)
    manifest = ModelComponentManifest
    for offset in range(0, len(ordered), 200):
        rows = session.execute(
            select(
                manifest.target_folder,
                func.lower(manifest.sha256),
                ModelInstall.id,
                ModelInstall.name,
            )
            .join(ModelInstall, ModelInstall.id == manifest.model_install_id)
            .where(
                manifest.required.is_(True),
                tuple_(manifest.target_folder, func.lower(manifest.sha256)).in_(
                    ordered[offset : offset + 200]
                ),
            )
            .execution_options(autoflush=False)
        )
        for folder, digest, model_id, name in rows:
            result.setdefault((folder, digest), set()).add((model_id, name))
    return result


def workflow_family_dependency_summaries(
    session: Session,
    family_ids: Sequence[str],
) -> dict[str, WorkflowFamilyDependencySummary]:
    """Summarize stored slots, current declarations, and profile-backed models.

    Counts describe recorded requirements, not unique downloaded files. Only
    known dependency fields are read; graphs, settings and content stay outside
    the projection. Lookup batches are independent of the number of families.
    """
    if not family_ids:
        return {}
    summaries, slotted_revisions = _stored_slot_summaries(session, family_ids)
    names = {family_id: set(summary.names) for family_id, summary in summaries.items()}
    references: dict[str, dict[tuple[str, str], str | None]] = {
        family_id: {} for family_id in family_ids
    }
    components: dict[str, set[tuple[str, str]]] = {family_id: set() for family_id in family_ids}
    revisions = session.execute(
        select(
            WorkflowDefinition.family_id, WorkflowRevision.id, WorkflowRevision.dependencies_json
        )
        .join(
            WorkflowRevision,
            (WorkflowRevision.id == WorkflowDefinition.current_revision_id)
            & (WorkflowRevision.workflow_id == WorkflowDefinition.id),
        )
        .where(WorkflowDefinition.family_id.in_(family_ids))
        .execution_options(autoflush=False)
    )
    for family_id, revision_id, declarations in revisions:
        if revision_id not in slotted_revisions:
            references[family_id].update(_declared_references(declarations))
            if isinstance(declarations, dict):
                components[family_id].update(
                    (item["target_folder"], item["sha256"])
                    for item in declared_model_components(declarations)
                )
    implicit_models = session.execute(
        select(WorkflowProfileCompatibility.workflow_family_id, ModelInstall.id, ModelInstall.name)
        .join(ModelProfile, ModelProfile.id == WorkflowProfileCompatibility.model_profile_id)
        .join(ModelInstall, ModelInstall.id == ModelProfile.model_install_id)
        .where(
            WorkflowProfileCompatibility.workflow_family_id.in_(family_ids),
            select(WorkflowDefinition.id)
            .where(
                WorkflowDefinition.family_id == WorkflowProfileCompatibility.workflow_family_id,
                WorkflowDefinition.current_revision_id.is_(None),
            )
            .exists(),
        )
        .execution_options(autoflush=False)
    )
    for family_id, model_id, name in implicit_models:
        references[family_id][("model_install", model_id)] = name
    all_references = {key for refs in references.values() for key in refs}
    installed = _installed_reference_names(session, all_references)
    installed_components = _installed_component_names(
        session, {key for items in components.values() for key in items}
    )
    result = {}
    for family_id, summary in summaries.items():
        identities: set[tuple[str, ...]] = set()
        for key, fallback in references[family_id].items():
            matches = installed.get(key, set())
            identity = (key[0], next(iter(matches))[0]) if len(matches) == 1 else key
            identities.add(identity)
            names[family_id].update(name for _, name in matches if name)
            if fallback and not matches:
                names[family_id].add(fallback)
        for component in components[family_id]:
            matches = installed_components.get(component, set())
            identities.add(
                ("model_install", next(iter(matches))[0])
                if len(matches) == 1
                else ("model_component", *component)
            )
            names[family_id].update(name for _, name in matches if name)
        result[family_id] = WorkflowFamilyDependencySummary(
            dependency_count=summary.dependency_count + len(identities),
            names=tuple(sorted(names[family_id], key=lambda name: (name.casefold(), name))),
        )
    return result
