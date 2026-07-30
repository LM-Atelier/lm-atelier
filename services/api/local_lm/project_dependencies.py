from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .models import (
    Chat,
    GenerationPreset,
    ModelProfile,
    Project,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
)
from .profile_service import AUTO_PROFILE_ID
from .project_portability import redact_local_paths
from .settings_registry import ROLE_SETTINGS
from .workflow_edit_calibration import validate_workflow_edit_calibration

ModelRoleName = Literal["chat", "image", "video"]
OperationName = Literal[
    "text",
    "text_to_image",
    "image_to_image",
    "text_to_video",
    "image_to_video",
]


class _PortableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PortableProfile(_PortableModel):
    source_id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=200)
    use_case: str = Field(default="", max_length=1_000)
    role: ModelRoleName
    engine: str = Field(min_length=1, max_length=32)
    load_settings: dict[str, Any] = Field(default_factory=dict, max_length=256)
    request_settings: dict[str, Any] = Field(default_factory=dict, max_length=256)


class PortablePreset(_PortableModel):
    source_id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=200)
    role: ModelRoleName
    settings: dict[str, Any] = Field(default_factory=dict, max_length=256)


class PortableWorkflowRevision(_PortableModel):
    source_id: str = Field(min_length=1, max_length=80)
    source_version: int = Field(gt=0, le=2_147_483_647)
    engine: str = Field(min_length=1, max_length=32)
    engine_version: str | None = Field(default=None, max_length=100)
    ui_graph: dict[str, Any] = Field(default_factory=dict)
    api_graph: dict[str, Any] = Field(default_factory=dict)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict)
    trusted: bool = False


class PortableWorkflow(_PortableModel):
    source_id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=240)
    operation: OperationName
    description: str = Field(default="", max_length=10_000)
    current_revision_source_id: str = Field(min_length=1, max_length=80)
    revisions: list[PortableWorkflowRevision] = Field(min_length=1, max_length=10_000)


class PortableDependencies(_PortableModel):
    profiles: list[PortableProfile] = Field(default_factory=list, max_length=10_000)
    presets: list[PortablePreset] = Field(default_factory=list, max_length=10_000)
    workflows: list[PortableWorkflow] = Field(default_factory=list, max_length=10_000)


@dataclass(frozen=True)
class DependencySourceIndex:
    profile_roles: dict[str, str]
    preset_roles: dict[str, str]
    revision_operations: dict[str, str]
    revision_workflow_ids: dict[str, str]


@dataclass(frozen=True)
class ImportedDependencies:
    profile_ids: dict[str, str]
    profile_roles: dict[str, str]
    preset_ids: dict[str, str]
    preset_roles: dict[str, str]
    workflow_ids: dict[str, str]
    revision_ids: dict[str, str]
    revision_operations: dict[str, str]
    revision_workflow_source_ids: dict[str, str]

    def profile(
        self,
        source_id: object,
        expected_role: str,
        *,
        allow_auto: bool = False,
    ) -> str | None:
        if source_id is None:
            return None
        if allow_auto and source_id == AUTO_PROFILE_ID:
            return AUTO_PROFILE_ID
        if not isinstance(source_id, str) or source_id not in self.profile_ids:
            raise ValueError("project manifest references an unavailable profile")
        if self.profile_roles[source_id] != expected_role:
            raise ValueError("project manifest references a profile with an incompatible role")
        return self.profile_ids[source_id]

    def preset(self, source_id: object, expected_role: str) -> str | None:
        if source_id is None:
            return None
        if not isinstance(source_id, str) or source_id not in self.preset_ids:
            raise ValueError("project manifest references an unavailable generation preset")
        if self.preset_roles[source_id] != expected_role:
            raise ValueError(
                "project manifest references a generation preset with an incompatible role"
            )
        return self.preset_ids[source_id]

    def revision(self, source_id: object, operations: set[str]) -> str | None:
        if source_id is None:
            return None
        if not isinstance(source_id, str) or source_id not in self.revision_ids:
            raise ValueError("project manifest references an unavailable workflow revision")
        if self.revision_operations[source_id] not in operations:
            raise ValueError(
                "project manifest references a workflow revision with an incompatible operation"
            )
        return self.revision_ids[source_id]

    def workflow(self, source_id: object) -> str | None:
        if source_id is None:
            return None
        if not isinstance(source_id, str) or source_id not in self.workflow_ids:
            raise ValueError("project manifest references an unavailable workflow")
        return self.workflow_ids[source_id]


def build_dependency_manifest(
    session: Session,
    project: Project,
    chats: list[Chat],
    runs: list[Run],
) -> tuple[dict[str, Any], DependencySourceIndex]:
    profile_ids = {
        value
        for chat in chats
        for value in (
            chat.active_chat_profile_id,
            chat.active_vision_profile_id,
            chat.active_image_profile_id,
            chat.active_video_profile_id,
        )
        if isinstance(value, str) and value != AUTO_PROFILE_ID
    }
    profile_ids.update(run.profile_id for run in runs if isinstance(run.profile_id, str))
    profile_ids.update(
        run.vision_profile_id for run in runs if isinstance(run.vision_profile_id, str)
    )
    profiles = [
        profile
        for profile_id in sorted(profile_ids)
        if (profile := session.get(ModelProfile, profile_id)) is not None
    ]

    preset_ids: set[str] = set()
    owners: list[Project | Chat] = [project, *chats]
    for owner in owners:
        bindings = owner.generation_preset_ids_json
        preset_ids.update(
            preset_id
            for preset_id in (bindings.values() if isinstance(bindings, dict) else [])
            if isinstance(preset_id, str)
        )
    for run in runs:
        provenance = run.provenance_json if isinstance(run.provenance_json, dict) else {}
        preset = provenance.get("preset")
        if isinstance(preset, dict) and isinstance(preset.get("id"), str):
            preset_ids.add(preset["id"])
        layers = provenance.get("preset_layers")
        if isinstance(layers, list):
            preset_ids.update(
                layer["id"]
                for layer in layers
                if isinstance(layer, dict) and isinstance(layer.get("id"), str)
            )
    presets = [
        preset
        for preset_id in sorted(preset_ids)
        if (preset := session.get(GenerationPreset, preset_id)) is not None
    ]

    revision_ids = {
        revision_id
        for revision_id in (
            project.image_workflow_revision_id,
            project.video_workflow_revision_id,
            *(run.workflow_revision_id for run in runs),
        )
        if isinstance(revision_id, str)
    }
    workflow_ids: set[str] = set()
    for revision_id in revision_ids:
        revision = session.get(WorkflowRevision, revision_id)
        if revision and session.get(WorkflowDefinition, revision.workflow_id):
            workflow_ids.add(revision.workflow_id)

    workflows: list[PortableWorkflow] = []
    for workflow_id in sorted(workflow_ids):
        definition = session.get(WorkflowDefinition, workflow_id)
        if not definition:
            continue
        revisions = list(
            session.scalars(
                select(WorkflowRevision)
                .where(WorkflowRevision.workflow_id == workflow_id)
                .order_by(WorkflowRevision.version, WorkflowRevision.id)
            ).all()
        )
        if not revisions:
            continue
        revision_source_ids = {revision.id for revision in revisions}
        current_revision_source_id = (
            definition.current_revision_id
            if definition.current_revision_id in revision_source_ids
            else revisions[-1].id
        )
        workflows.append(
            PortableWorkflow(
                source_id=definition.id,
                name=definition.name,
                operation=definition.operation,  # type: ignore[arg-type]
                description=definition.description,
                current_revision_source_id=current_revision_source_id,
                revisions=[
                    PortableWorkflowRevision(
                        source_id=revision.id,
                        source_version=revision.version,
                        engine=revision.engine,
                        engine_version=revision.engine_version,
                        ui_graph=_portable_mapping(revision.ui_graph_json),
                        api_graph=_portable_mapping(revision.api_graph_json),
                        input_schema=_portable_mapping(revision.input_schema_json),
                        dependencies=_portable_mapping(revision.dependencies_json),
                        trusted=revision.trusted,
                    )
                    for revision in revisions
                ],
            )
        )

    dependency_model = PortableDependencies(
        profiles=[
            PortableProfile(
                source_id=profile.id,
                name=profile.name,
                use_case=profile.use_case,
                role=profile.role,  # type: ignore[arg-type]
                engine=profile.engine,
                load_settings=_portable_mapping(profile.load_settings_json),
                request_settings=_portable_mapping(profile.request_settings_json),
            )
            for profile in profiles
        ],
        presets=[
            PortablePreset(
                source_id=preset.id,
                name=preset.name,
                role=preset.role,  # type: ignore[arg-type]
                settings=_portable_mapping(preset.settings_json),
            )
            for preset in presets
        ],
        workflows=workflows,
    )
    index = dependency_source_index(dependency_model)
    return dependency_model.model_dump(mode="json"), index


def parse_dependency_manifest(value: object) -> PortableDependencies:
    try:
        dependencies = PortableDependencies.model_validate(
            redact_local_paths(value),
            strict=True,
        )
    except ValidationError as exc:
        raise ValueError("project manifest has invalid portable dependencies") from exc
    dependency_source_index(dependencies)
    return dependencies


def dependency_source_index(dependencies: PortableDependencies) -> DependencySourceIndex:
    profile_roles: dict[str, str] = {}
    for profile in dependencies.profiles:
        if profile.source_id == AUTO_PROFILE_ID or profile.source_id in profile_roles:
            raise ValueError("project manifest contains duplicate profile dependency ids")
        profile_roles[profile.source_id] = profile.role

    preset_roles: dict[str, str] = {}
    for preset in dependencies.presets:
        if preset.source_id in preset_roles:
            raise ValueError("project manifest contains duplicate preset dependency ids")
        preset_roles[preset.source_id] = preset.role

    workflow_ids: set[str] = set()
    revision_operations: dict[str, str] = {}
    revision_workflow_ids: dict[str, str] = {}
    for workflow in dependencies.workflows:
        if workflow.source_id in workflow_ids:
            raise ValueError("project manifest contains duplicate workflow dependency ids")
        workflow_ids.add(workflow.source_id)
        versions: set[int] = set()
        workflow_revision_ids: set[str] = set()
        for revision in workflow.revisions:
            validate_workflow_edit_calibration(revision.input_schema)
            if revision.source_id in revision_operations:
                raise ValueError(
                    "project manifest contains duplicate workflow revision dependency ids"
                )
            if revision.source_version in versions:
                raise ValueError("project manifest contains duplicate workflow revision versions")
            versions.add(revision.source_version)
            workflow_revision_ids.add(revision.source_id)
            revision_operations[revision.source_id] = workflow.operation
            revision_workflow_ids[revision.source_id] = workflow.source_id
        if workflow.current_revision_source_id not in workflow_revision_ids:
            raise ValueError("project manifest has an invalid current workflow revision")
    return DependencySourceIndex(
        profile_roles=profile_roles,
        preset_roles=preset_roles,
        revision_operations=revision_operations,
        revision_workflow_ids=revision_workflow_ids,
    )


def _without_load_scope_settings(values: dict[str, Any], role: str) -> dict[str, Any]:
    """Strip settings only a load-scope layer may carry.

    The REST API refuses load-scope keys on request-scope layers, because a value
    stored there is validated against every field and would then outrank layers
    that cannot express it. Archive import wrote its settings verbatim, so an old
    or crafted archive could place them where the API will not.

    Only keys the registry positively identifies as load-scope are removed.
    Anything else is left alone: a workflow can extend the field set beyond what
    the registry knows, and dropping unrecognised keys here would silently
    discard legitimate settings.
    """

    load_scope = {field.key for field in ROLE_SETTINGS.get(role, []) if field.scope == "load"}
    return {key: value for key, value in values.items() if key not in load_scope}


def install_dependency_manifest(
    session: Session, dependencies: PortableDependencies
) -> ImportedDependencies:
    profile_ids: dict[str, str] = {}
    profile_roles: dict[str, str] = {}
    for profile_source in dependencies.profiles:
        profile = _matching_profile(session, profile_source)
        if not profile:
            profile = ModelProfile(
                name=profile_source.name,
                use_case=profile_source.use_case,
                role=profile_source.role,
                engine=profile_source.engine,
                model_install_id=None,
                load_settings_json=profile_source.load_settings,
                request_settings_json=_without_load_scope_settings(
                    profile_source.request_settings, profile_source.role
                ),
                is_default=False,
            )
            session.add(profile)
            session.flush()
        profile_ids[profile_source.source_id] = profile.id
        profile_roles[profile_source.source_id] = profile_source.role

    preset_ids: dict[str, str] = {}
    preset_roles: dict[str, str] = {}
    for preset_source in dependencies.presets:
        preset = _matching_preset(session, preset_source)
        if preset:
            imported = preset
        else:
            imported = GenerationPreset(
                name=_available_preset_name(session, preset_source.role, preset_source.name),
                role=preset_source.role,
                settings_json=_without_load_scope_settings(
                    preset_source.settings, preset_source.role
                ),
                is_default=False,
            )
            session.add(imported)
            session.flush()
        preset_ids[preset_source.source_id] = imported.id
        preset_roles[preset_source.source_id] = preset_source.role

    revision_ids: dict[str, str] = {}
    revision_operations: dict[str, str] = {}
    imported_workflow_ids: dict[str, str] = {}
    for workflow_source in dependencies.workflows:
        match = _matching_workflow(session, workflow_source)
        if match:
            definition, matched_revisions = match
        else:
            definition = WorkflowDefinition(
                name=workflow_source.name,
                operation=workflow_source.operation,
                description=workflow_source.description,
            )
            session.add(definition)
            session.flush()
            matched_revisions = {}
            for source_revision in workflow_source.revisions:
                revision = WorkflowRevision(
                    workflow_id=definition.id,
                    version=source_revision.source_version,
                    engine=source_revision.engine,
                    engine_version=source_revision.engine_version,
                    ui_graph_json=source_revision.ui_graph,
                    api_graph_json=source_revision.api_graph,
                    input_schema_json=source_revision.input_schema,
                    dependencies_json=source_revision.dependencies,
                    # Trust is local security state and never crosses an archive boundary.
                    trusted=False,
                )
                session.add(revision)
                session.flush()
                matched_revisions[source_revision.source_id] = revision
            definition.current_revision_id = matched_revisions[
                workflow_source.current_revision_source_id
            ].id
        imported_workflow_ids[workflow_source.source_id] = definition.id
        for source_revision in workflow_source.revisions:
            revision = matched_revisions[source_revision.source_id]
            revision_ids[source_revision.source_id] = revision.id
            revision_operations[source_revision.source_id] = workflow_source.operation

    return ImportedDependencies(
        profile_ids=profile_ids,
        profile_roles=profile_roles,
        preset_ids=preset_ids,
        preset_roles=preset_roles,
        workflow_ids=imported_workflow_ids,
        revision_ids=revision_ids,
        revision_operations=revision_operations,
        revision_workflow_source_ids={
            revision.source_id: workflow.source_id
            for workflow in dependencies.workflows
            for revision in workflow.revisions
        },
    )


def _matching_profile(session: Session, source: PortableProfile) -> ModelProfile | None:
    candidates = session.scalars(
        select(ModelProfile).where(
            ModelProfile.name == source.name,
            ModelProfile.use_case == source.use_case,
            ModelProfile.role == source.role,
            ModelProfile.engine == source.engine,
            ModelProfile.model_install_id.is_(None),
        )
    ).all()
    return next(
        (
            profile
            for profile in candidates
            if _json_equal(profile.load_settings_json, source.load_settings)
            and _json_equal(profile.request_settings_json, source.request_settings)
        ),
        None,
    )


def _matching_preset(session: Session, source: PortablePreset) -> GenerationPreset | None:
    candidates = session.scalars(
        select(GenerationPreset).where(GenerationPreset.role == source.role)
    ).all()
    return next(
        (
            preset
            for preset in candidates
            if _json_equal(preset.settings_json, source.settings)
            and (preset.name == source.name or _is_imported_name(preset.name, source.name))
        ),
        None,
    )


def _is_imported_name(candidate: str, source: str) -> bool:
    match = re.search(r" \(imported(?: ([1-9][0-9]{0,3}|10000))?\)$", candidate)
    if not match:
        return False
    if match.group(1) is not None and int(match.group(1)) < 2:
        return False
    label = match.group(0)
    return candidate == f"{source[: 200 - len(label)]}{label}"


def _available_preset_name(session: Session, role: str, requested: str) -> str:
    if not session.scalar(
        select(GenerationPreset.id).where(
            GenerationPreset.role == role,
            GenerationPreset.name == requested,
        )
    ):
        return requested
    suffix = " (imported)"
    counter = 1
    while True:
        label = suffix if counter == 1 else f" (imported {counter})"
        candidate = f"{requested[: 200 - len(label)]}{label}"
        if not session.scalar(
            select(GenerationPreset.id).where(
                GenerationPreset.role == role,
                GenerationPreset.name == candidate,
            )
        ):
            return candidate
        counter += 1
        if counter > 10_000:
            raise ValueError("could not allocate a generation preset name during import")


def _matching_workflow(
    session: Session, source: PortableWorkflow
) -> tuple[WorkflowDefinition, dict[str, WorkflowRevision]] | None:
    candidates = session.scalars(
        select(WorkflowDefinition)
        .options(selectinload(WorkflowDefinition.revisions))
        .where(
            WorkflowDefinition.name == source.name,
            WorkflowDefinition.operation == source.operation,
            WorkflowDefinition.description == source.description,
        )
    ).all()
    source_by_version = {revision.source_version: revision for revision in source.revisions}
    current_version = next(
        revision.source_version
        for revision in source.revisions
        if revision.source_id == source.current_revision_source_id
    )
    for definition in candidates:
        existing_by_version = {revision.version: revision for revision in definition.revisions}
        if set(existing_by_version) != set(source_by_version):
            continue
        if not all(
            _revision_equal(existing_by_version[version], source_revision)
            for version, source_revision in source_by_version.items()
        ):
            continue
        current = (
            session.get(WorkflowRevision, definition.current_revision_id)
            if definition.current_revision_id
            else None
        )
        if not current or current.version != current_version:
            continue
        return definition, {
            source_revision.source_id: existing_by_version[source_revision.source_version]
            for source_revision in source.revisions
        }
    return None


def _revision_equal(existing: WorkflowRevision, source: PortableWorkflowRevision) -> bool:
    return (
        existing.engine == source.engine
        and existing.engine_version == source.engine_version
        and _json_equal(existing.ui_graph_json, source.ui_graph)
        and _json_equal(existing.api_graph_json, source.api_graph)
        and _json_equal(existing.input_schema_json, source.input_schema)
        and _json_equal(existing.dependencies_json, source.dependencies)
    )


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _portable_mapping(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], redact_local_paths(_mapping(value)))


def _json_equal(left: object, right: object) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
