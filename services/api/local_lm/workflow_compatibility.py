from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    Chat,
    ChatWorkflowSelection,
    ModelProfile,
    Project,
    ProjectWorkflowSelection,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowProfileCompatibility,
)

AUTO_PROFILE_ID = "__auto__"

ChatSelectorCapability = Literal["chat", "vision", "image", "video"]
ProjectSelectorCapability = Literal["image", "video"]

_CHAT_PROFILE_FIELDS: dict[ChatSelectorCapability, str] = {
    "chat": "active_chat_profile_id",
    "vision": "active_vision_profile_id",
    "image": "active_image_profile_id",
    "video": "active_video_profile_id",
}
_PROJECT_REVISION_FIELDS: dict[ProjectSelectorCapability, str] = {
    "image": "image_workflow_revision_id",
    "video": "video_workflow_revision_id",
}
_CAPABILITY_ROLE: dict[ChatSelectorCapability, str] = {
    "chat": "chat",
    "vision": "chat",
    "image": "image",
    "video": "video",
}
_PROFILE_CAPABILITIES: dict[str, tuple[ChatSelectorCapability, ...]] = {
    "chat": ("chat", "vision"),
    "image": ("image",),
    "video": ("video",),
}
_PROFILE_VARIANTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "chat": (("text", "text", "Chat"),),
    "image": (
        ("create", "text_to_image", "Create"),
        ("edit", "image_to_image", "Edit"),
    ),
    "video": (
        ("create", "text_to_video", "Create"),
        ("animate", "image_to_video", "Animate"),
    ),
}


class WorkflowCompatibilityConflict(RuntimeError):
    """A deterministic compatibility identity collides with different data."""


class WorkflowSelectionInvalid(RuntimeError):
    def __init__(self, *, capability: str, reason: str) -> None:
        self.capability = capability
        self.reason = reason
        super().__init__(f"{capability} workflow selection is invalid: {reason}")


@dataclass(frozen=True)
class WorkflowCompatibilityReport:
    profiles: int
    families_created: int
    definitions_created: int
    preferences_created: int
    chat_selections_created: int
    project_selections_created: int


@dataclass(frozen=True)
class ResolvedChatWorkflowSelection:
    capability: ChatSelectorCapability
    mode: Literal["legacy", "automatic", "family"]
    profile_id: str | None
    workflow_family_id: str | None


@dataclass(frozen=True)
class ResolvedProjectWorkflowSelection:
    capability: ProjectSelectorCapability
    mode: Literal["legacy", "automatic", "family", "revision"]
    workflow_family_id: str | None
    workflow_revision_id: str | None


def _stable_id(prefix: str, *parts: str, maximum: int) -> str:
    digest = hashlib.sha256("\\x00".join(parts).encode("utf-8")).hexdigest()
    available = maximum - len(prefix) - 1
    if available <= 0:
        raise ValueError("stable id prefix exceeds its target width")
    return f"{prefix}_{digest[:available]}"


def compatibility_family_id(profile_id: str) -> str:
    return _stable_id("wffamily_compat", profile_id, maximum=64)


def compatibility_definition_id(profile_id: str, variant_key: str) -> str:
    return _stable_id("workflow_compat", profile_id, variant_key, maximum=64)


def _selection_id(owner_id: str, capability: str, scope: str) -> str:
    return _stable_id("wfsel", scope, owner_id, capability, maximum=40)


def _preference_id(profile_id: str, capability: str) -> str:
    return _stable_id("wfpref", profile_id, capability, maximum=40)


def _profile_fingerprint(profile: ModelProfile) -> str:
    payload = {
        "engine": profile.engine,
        "is_default": profile.is_default,
        "load_settings": profile.load_settings_json,
        "model_install_id": profile.model_install_id,
        "name": profile.name,
        "request_settings": profile.request_settings_json,
        "role": profile.role,
        "use_case": profile.use_case,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _family_for_profile(
    session: Session,
    profile: ModelProfile,
) -> tuple[WorkflowFamily, WorkflowProfileCompatibility, bool]:
    family_id = compatibility_family_id(profile.id)
    fingerprint = _profile_fingerprint(profile)
    mapping = session.get(WorkflowProfileCompatibility, profile.id)
    if mapping and mapping.workflow_family_id != family_id:
        raise WorkflowCompatibilityConflict(
            f"profile {profile.id} is mapped to a non-deterministic workflow family"
        )

    family = session.get(WorkflowFamily, family_id)
    if family and not mapping:
        owner = session.scalar(
            select(WorkflowProfileCompatibility).where(
                WorkflowProfileCompatibility.workflow_family_id == family_id
            )
        )
        if not owner or owner.model_profile_id != profile.id:
            raise WorkflowCompatibilityConflict(
                f"workflow family id collision for legacy profile {profile.id}"
            )

    created = family is None
    if family is None:
        family = WorkflowFamily(
            id=family_id,
            name=profile.name,
            description="",
            use_case=profile.use_case,
            tags_json=[],
            enabled=True,
            archived=False,
        )
        session.add(family)
        session.flush()
    elif mapping and mapping.source_fingerprint_sha256 != fingerprint:
        # Legacy profile writes remain authoritative during the compatibility
        # window. Workflow-first writes use the explicit mirror functions below.
        family.name = profile.name
        family.use_case = profile.use_case

    if mapping is None:
        mapping = WorkflowProfileCompatibility(
            model_profile_id=profile.id,
            workflow_family_id=family.id,
            source_fingerprint_sha256=fingerprint,
        )
        session.add(mapping)
    else:
        mapping.source_fingerprint_sha256 = fingerprint
    return family, mapping, created


def _ensure_profile_variants(
    session: Session,
    profile: ModelProfile,
    family: WorkflowFamily,
) -> int:
    created = 0
    for variant_key, operation, label in _PROFILE_VARIANTS.get(profile.role, ()):
        definition_id = compatibility_definition_id(profile.id, variant_key)
        definition = session.get(WorkflowDefinition, definition_id)
        if definition is None:
            definition = WorkflowDefinition(
                id=definition_id,
                family_id=family.id,
                variant_key=variant_key,
                name=f"{profile.name} - {label}",
                operation=operation,
                description="",
            )
            session.add(definition)
            created += 1
            continue
        if (
            definition.family_id != family.id
            or definition.variant_key != variant_key
            or definition.operation != operation
        ):
            raise WorkflowCompatibilityConflict(
                f"workflow definition id collision for legacy profile {profile.id}"
            )
        if definition.current_revision_id is None:
            definition.name = f"{profile.name} - {label}"
    return created


def _ensure_profile_preferences(
    session: Session,
    profile: ModelProfile,
    family: WorkflowFamily,
) -> int:
    created = 0
    for capability in _PROFILE_CAPABILITIES.get(profile.role, ()):
        preference = session.scalar(
            select(WorkflowPreference).where(
                WorkflowPreference.workflow_family_id == family.id,
                WorkflowPreference.selector_capability == capability,
            )
        )
        if preference is not None:
            continue
        preference_id = _preference_id(profile.id, capability)
        collision = session.get(WorkflowPreference, preference_id)
        if collision is not None:
            raise WorkflowCompatibilityConflict(
                f"workflow preference id collision for legacy profile {profile.id}"
            )
        session.add(
            WorkflowPreference(
                id=preference_id,
                workflow_family_id=family.id,
                selector_capability=capability,
                enabled=True,
                is_default=False,
            )
        )
        created += 1
    return created


def _reconcile_compatibility_defaults(session: Session, profiles: list[ModelProfile]) -> None:
    mappings = list(session.scalars(select(WorkflowProfileCompatibility)).all())
    compatibility_family_ids = {mapping.workflow_family_id for mapping in mappings}
    generated_preferences = (
        list(
            session.scalars(
                select(WorkflowPreference).where(
                    WorkflowPreference.workflow_family_id.in_(compatibility_family_ids)
                )
            ).all()
        )
        if compatibility_family_ids
        else []
    )
    for preference in generated_preferences:
        preference.is_default = False
    session.flush()

    profiles_by_role: dict[str, list[ModelProfile]] = {}
    for profile in profiles:
        profiles_by_role.setdefault(profile.role, []).append(profile)
    family_by_profile = {
        mapping.model_profile_id: mapping.workflow_family_id for mapping in mappings
    }
    for role, capabilities in _PROFILE_CAPABILITIES.items():
        defaults = [profile for profile in profiles_by_role.get(role, []) if profile.is_default]
        defaults.sort(key=lambda item: (item.updated_at, item.id), reverse=True)
        if not defaults:
            continue
        winner_family_id = family_by_profile.get(defaults[0].id)
        if not winner_family_id:
            continue
        for capability in capabilities:
            real_default = session.scalar(
                select(WorkflowPreference).where(
                    WorkflowPreference.selector_capability == capability,
                    WorkflowPreference.is_default.is_(True),
                    WorkflowPreference.workflow_family_id.not_in(compatibility_family_ids),
                )
            )
            if real_default:
                continue
            winner = session.scalar(
                select(WorkflowPreference).where(
                    WorkflowPreference.workflow_family_id == winner_family_id,
                    WorkflowPreference.selector_capability == capability,
                )
            )
            if winner:
                winner.is_default = True


def _backfill_chat_selections(session: Session) -> int:
    created = 0
    mappings = {
        mapping.model_profile_id: mapping.workflow_family_id
        for mapping in session.scalars(select(WorkflowProfileCompatibility)).all()
    }
    profiles = {profile.id: profile for profile in session.scalars(select(ModelProfile)).all()}
    for chat in session.scalars(select(Chat).order_by(Chat.created_at, Chat.id)).all():
        for capability, field in _CHAT_PROFILE_FIELDS.items():
            existing = session.scalar(
                select(ChatWorkflowSelection).where(
                    ChatWorkflowSelection.chat_id == chat.id,
                    ChatWorkflowSelection.selector_capability == capability,
                )
            )
            if existing is not None:
                continue
            legacy_profile_id = getattr(chat, field)
            if legacy_profile_id is None:
                continue
            if legacy_profile_id == AUTO_PROFILE_ID:
                mode = "automatic"
                family_id = None
            else:
                profile = profiles.get(legacy_profile_id)
                family_id = mappings.get(legacy_profile_id)
                if (
                    profile is None
                    or family_id is None
                    or profile.role != _CAPABILITY_ROLE[capability]
                ):
                    continue
                mode = "family"
            session.add(
                ChatWorkflowSelection(
                    id=_selection_id(chat.id, capability, "chat"),
                    chat_id=chat.id,
                    selector_capability=capability,
                    mode=mode,
                    workflow_family_id=family_id,
                )
            )
            created += 1
    return created


def _backfill_project_selections(session: Session) -> int:
    created = 0
    for project in session.scalars(select(Project).order_by(Project.created_at, Project.id)).all():
        for capability, field in _PROJECT_REVISION_FIELDS.items():
            existing = session.scalar(
                select(ProjectWorkflowSelection).where(
                    ProjectWorkflowSelection.project_id == project.id,
                    ProjectWorkflowSelection.selector_capability == capability,
                )
            )
            if existing is not None:
                continue
            revision_id = getattr(project, field)
            if revision_id is None:
                continue
            session.add(
                ProjectWorkflowSelection(
                    id=_selection_id(project.id, capability, "project"),
                    project_id=project.id,
                    selector_capability=capability,
                    mode="revision",
                    workflow_revision_id=revision_id,
                )
            )
            created += 1
    return created


def ensure_legacy_profile_workflow(
    session: Session,
    profile: ModelProfile,
) -> WorkflowFamily:
    """Ensure one profile's stable family, variants, and selector preferences."""

    family, _, _ = _family_for_profile(session, profile)
    _ensure_profile_variants(session, profile, family)
    _ensure_profile_preferences(session, profile, family)
    session.flush()
    profiles = list(
        session.scalars(
            select(ModelProfile).order_by(ModelProfile.created_at, ModelProfile.id)
        ).all()
    )
    _reconcile_compatibility_defaults(session, profiles)
    session.flush()
    return family


def reconcile_legacy_workflow_compatibility(
    session: Session,
) -> WorkflowCompatibilityReport:
    """Idempotently mirror legacy choices without interpreting ambiguous data."""

    profiles = list(
        session.scalars(
            select(ModelProfile).order_by(ModelProfile.created_at, ModelProfile.id)
        ).all()
    )
    families_created = 0
    definitions_created = 0
    preferences_created = 0
    for profile in profiles:
        family, _, family_created = _family_for_profile(session, profile)
        families_created += int(family_created)
        definitions_created += _ensure_profile_variants(session, profile, family)
        preferences_created += _ensure_profile_preferences(session, profile, family)
    session.flush()
    _reconcile_compatibility_defaults(session, profiles)
    chat_selections_created = _backfill_chat_selections(session)
    project_selections_created = _backfill_project_selections(session)
    session.flush()
    return WorkflowCompatibilityReport(
        profiles=len(profiles),
        families_created=families_created,
        definitions_created=definitions_created,
        preferences_created=preferences_created,
        chat_selections_created=chat_selections_created,
        project_selections_created=project_selections_created,
    )


def resolve_chat_workflow_selection(
    session: Session,
    chat: Chat,
    capability: ChatSelectorCapability,
) -> ResolvedChatWorkflowSelection:
    selection = session.scalar(
        select(ChatWorkflowSelection).where(
            ChatWorkflowSelection.chat_id == chat.id,
            ChatWorkflowSelection.selector_capability == capability,
        )
    )
    legacy_profile_id = getattr(chat, _CHAT_PROFILE_FIELDS[capability])
    if selection is None:
        return ResolvedChatWorkflowSelection(
            capability=capability,
            mode="legacy",
            profile_id=legacy_profile_id,
            workflow_family_id=None,
        )
    if selection.mode == "automatic":
        return ResolvedChatWorkflowSelection(
            capability=capability,
            mode="automatic",
            profile_id=AUTO_PROFILE_ID,
            workflow_family_id=None,
        )
    if not selection.workflow_family_id:
        raise WorkflowSelectionInvalid(capability=capability, reason="family_missing")
    mapping = session.scalar(
        select(WorkflowProfileCompatibility).where(
            WorkflowProfileCompatibility.workflow_family_id == selection.workflow_family_id
        )
    )
    if mapping is None:
        family = session.get(WorkflowFamily, selection.workflow_family_id)
        preference = session.scalar(
            select(WorkflowPreference).where(
                WorkflowPreference.workflow_family_id == selection.workflow_family_id,
                WorkflowPreference.selector_capability == capability,
            )
        )
        if family is None or preference is None:
            raise WorkflowSelectionInvalid(capability=capability, reason="family_not_bound")
        return ResolvedChatWorkflowSelection(
            capability=capability,
            mode="family",
            profile_id=None,
            workflow_family_id=selection.workflow_family_id,
        )
    profile = session.get(ModelProfile, mapping.model_profile_id)
    if profile is None or profile.role != _CAPABILITY_ROLE[capability]:
        raise WorkflowSelectionInvalid(capability=capability, reason="profile_incompatible")
    return ResolvedChatWorkflowSelection(
        capability=capability,
        mode="family",
        profile_id=profile.id,
        workflow_family_id=selection.workflow_family_id,
    )


def resolve_project_workflow_selection(
    session: Session,
    project: Project,
    capability: ProjectSelectorCapability,
) -> ResolvedProjectWorkflowSelection:
    selection = session.scalar(
        select(ProjectWorkflowSelection).where(
            ProjectWorkflowSelection.project_id == project.id,
            ProjectWorkflowSelection.selector_capability == capability,
        )
    )
    legacy_revision_id = getattr(project, _PROJECT_REVISION_FIELDS[capability])
    if selection is None:
        return ResolvedProjectWorkflowSelection(
            capability=capability,
            mode="legacy",
            workflow_family_id=None,
            workflow_revision_id=legacy_revision_id,
        )
    return ResolvedProjectWorkflowSelection(
        capability=capability,
        mode=selection.mode,  # type: ignore[arg-type]
        workflow_family_id=selection.workflow_family_id,
        workflow_revision_id=selection.workflow_revision_id,
    )


def mirror_legacy_chat_workflow_selections(
    session: Session,
    chat: Chat,
    capabilities: Iterable[ChatSelectorCapability] = _CHAT_PROFILE_FIELDS,
) -> None:
    """Mirror an intentional legacy chat write where a local bridge exists."""

    for capability in capabilities:
        field = _CHAT_PROFILE_FIELDS[capability]
        legacy_profile_id = getattr(chat, field)
        selection = session.scalar(
            select(ChatWorkflowSelection).where(
                ChatWorkflowSelection.chat_id == chat.id,
                ChatWorkflowSelection.selector_capability == capability,
            )
        )
        if legacy_profile_id is None:
            if selection is not None:
                session.delete(selection)
            continue
        family_id: str | None = None
        mode = "automatic"
        if legacy_profile_id != AUTO_PROFILE_ID:
            profile = session.get(ModelProfile, legacy_profile_id)
            if profile is None or profile.role != _CAPABILITY_ROLE[capability]:
                raise WorkflowSelectionInvalid(capability=capability, reason="profile_incompatible")
            family = ensure_legacy_profile_workflow(session, profile)
            family_id = family.id
            mode = "family"
        if selection is None:
            selection = ChatWorkflowSelection(
                id=_selection_id(chat.id, capability, "chat"),
                chat_id=chat.id,
                selector_capability=capability,
                mode=mode,
                workflow_family_id=family_id,
            )
            session.add(selection)
        else:
            selection.mode = mode
            selection.workflow_family_id = family_id
    session.flush()


def mirror_legacy_project_workflow_selections(
    session: Session,
    project: Project,
    capabilities: Iterable[ProjectSelectorCapability] = _PROJECT_REVISION_FIELDS,
) -> None:
    """Mirror intentional legacy project revision writes without widening pins."""

    for capability in capabilities:
        field = _PROJECT_REVISION_FIELDS[capability]
        revision_id = getattr(project, field)
        selection = session.scalar(
            select(ProjectWorkflowSelection).where(
                ProjectWorkflowSelection.project_id == project.id,
                ProjectWorkflowSelection.selector_capability == capability,
            )
        )
        if revision_id is None:
            if selection is not None:
                session.delete(selection)
            continue
        if selection is None:
            selection = ProjectWorkflowSelection(
                id=_selection_id(project.id, capability, "project"),
                project_id=project.id,
                selector_capability=capability,
                mode="revision",
                workflow_revision_id=revision_id,
            )
            session.add(selection)
        else:
            selection.mode = "revision"
            selection.workflow_family_id = None
            selection.workflow_revision_id = revision_id
    session.flush()


def copy_chat_workflow_selections(session: Session, source: Chat, target: Chat) -> None:
    """Copy workflow-first choices when a chat-like session is cloned."""

    selections = list(
        session.scalars(
            select(ChatWorkflowSelection).where(ChatWorkflowSelection.chat_id == source.id)
        ).all()
    )
    copied_capabilities: set[ChatSelectorCapability] = set()
    for selection in selections:
        capability = selection.selector_capability
        if capability not in _CHAT_PROFILE_FIELDS:
            continue
        typed_capability: ChatSelectorCapability = capability
        session.add(
            ChatWorkflowSelection(
                id=_selection_id(target.id, typed_capability, "chat"),
                chat_id=target.id,
                selector_capability=typed_capability,
                mode=selection.mode,
                workflow_family_id=selection.workflow_family_id,
            )
        )
        copied_capabilities.add(typed_capability)
    missing = [
        capability for capability in _CHAT_PROFILE_FIELDS if capability not in copied_capabilities
    ]
    if missing:
        mirror_legacy_chat_workflow_selections(session, target, missing)
    session.flush()


def retire_legacy_profile_workflow(session: Session, profile: ModelProfile) -> None:
    """Detach one profile bridge without leaving chats pointed at an orphan."""

    mapping = session.get(WorkflowProfileCompatibility, profile.id)
    family = (
        session.get(WorkflowFamily, mapping.workflow_family_id) if mapping is not None else None
    )
    family_id = family.id if family is not None else None
    affected_chats = list(session.scalars(select(Chat)).all())
    for chat in affected_chats:
        changed: list[ChatSelectorCapability] = []
        for capability, field in _CHAT_PROFILE_FIELDS.items():
            if getattr(chat, field) == profile.id:
                setattr(chat, field, AUTO_PROFILE_ID)
                changed.append(capability)
        if changed:
            mirror_legacy_chat_workflow_selections(session, chat, changed)

    if family_id:
        selections = list(
            session.scalars(
                select(ChatWorkflowSelection).where(
                    ChatWorkflowSelection.workflow_family_id == family_id
                )
            ).all()
        )
        for selection in selections:
            selection.mode = "automatic"
            selection.workflow_family_id = None

    if mapping is not None:
        session.delete(mapping)
    if family is not None:
        definitions = list(
            session.scalars(
                select(WorkflowDefinition).where(WorkflowDefinition.family_id == family.id)
            ).all()
        )
        if all(not definition.revisions for definition in definitions):
            for definition in definitions:
                session.delete(definition)
            session.flush()
            session.delete(family)
    session.flush()
