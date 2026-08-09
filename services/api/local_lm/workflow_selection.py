from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .domain import Operation
from .models import (
    ModelInstall,
    ModelProfile,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowProfileCompatibility,
    WorkflowRevision,
)

WorkflowSelectorCapability = Literal["chat", "vision", "image", "video"]
WorkflowSelectionMode = Literal["explicit", "default", "automatic"]
LegacyRevisionResolver = Callable[[Session, ModelProfile, Operation], WorkflowRevision | None]

_CAPABILITY_ROLE: dict[WorkflowSelectorCapability, str] = {
    "chat": "chat",
    "vision": "chat",
    "image": "image",
    "video": "video",
}
_CAPABILITY_OPERATIONS: dict[WorkflowSelectorCapability, frozenset[Operation]] = {
    "chat": frozenset({Operation.TEXT}),
    "vision": frozenset({Operation.TEXT}),
    "image": frozenset({Operation.TEXT_TO_IMAGE, Operation.IMAGE_TO_IMAGE}),
    "video": frozenset({Operation.TEXT_TO_VIDEO, Operation.IMAGE_TO_VIDEO}),
}
_TERM_ALIASES = {
    "animation": "video",
    "animations": "video",
    "artwork": "image",
    "cinematic": "video",
    "coding": "code",
    "debugging": "debug",
    "developer": "code",
    "development": "code",
    "draw": "image",
    "drawing": "image",
    "fiction": "writing",
    "illustration": "image",
    "illustrations": "image",
    "images": "image",
    "motion": "video",
    "narrative": "writing",
    "photo": "image",
    "photos": "image",
    "photography": "image",
    "picture": "image",
    "pictures": "image",
    "programming": "code",
    "prose": "writing",
    "software": "code",
    "stories": "writing",
    "story": "writing",
    "storytelling": "writing",
    "summarization": "summarize",
    "summary": "summarize",
    "translation": "translate",
    "translator": "translate",
    "troubleshooting": "debug",
    "videos": "video",
    "write": "writing",
    "writer": "writing",
}
_STOP_WORDS = {
    "about",
    "and",
    "are",
    "for",
    "from",
    "into",
    "model",
    "that",
    "the",
    "this",
    "use",
    "with",
    "workflow",
}
_DIGEST = re.compile(r"[0-9a-f]{64}")


class WorkflowFamilySelectionError(ValueError):
    def __init__(
        self,
        *,
        capability: WorkflowSelectorCapability,
        operation: Operation,
        reason: str,
        workflow_family_id: str | None = None,
    ) -> None:
        self.capability = capability
        self.operation = operation
        self.reason = reason
        self.workflow_family_id = workflow_family_id
        target = f" {workflow_family_id}" if workflow_family_id else ""
        super().__init__(
            f"{capability} workflow family{target} cannot run {operation.value}: {reason}"
        )


@dataclass(frozen=True)
class ResolvedWorkflowFamily:
    mode: WorkflowSelectionMode
    capability: WorkflowSelectorCapability
    operation: Operation
    workflow_family_id: str
    workflow_family_name: str
    workflow_definition_id: str | None
    workflow_revision_id: str | None
    workflow_activation_id: str | None
    profile_id: str | None
    compatibility: bool
    is_default: bool
    score: int
    matched_terms: tuple[str, ...]

    def provenance(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "workflow_family_id": self.workflow_family_id,
            "workflow_family_name": self.workflow_family_name,
            "workflow_definition_id": self.workflow_definition_id,
            "workflow_revision_id": self.workflow_revision_id,
            "workflow_activation_id": self.workflow_activation_id,
            "profile_id": self.profile_id,
            "compatibility": self.compatibility,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True)
class _Candidate:
    family: WorkflowFamily
    preference: WorkflowPreference
    definition: WorkflowDefinition | None
    revision: WorkflowRevision | None
    activation: WorkflowActivation | None
    profile: ModelProfile | None
    compatibility: bool
    score: int
    matched_terms: tuple[str, ...]


def resolve_workflow_family(
    session: Session,
    *,
    capability: WorkflowSelectorCapability,
    operation: Operation,
    mode: WorkflowSelectionMode,
    workflow_family_id: str | None = None,
    prompt: str = "",
    engine: str | None = None,
    required_capabilities: Iterable[str] = (),
    legacy_revision_resolver: LegacyRevisionResolver | None = None,
) -> ResolvedWorkflowFamily:
    """Resolve one broad selector to an exact operation variant without guessing."""

    if operation not in _CAPABILITY_OPERATIONS[capability]:
        raise _error(capability, operation, "selector_operation_mismatch", workflow_family_id)
    required = frozenset(item.strip() for item in required_capabilities if item.strip())

    if mode == "explicit":
        if not workflow_family_id:
            raise _error(capability, operation, "family_missing")
        family = session.get(WorkflowFamily, workflow_family_id)
        if family is None:
            raise _error(capability, operation, "family_not_found", workflow_family_id)
        preference = _preference(session, family.id, capability)
        candidate = _candidate(
            session,
            family,
            preference,
            capability=capability,
            operation=operation,
            prompt=prompt,
            engine=engine,
            required_capabilities=required,
            legacy_revision_resolver=legacy_revision_resolver,
        )
        return _resolved(candidate, mode)

    preferences = list(
        session.scalars(
            select(WorkflowPreference)
            .where(
                WorkflowPreference.selector_capability == capability,
                WorkflowPreference.enabled.is_(True),
            )
            .order_by(WorkflowPreference.sort_order, WorkflowPreference.id)
        ).all()
    )
    if mode == "default":
        defaults = [preference for preference in preferences if preference.is_default]
        if not defaults:
            raise _error(capability, operation, "default_missing")
        preference = defaults[0]
        family = session.get(WorkflowFamily, preference.workflow_family_id)
        if family is None:
            raise _error(
                capability,
                operation,
                "family_not_found",
                preference.workflow_family_id,
            )
        candidate = _candidate(
            session,
            family,
            preference,
            capability=capability,
            operation=operation,
            prompt=prompt,
            engine=engine,
            required_capabilities=required,
            legacy_revision_resolver=legacy_revision_resolver,
        )
        return _resolved(candidate, mode)

    candidates: list[_Candidate] = []
    for preference in preferences:
        family = session.get(WorkflowFamily, preference.workflow_family_id)
        if family is None:
            continue
        if not _automatic_family_eligible(session, family, preference):
            continue
        try:
            candidates.append(
                _candidate(
                    session,
                    family,
                    preference,
                    capability=capability,
                    operation=operation,
                    prompt=prompt,
                    engine=engine,
                    required_capabilities=required,
                    legacy_revision_resolver=legacy_revision_resolver,
                )
            )
        except WorkflowFamilySelectionError:
            continue
    if not candidates:
        raise _error(capability, operation, "no_ready_workflow")
    candidates.sort(
        key=lambda item: (
            -item.score,
            not item.preference.is_default,
            item.preference.sort_order,
            item.family.id,
        )
    )
    return _resolved(candidates[0], mode)


def _automatic_family_eligible(
    session: Session,
    family: WorkflowFamily,
    preference: WorkflowPreference,
) -> bool:
    if preference.is_default or family.use_case.strip():
        return True
    if any(isinstance(tag, str) and tag.strip() for tag in family.tags_json):
        return True
    return (
        session.scalar(
            select(WorkflowProfileCompatibility.model_profile_id).where(
                WorkflowProfileCompatibility.workflow_family_id == family.id
            )
        )
        is not None
    )


def _candidate(
    session: Session,
    family: WorkflowFamily,
    preference: WorkflowPreference | None,
    *,
    capability: WorkflowSelectorCapability,
    operation: Operation,
    prompt: str,
    engine: str | None,
    required_capabilities: frozenset[str],
    legacy_revision_resolver: LegacyRevisionResolver | None,
) -> _Candidate:
    if family.archived:
        raise _error(capability, operation, "family_archived", family.id)
    if not family.enabled:
        raise _error(capability, operation, "family_disabled", family.id)
    if preference is None:
        raise _error(capability, operation, "selector_not_enabled", family.id)
    if not preference.enabled:
        raise _error(capability, operation, "selector_disabled", family.id)

    mapping = session.scalar(
        select(WorkflowProfileCompatibility).where(
            WorkflowProfileCompatibility.workflow_family_id == family.id
        )
    )
    score, matched_terms = _intent_score(family, prompt)
    if mapping is not None:
        profile = session.get(ModelProfile, mapping.model_profile_id)
        if profile is None or profile.role != _CAPABILITY_ROLE[capability]:
            raise _error(capability, operation, "profile_incompatible", family.id)
        _validate_compatibility_profile(
            session,
            profile,
            capability=capability,
            operation=operation,
            engine=engine,
            workflow_family_id=family.id,
        )
        revision = (
            legacy_revision_resolver(session, profile, operation)
            if legacy_revision_resolver is not None
            else None
        )
        if operation != Operation.TEXT and revision is None:
            raise _error(capability, operation, "operation_unavailable", family.id)
        activation = _validate_revision(
            session,
            revision,
            capability=capability,
            operation=operation,
            engine=engine,
            workflow_family_id=family.id,
            required_capabilities=required_capabilities,
        )
        definition = session.get(WorkflowDefinition, revision.workflow_id) if revision else None
        return _Candidate(
            family,
            preference,
            definition,
            revision,
            activation,
            profile,
            True,
            score,
            matched_terms,
        )

    definitions = list(
        session.scalars(
            select(WorkflowDefinition).where(
                WorkflowDefinition.family_id == family.id,
                WorkflowDefinition.operation == operation.value,
            )
        ).all()
    )
    viable: list[tuple[WorkflowDefinition, WorkflowRevision, WorkflowActivation | None]] = []
    failure_reasons: list[str] = []
    for definition in definitions:
        if not definition.current_revision_id:
            failure_reasons.append("current_revision_missing")
            continue
        revision = session.get(WorkflowRevision, definition.current_revision_id)
        if revision is None or revision.workflow_id != definition.id:
            failure_reasons.append("current_revision_invalid")
            continue
        try:
            activation = _validate_revision(
                session,
                revision,
                capability=capability,
                operation=operation,
                engine=engine,
                workflow_family_id=family.id,
                required_capabilities=required_capabilities,
            )
        except WorkflowFamilySelectionError as exc:
            failure_reasons.append(exc.reason)
            continue
        viable.append((definition, revision, activation))
    if not viable:
        reason = failure_reasons[0] if len(set(failure_reasons)) == 1 else "operation_unavailable"
        if not definitions:
            reason = "variant_missing"
        raise _error(capability, operation, reason, family.id)
    if len(viable) != 1:
        raise _error(capability, operation, "variant_ambiguous", family.id)
    definition, revision, activation = viable[0]
    profile = _activation_profile(
        session,
        activation,
        role=_CAPABILITY_ROLE[capability],
        engine=engine,
    )
    if operation == Operation.TEXT and profile is None:
        raise _error(capability, operation, "profile_binding_missing", family.id)
    return _Candidate(
        family,
        preference,
        definition,
        revision,
        activation,
        profile,
        False,
        score,
        matched_terms,
    )


def _validate_compatibility_profile(
    session: Session,
    profile: ModelProfile,
    *,
    capability: WorkflowSelectorCapability,
    operation: Operation,
    engine: str | None,
    workflow_family_id: str,
) -> None:
    if engine is not None and profile.engine != engine:
        raise _error(capability, operation, "engine_mismatch", workflow_family_id)
    if profile.model_install_id is None:
        if engine is not None and engine != "mock":
            raise _error(capability, operation, "model_unavailable", workflow_family_id)
        return
    install = session.get(ModelInstall, profile.model_install_id)
    if (
        install is None
        or not install.active
        or install.role != profile.role
        or install.engine != profile.engine
    ):
        raise _error(capability, operation, "model_unavailable", workflow_family_id)


def _validate_revision(
    session: Session,
    revision: WorkflowRevision | None,
    *,
    capability: WorkflowSelectorCapability,
    operation: Operation,
    engine: str | None,
    workflow_family_id: str,
    required_capabilities: frozenset[str],
) -> WorkflowActivation | None:
    if revision is None:
        return None
    definition = session.get(WorkflowDefinition, revision.workflow_id)
    if definition is None or definition.operation != operation.value:
        raise _error(capability, operation, "operation_mismatch", workflow_family_id)
    if engine is not None and revision.engine != engine:
        raise _error(capability, operation, "engine_mismatch", workflow_family_id)
    if operation != Operation.TEXT and engine != "mock" and not revision.api_graph_json:
        raise _error(capability, operation, "revision_not_executable", workflow_family_id)
    if not revision.trusted:
        raise _error(capability, operation, "revision_untrusted", workflow_family_id)
    if not required_capabilities.issubset(set(revision.capabilities_json)):
        raise _error(capability, operation, "capability_mismatch", workflow_family_id)
    if revision.dependency_contract_sha256 is None:
        return None
    activation = session.scalar(
        select(WorkflowActivation).where(
            WorkflowActivation.workflow_revision_id == revision.id,
            WorkflowActivation.is_active.is_(True),
            WorkflowActivation.state == "ready",
            WorkflowActivation.invalidated_at.is_(None),
        )
    )
    launch_sha256 = (
        activation.details_json.get("launch_sha256")
        if activation is not None and isinstance(activation.details_json, dict)
        else None
    )
    if (
        activation is None
        or activation.dependency_contract_sha256 != revision.dependency_contract_sha256
        or not isinstance(launch_sha256, str)
        or _DIGEST.fullmatch(launch_sha256) is None
    ):
        raise _error(capability, operation, "activation_not_ready", workflow_family_id)
    return activation


def _activation_profile(
    session: Session,
    activation: WorkflowActivation | None,
    *,
    role: str,
    engine: str | None,
) -> ModelProfile | None:
    if activation is None:
        return None
    profile_ids = list(
        session.scalars(
            select(WorkflowDependencyBinding.model_profile_id).where(
                WorkflowDependencyBinding.workflow_activation_id == activation.id,
                WorkflowDependencyBinding.model_profile_id.is_not(None),
            )
        ).all()
    )
    profiles: list[ModelProfile] = []
    for profile_id in sorted({item for item in profile_ids if item is not None}):
        profile = session.get(ModelProfile, profile_id)
        if profile is None or profile.role != role:
            continue
        if engine is not None and profile.engine != engine:
            continue
        if profile.model_install_id:
            install = session.get(ModelInstall, profile.model_install_id)
            if install is None or not install.active or install.engine != profile.engine:
                continue
        profiles.append(profile)
    return profiles[0] if len(profiles) == 1 else None


def _preference(
    session: Session,
    workflow_family_id: str,
    capability: WorkflowSelectorCapability,
) -> WorkflowPreference | None:
    return session.scalar(
        select(WorkflowPreference).where(
            WorkflowPreference.workflow_family_id == workflow_family_id,
            WorkflowPreference.selector_capability == capability,
        )
    )


def _intent_score(family: WorkflowFamily, prompt: str) -> tuple[int, tuple[str, ...]]:
    prompt_text = prompt.casefold()
    prompt_terms = _selection_terms(prompt_text)
    use_case = family.use_case.strip().casefold()
    use_case_terms = _selection_terms(use_case)
    matched_terms = tuple(sorted(prompt_terms & use_case_terms))
    score = len(matched_terms) * 10
    if use_case and use_case in prompt_text:
        score += 25
    tags = [item for item in family.tags_json if isinstance(item, str)]
    labels = " ".join([family.name, *tags]).casefold()
    score += len(prompt_terms & _selection_terms(labels))
    return score, matched_terms


def _selection_terms(value: str) -> set[str]:
    return {
        _TERM_ALIASES.get(term, term)
        for term in re.findall(r"[a-z0-9][a-z0-9+#.-]{2,}", value)
        if term not in _STOP_WORDS
    }


def _resolved(candidate: _Candidate, mode: WorkflowSelectionMode) -> ResolvedWorkflowFamily:
    return ResolvedWorkflowFamily(
        mode=mode,
        capability=candidate.preference.selector_capability,  # type: ignore[arg-type]
        operation=Operation(candidate.definition.operation)
        if candidate.definition
        else Operation.TEXT,
        workflow_family_id=candidate.family.id,
        workflow_family_name=candidate.family.name,
        workflow_definition_id=candidate.definition.id if candidate.definition else None,
        workflow_revision_id=candidate.revision.id if candidate.revision else None,
        workflow_activation_id=candidate.activation.id if candidate.activation else None,
        profile_id=candidate.profile.id if candidate.profile else None,
        compatibility=candidate.compatibility,
        is_default=candidate.preference.is_default,
        score=candidate.score,
        matched_terms=candidate.matched_terms,
    )


def _error(
    capability: WorkflowSelectorCapability,
    operation: Operation,
    reason: str,
    workflow_family_id: str | None = None,
) -> WorkflowFamilySelectionError:
    return WorkflowFamilySelectionError(
        capability=capability,
        operation=operation,
        reason=reason,
        workflow_family_id=workflow_family_id,
    )
