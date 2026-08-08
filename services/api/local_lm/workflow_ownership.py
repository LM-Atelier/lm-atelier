from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import (
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from .workflow_package_drafts import is_workflow_package_draft

WorkflowSelectorCapability = Literal["image", "video"]

_OPERATION_VARIANTS: dict[str, tuple[str, WorkflowSelectorCapability]] = {
    "text_to_image": ("create", "image"),
    "image_to_image": ("edit", "image"),
    "text_to_video": ("create", "video"),
    "image_to_video": ("animate", "video"),
}


class WorkflowOwnershipConflict(RuntimeError):
    """A deterministic ownership identity already names unrelated data."""


@dataclass(frozen=True)
class WorkflowOwnershipReport:
    definitions: int
    families_created: int
    assignments_created: int
    preferences_created: int


def workflow_family_id(definition_id: str) -> str:
    digest = hashlib.sha256(f"workflow-family:{definition_id}".encode()).hexdigest()
    return f"wffamily_{digest[:48]}"


def workflow_preference_id(family_id: str, capability: str) -> str:
    digest = hashlib.sha256(f"workflow-preference:{family_id}:{capability}".encode()).hexdigest()
    return f"wfpref_{digest[:32]}"


def ensure_workflow_family_ownership(
    session: Session,
    definition: WorkflowDefinition,
    revision: WorkflowRevision | None = None,
) -> WorkflowFamily | None:
    revision = revision or (
        session.get(WorkflowRevision, definition.current_revision_id)
        if definition.current_revision_id
        else None
    )
    if revision is not None and (
        revision.workflow_id != definition.id or revision.id != definition.current_revision_id
    ):
        raise WorkflowOwnershipConflict(
            f"workflow revision {revision.id} is not the current revision of {definition.id}"
        )
    taxonomy = _OPERATION_VARIANTS.get(definition.operation)
    if revision is None or taxonomy is None or is_workflow_package_draft(revision):
        return None
    variant_key, primary_capability = taxonomy
    family = session.get(WorkflowFamily, definition.family_id) if definition.family_id else None
    if definition.family_id and family is None:
        raise WorkflowOwnershipConflict(
            f"workflow {definition.id} references a missing family {definition.family_id}"
        )
    if family is None:
        family_id = workflow_family_id(definition.id)
        family = session.get(WorkflowFamily, family_id)
        if family is not None:
            claimed_definition = session.scalar(
                select(WorkflowDefinition.id).where(WorkflowDefinition.family_id == family.id)
            )
            if claimed_definition != definition.id:
                raise WorkflowOwnershipConflict(
                    f"workflow family id collision for definition {definition.id}"
                )
        else:
            family = WorkflowFamily(
                id=family_id,
                name=definition.name,
                description=definition.description,
                use_case="",
                tags_json=[],
                enabled=True,
                archived=False,
            )
            session.add(family)
            session.flush()
        definition.family_id = family.id
        definition.variant_key = variant_key
    executable_variants: list[WorkflowDefinition] = []
    candidates = session.scalars(
        select(WorkflowDefinition).where(
            WorkflowDefinition.family_id == family.id,
            WorkflowDefinition.operation == definition.operation,
            WorkflowDefinition.current_revision_id.is_not(None),
        )
    ).all()
    for candidate in candidates:
        candidate_revision = session.get(WorkflowRevision, candidate.current_revision_id)
        if candidate_revision is not None and not is_workflow_package_draft(candidate_revision):
            executable_variants.append(candidate)
    if len(executable_variants) > 1:
        raise WorkflowOwnershipConflict(
            f"workflow family {family.id} has ambiguous {definition.operation} variants"
        )
    for capability in [primary_capability]:
        existing = session.scalar(
            select(WorkflowPreference).where(
                WorkflowPreference.workflow_family_id == family.id,
                WorkflowPreference.selector_capability == capability,
            )
        )
        if existing is not None:
            continue
        preference_id = workflow_preference_id(family.id, capability)
        collision = session.get(WorkflowPreference, preference_id)
        if collision is not None:
            raise WorkflowOwnershipConflict(
                f"workflow preference id collision for family {family.id}"
            )
        session.add(
            WorkflowPreference(
                id=preference_id,
                workflow_family_id=family.id,
                selector_capability=capability,
                enabled=family.enabled and not family.archived,
                is_default=False,
            )
        )
    return family


def reconcile_workflow_family_ownership(session: Session) -> WorkflowOwnershipReport:
    definitions = list(
        session.scalars(
            select(WorkflowDefinition).order_by(
                WorkflowDefinition.created_at,
                WorkflowDefinition.id,
            )
        ).all()
    )
    families_before = session.scalar(select(func.count()).select_from(WorkflowFamily))
    preferences_before = session.scalar(select(func.count()).select_from(WorkflowPreference))
    assignments_before = sum(definition.family_id is not None for definition in definitions)
    for definition in definitions:
        if definition.family_id is None:
            ensure_workflow_family_ownership(session, definition)
    session.flush()
    families_after = session.scalar(select(func.count()).select_from(WorkflowFamily))
    preferences_after = session.scalar(select(func.count()).select_from(WorkflowPreference))
    assignments_after = sum(definition.family_id is not None for definition in definitions)
    return WorkflowOwnershipReport(
        definitions=len(definitions),
        families_created=int(families_after or 0) - int(families_before or 0),
        assignments_created=assignments_after - assignments_before,
        preferences_created=int(preferences_after or 0) - int(preferences_before or 0),
    )
