"""Page workflow families and variants after applying their complete filters."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Literal

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.orm import Session, aliased, selectinload

from .domain import Operation
from .models import (
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowProfileCompatibility,
    WorkflowRevision,
)
from .schemas import (
    WorkflowFamilyDependencySummaryOut,
    WorkflowFamilyOut,
    WorkflowFamilyVariantOut,
    WorkflowReadyRevisionOut,
    WorkflowSelectorCapability,
    WorkflowVariantReadiness,
)
from .workflow_family_dependencies import workflow_family_dependency_summaries
from .workflow_package_drafts import WORKFLOW_PACKAGE_DRAFT_MARKER

READINESS_ORDER: tuple[WorkflowVariantReadiness, ...] = (
    "ready",
    "setup_required",
    "review_required",
    "unavailable",
)

SELECTOR_OPERATIONS: dict[WorkflowSelectorCapability, tuple[str, ...]] = {
    "chat": ("text",),
    "image": ("text_to_image", "image_to_image"),
    "video": ("text_to_video", "image_to_video"),
    "vision": (),
}


def _visible_definitions() -> Select[tuple[WorkflowDefinition]]:
    current = aliased(WorkflowRevision)
    draft_type = func.json_type(current.dependencies_json, "$." + WORKFLOW_PACKAGE_DRAFT_MARKER)
    return (
        select(WorkflowDefinition)
        .outerjoin(
            current,
            and_(
                current.id == WorkflowDefinition.current_revision_id,
                current.workflow_id == WorkflowDefinition.id,
            ),
        )
        .where(or_(draft_type.is_(None), draft_type != "object"))
    )


@dataclass(frozen=True)
class VariantPage:
    variants: list[WorkflowFamilyVariantOut]
    count: int
    ready_count: int
    best_readiness: WorkflowVariantReadiness


def read_family_variants(
    session: Session,
    family_id: str,
    project: Callable[[WorkflowDefinition], WorkflowFamilyVariantOut],
    *,
    limit: int | None = None,
    offset: int = 0,
    operation: Operation | None = None,
    readiness: WorkflowVariantReadiness | None = None,
    capability: WorkflowSelectorCapability | None = None,
    workflow_ids: Sequence[str] = (),
) -> VariantPage:
    query = _visible_definitions().where(WorkflowDefinition.family_id == family_id)
    if workflow_ids:
        query = query.where(WorkflowDefinition.id.in_(workflow_ids))
    if operation is not None:
        query = query.where(WorkflowDefinition.operation == operation.value)
    query = query.order_by(
        WorkflowDefinition.operation,
        WorkflowDefinition.variant_key,
        WorkflowDefinition.id,
    )
    variants: list[WorkflowFamilyVariantOut] = []
    count = ready_count = 0
    best = len(READINESS_ORDER) - 1
    for definition in session.scalars(query.execution_options(yield_per=50)):
        value = project(definition)
        if capability is not None and not _serves_capability(value, capability):
            continue
        if readiness is not None and value.readiness != readiness:
            continue
        count += 1
        ready_count += value.readiness == "ready"
        best = min(best, READINESS_ORDER.index(value.readiness))
        if count > offset and (limit is None or len(variants) < limit):
            variants.append(value)
    return VariantPage(variants, count, ready_count, READINESS_ORDER[best])


def _serves_capability(
    variant: WorkflowFamilyVariantOut, capability: WorkflowSelectorCapability
) -> bool:
    if capability in variant.capabilities:
        return True
    return variant.operation in SELECTOR_OPERATIONS[capability]


def family_supported_selector_capabilities(
    session: Session, family_id: str
) -> list[WorkflowSelectorCapability]:
    """Read every operation a family offers without projecting its variant pages."""
    operations = set(
        session.scalars(
            _visible_definitions()
            .with_only_columns(WorkflowDefinition.operation)
            .where(WorkflowDefinition.family_id == family_id)
            .distinct()
        )
    )
    return [
        capability
        for capability, supported in SELECTOR_OPERATIONS.items()
        if operations.intersection(supported)
    ]


def read_family_page(
    session: Session,
    project: Callable[[WorkflowFamily], WorkflowFamilyOut],
    *,
    limit: int | None = None,
    offset: int = 0,
    search: str = "",
    selector_capability: WorkflowSelectorCapability | None = None,
    include_archived: bool = False,
    include_dependencies: bool = False,
    family_ids: Sequence[str] = (),
    workflow_ids: Sequence[str] = (),
    defaults_only: bool = False,
    enabled_only: bool = False,
    source: Literal["profile", "workflow"] | None = None,
    order: Literal["name", "readiness", "preference"] = "name",
    require_variants: bool = False,
) -> list[WorkflowFamilyOut]:
    query = select(WorkflowFamily).options(selectinload(WorkflowFamily.preferences))
    if not include_archived:
        query = query.where(WorkflowFamily.archived.is_(False))
    if family_ids:
        query = query.where(WorkflowFamily.id.in_(family_ids))
    if workflow_ids:
        query = query.where(
            WorkflowFamily.id.in_(
                _visible_definitions()
                .with_only_columns(WorkflowDefinition.family_id)
                .where(WorkflowDefinition.id.in_(workflow_ids))
            )
        )
    if enabled_only:
        query = query.where(WorkflowFamily.enabled.is_(True))
    preferences = select(WorkflowPreference.id).where(
        WorkflowPreference.workflow_family_id == WorkflowFamily.id,
    )
    if selector_capability is not None:
        preferences = preferences.where(
            WorkflowPreference.selector_capability == selector_capability,
        )
    if enabled_only:
        preferences = preferences.where(WorkflowPreference.enabled.is_(True))
    if defaults_only:
        preferences = preferences.where(WorkflowPreference.is_default.is_(True))
    if selector_capability is not None or defaults_only or enabled_only:
        query = query.where(preferences.exists())
    if source is not None:
        compatibility = (
            select(WorkflowProfileCompatibility.workflow_family_id)
            .where(
                WorkflowProfileCompatibility.workflow_family_id == WorkflowFamily.id,
            )
            .exists()
        )
        query = query.where(compatibility if source == "profile" else ~compatibility)
    if order == "preference":
        rank = select(func.min(WorkflowPreference.sort_order)).where(
            WorkflowPreference.workflow_family_id == WorkflowFamily.id,
        )
        if selector_capability is not None:
            rank = rank.where(WorkflowPreference.selector_capability == selector_capability)
        query = query.order_by(func.coalesce(rank.scalar_subquery(), 2**63 - 1))
    query = query.order_by(WorkflowFamily.name, WorkflowFamily.id)
    needle = search.strip().casefold()

    def matches() -> Iterator[WorkflowFamilyOut]:
        # Each readiness pass retains one family's bounded result. A large library
        # does not require keeping every graph or every family card for sorting.
        ranks = READINESS_ORDER if order == "readiness" else (None,)
        for target in ranks:
            for family in session.scalars(query.execution_options(yield_per=50)):
                summary = (
                    workflow_family_dependency_summaries(session, [family.id]).get(family.id)
                    if include_dependencies or needle
                    else None
                )
                if needle:
                    labels = [family.name, family.description, family.use_case, *family.tags_json]
                    if summary is not None:
                        labels.extend(summary.names)
                    if not any(
                        needle in label.casefold() for label in labels if isinstance(label, str)
                    ):
                        names = (
                            _visible_definitions()
                            .with_only_columns(WorkflowDefinition.name)
                            .where(
                                WorkflowDefinition.family_id == family.id,
                            )
                        )
                        if not any(needle in name.casefold() for name in session.scalars(names)):
                            continue
                value = project(family)
                if require_variants and value.variant_count == 0:
                    continue
                if target is not None and value.best_readiness != target:
                    continue
                if include_dependencies and summary is not None:
                    value.dependency_summary = WorkflowFamilyDependencySummaryOut(
                        dependency_count=summary.dependency_count,
                        names=list(summary.names),
                    )
                yield value

    return list(islice(islice(matches(), offset, None), limit))


def family_operation_choices(
    session: Session, *, include_archived: bool = False
) -> list[Operation]:
    query = (
        _visible_definitions()
        .with_only_columns(WorkflowDefinition.operation)
        .outerjoin(
            WorkflowFamily,
            WorkflowFamily.id == WorkflowDefinition.family_id,
        )
    )
    if not include_archived:
        query = query.where(or_(WorkflowFamily.id.is_(None), WorkflowFamily.archived.is_(False)))
    return [
        Operation(value)
        for value in session.scalars(query.distinct().order_by(WorkflowDefinition.operation))
    ]


def read_ready_revision_page(
    session: Session,
    project: Callable[
        [WorkflowDefinition, WorkflowFamily, WorkflowProfileCompatibility | None],
        WorkflowFamilyVariantOut,
    ],
    *,
    capability: WorkflowSelectorCapability,
    operation: Operation,
    limit: int,
    offset: int = 0,
    search: str = "",
    revision_ids: Sequence[str] = (),
) -> list[WorkflowReadyRevisionOut]:
    preference = (
        select(WorkflowPreference.id)
        .where(
            WorkflowPreference.workflow_family_id == WorkflowFamily.id,
            WorkflowPreference.selector_capability == capability,
            WorkflowPreference.enabled.is_(True),
        )
        .exists()
    )
    query = (
        _visible_definitions()
        .add_columns(WorkflowFamily, WorkflowProfileCompatibility)
        .join(WorkflowFamily, WorkflowFamily.id == WorkflowDefinition.family_id)
        .outerjoin(
            WorkflowProfileCompatibility,
            WorkflowProfileCompatibility.workflow_family_id == WorkflowFamily.id,
        )
        .where(
            WorkflowFamily.enabled.is_(True),
            WorkflowFamily.archived.is_(False),
            WorkflowDefinition.operation == operation.value,
            preference,
        )
        .order_by(
            WorkflowFamily.name,
            WorkflowFamily.id,
            WorkflowDefinition.name,
            WorkflowDefinition.id,
        )
    )
    if revision_ids:
        has_revision = (
            select(WorkflowRevision.id)
            .where(
                WorkflowRevision.id == WorkflowDefinition.current_revision_id,
            )
            .exists()
        )
        query = query.where(
            or_(
                WorkflowDefinition.current_revision_id.in_(revision_ids),
                and_(WorkflowProfileCompatibility.workflow_family_id.is_not(None), ~has_revision),
            )
        )
    needle = search.strip().casefold()
    exact = set(revision_ids)

    def matches() -> Iterator[WorkflowReadyRevisionOut]:
        seen: set[str] = set()
        for definition, family, compatibility in session.execute(
            query.execution_options(yield_per=50)
        ):
            if needle and not any(
                needle in value.casefold() for value in (family.name, definition.name)
            ):
                continue
            value = project(definition, family, compatibility)
            identifier = value.current_revision_id
            if (
                value.readiness != "ready"
                or identifier is None
                or value.current_revision_version is None
            ):
                continue
            if identifier in seen or (exact and identifier not in exact):
                continue
            seen.add(identifier)
            yield WorkflowReadyRevisionOut(
                family_id=family.id,
                family_name=family.name,
                workflow_id=definition.id,
                workflow_name=definition.name,
                revision_id=identifier,
                revision_version=value.current_revision_version,
                operation=value.operation,
            )

    return list(islice(islice(matches(), offset, None), limit))
