"""Read lightweight workflow list metadata and one selected definition."""

from __future__ import annotations

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.orm import Session, aliased, selectinload

from .models import WorkflowDefinition, WorkflowRevision
from .schemas import WorkflowRevisionChoiceOut, WorkflowRevisionSchemaOut, WorkflowSummaryOut
from .workflow_package_drafts import WORKFLOW_PACKAGE_DRAFT_MARKER, is_workflow_package_draft


def list_workflow_summaries(session: Session) -> list[WorkflowSummaryOut]:
    counts = (
        select(
            WorkflowRevision.workflow_id, func.count(WorkflowRevision.id).label("revision_count")
        )
        .group_by(WorkflowRevision.workflow_id)
        .subquery()
    )
    current = aliased(WorkflowRevision)
    draft_type = func.json_type(current.dependencies_json, "$." + WORKFLOW_PACKAGE_DRAFT_MARKER)
    statement = (
        select(
            WorkflowDefinition.id,
            WorkflowDefinition.family_id,
            WorkflowDefinition.name,
            WorkflowDefinition.operation,
            WorkflowDefinition.description,
            WorkflowDefinition.current_revision_id,
            WorkflowDefinition.created_at,
            WorkflowDefinition.updated_at,
            func.coalesce(counts.c.revision_count, 0).label("revision_count"),
        )
        .outerjoin(counts, counts.c.workflow_id == WorkflowDefinition.id)
        .outerjoin(
            current,
            and_(
                current.id == WorkflowDefinition.current_revision_id,
                current.workflow_id == WorkflowDefinition.id,
            ),
        )
        .where(or_(draft_type.is_(None), draft_type != "object"))
        .order_by(WorkflowDefinition.name, WorkflowDefinition.id)
        .execution_options(autoflush=False)
    )
    return [WorkflowSummaryOut(**row) for row in session.execute(statement).mappings()]


def load_workflow_detail(session: Session, workflow_id: str) -> WorkflowDefinition | None:
    definition = session.scalar(
        select(WorkflowDefinition)
        .where(WorkflowDefinition.id == workflow_id)
        .options(selectinload(WorkflowDefinition.revisions))
        .execution_options(autoflush=False)
    )
    if definition is None:
        return None
    current = next(
        (
            revision
            for revision in definition.revisions
            if revision.id == definition.current_revision_id
        ),
        None,
    )
    return None if is_workflow_package_draft(current) else definition


def _visible_workflow_ids() -> Select[tuple[str]]:
    current = aliased(WorkflowRevision)
    draft_type = func.json_type(current.dependencies_json, "$." + WORKFLOW_PACKAGE_DRAFT_MARKER)
    return (
        select(WorkflowDefinition.id)
        .outerjoin(
            current,
            and_(
                current.id == WorkflowDefinition.current_revision_id,
                current.workflow_id == WorkflowDefinition.id,
            ),
        )
        .where(or_(draft_type.is_(None), draft_type != "object"))
    )


def list_workflow_revision_choices(session: Session) -> list[WorkflowRevisionChoiceOut]:
    statement = (
        select(
            WorkflowRevision.id.label("revision_id"),
            WorkflowDefinition.id.label("workflow_id"),
            WorkflowDefinition.name.label("workflow_name"),
            WorkflowDefinition.operation,
            WorkflowRevision.version,
        )
        .join(WorkflowDefinition, WorkflowDefinition.id == WorkflowRevision.workflow_id)
        .where(WorkflowDefinition.id.in_(_visible_workflow_ids()))
        .order_by(
            WorkflowDefinition.name,
            WorkflowDefinition.id,
            WorkflowRevision.version,
            WorkflowRevision.id,
        )
        .execution_options(autoflush=False)
    )
    return [WorkflowRevisionChoiceOut(**row) for row in session.execute(statement).mappings()]


def load_workflow_revision_schema(
    session: Session, revision_id: str
) -> WorkflowRevisionSchemaOut | None:
    statement = (
        select(
            WorkflowRevision.id.label("revision_id"),
            WorkflowDefinition.id.label("workflow_id"),
            WorkflowDefinition.operation,
            WorkflowRevision.input_schema_json,
        )
        .join(WorkflowDefinition, WorkflowDefinition.id == WorkflowRevision.workflow_id)
        .where(
            WorkflowRevision.id == revision_id,
            WorkflowDefinition.id.in_(_visible_workflow_ids()),
        )
        .execution_options(autoflush=False)
    )
    row = session.execute(statement).mappings().one_or_none()
    return None if row is None else WorkflowRevisionSchemaOut(**row)
