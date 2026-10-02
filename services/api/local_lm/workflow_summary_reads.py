"""Read lightweight workflow list metadata and one selected definition."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import islice
from typing import Literal

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.orm import Session, aliased, selectinload

from .models import WorkflowDefinition, WorkflowRevision
from .schemas import (
    WorkflowRevisionChoiceOut,
    WorkflowRevisionSchemaOut,
    WorkflowSummaryOut,
)
from .upscale_workflows import (
    UPSCALE_SETTING_KEY,
    effective_upscale_schema,
    workflow_declares_upscale,
)
from .workflow_package_drafts import (
    WORKFLOW_PACKAGE_DRAFT_MARKER,
    is_workflow_package_draft,
)


def list_workflow_summaries(
    session: Session,
    *,
    limit: int | None = None,
    offset: int = 0,
    search: str = "",
    operation: str | None = None,
    workflow_ids: Sequence[str] = (),
    ungrouped_only: bool = False,
) -> list[WorkflowSummaryOut]:
    counts = (
        select(
            WorkflowRevision.workflow_id,
            func.count(WorkflowRevision.id).label("revision_count"),
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
    if operation is not None:
        statement = statement.where(WorkflowDefinition.operation == operation)
    if workflow_ids:
        statement = statement.where(WorkflowDefinition.id.in_(workflow_ids))
    if ungrouped_only:
        statement = statement.where(WorkflowDefinition.family_id.is_(None))
    query = search.strip().casefold()
    if not query:
        statement = statement.offset(offset)
        if limit is not None:
            statement = statement.limit(limit)
    rows = session.execute(statement.execution_options(yield_per=200)).mappings()
    matches = (
        row
        for row in rows
        if not query
        or query in str(row["name"]).casefold()
        or query in str(row["description"] or "").casefold()
    )
    start = offset if query else 0
    page = islice(matches, start, None)
    return [WorkflowSummaryOut(**row) for row in islice(page, limit)]


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


def list_workflow_revision_choices(
    session: Session,
    *,
    limit: int | None = None,
    offset: int = 0,
    search: str = "",
    operation: str | None = None,
    workflow_ids: Sequence[str] = (),
    revision_ids: Sequence[str] = (),
    role: Literal["chat", "image", "video"] | None = None,
) -> list[WorkflowRevisionChoiceOut]:
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
    if operation is not None:
        statement = statement.where(WorkflowDefinition.operation == operation)
    if workflow_ids:
        statement = statement.where(WorkflowDefinition.id.in_(workflow_ids))
    if revision_ids:
        statement = statement.where(WorkflowRevision.id.in_(revision_ids))
    if role == "chat":
        statement = statement.where(WorkflowDefinition.operation == "text")
    elif role == "video":
        statement = statement.where(WorkflowDefinition.operation.contains("video"))
    elif role == "image":
        statement = statement.where(
            WorkflowDefinition.operation.contains("image"),
            ~WorkflowDefinition.operation.contains("video"),
        )
    query = search.strip().casefold()
    if not query:
        statement = statement.offset(offset)
        if limit is not None:
            statement = statement.limit(limit)
    rows = session.execute(statement.execution_options(yield_per=200)).mappings()
    matches = (row for row in rows if not query or query in str(row["workflow_name"]).casefold())
    start = offset if query else 0
    page = islice(matches, start, None)
    return [WorkflowRevisionChoiceOut(**row) for row in islice(page, limit)]


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
    if row is None:
        return None
    schema = row["input_schema_json"]
    if (
        workflow_declares_upscale(schema)
        and schema["properties"][UPSCALE_SETTING_KEY].get("readOnly") is not True
    ):
        # Only enlargement declarations need the selected graph to distinguish
        # an adjustable factor from a fixed output size.
        graph = session.scalar(
            select(WorkflowRevision.api_graph_json)
            .where(WorkflowRevision.id == revision_id)
            .execution_options(autoflush=False)
        )
        if graph is None:
            return None
        schema = effective_upscale_schema(graph, schema)
    return WorkflowRevisionSchemaOut(**{**row, "input_schema_json": schema})
