"""Hide deleted project configuration while preserving canonical filing links."""

from __future__ import annotations

from sqlalchemy import SQLColumnExpression, case, exists, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .models import Project, RecoveryItem
from .recovery_v1 import RecoveryKind


def visible_project(
    project_id: SQLColumnExpression[str] | SQLColumnExpression[str | None],
) -> ColumnElement[bool]:
    """Every recovery membership hides its project from ordinary workspace reads."""
    return ~exists(
        select(RecoveryItem.subject_id).where(
            RecoveryItem.kind == RecoveryKind.PROJECT.value,
            RecoveryItem.subject_id == project_id,
        )
    )


def effective_project_id(project_id: SQLColumnExpression[str | None]) -> ColumnElement[str | None]:
    """Project unfiled children in SQL without changing the stored foreign key."""
    return case((visible_project(project_id), project_id), else_=None)


def live_project(session: Session, project_id: str | None) -> Project | None:
    """Recheck membership even when the session already cached this project."""
    if project_id is None:
        return None
    with session.no_autoflush:
        return session.scalar(
            select(Project).where(Project.id == project_id, visible_project(Project.id))
        )


def live_project_ids(session: Session, identities: set[str]) -> set[str]:
    """Resolve a whole response page without a separate query for every chat."""
    if not identities:
        return set()
    with session.no_autoflush:
        return set(
            session.scalars(
                select(Project.id).where(Project.id.in_(identities), visible_project(Project.id))
            )
        )
