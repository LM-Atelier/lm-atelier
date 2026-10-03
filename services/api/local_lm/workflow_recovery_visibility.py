"""Hide deleted workflow families without hiding their historical revision identities."""

from sqlalchemy import SQLColumnExpression, exists, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .models import RecoveryItem, WorkflowFamily
from .recovery_v1 import RecoveryKind


def visible_workflow_family(
    family_id: SQLColumnExpression[str] | SQLColumnExpression[str | None],
) -> ColumnElement[bool]:
    return ~exists(
        select(RecoveryItem.subject_id).where(
            RecoveryItem.kind == RecoveryKind.WORKFLOW_FAMILY.value,
            RecoveryItem.subject_id == family_id,
        )
    )


def workflow_family_deleted(session: Session, family_id: str | None) -> bool:
    """Treat recoverable and permanently deleted family memberships as unavailable."""
    if family_id is None:
        return False
    return (
        session.scalar(
            select(RecoveryItem.deletion_id)
            .where(
                RecoveryItem.kind == RecoveryKind.WORKFLOW_FAMILY.value,
                RecoveryItem.subject_id == family_id,
            )
            .limit(1)
        )
        is not None
    )


def workflow_family_ready(session: Session, family_id: str | None) -> bool:
    """Read durable enablement rather than a family's cached ORM state."""
    if family_id is None:
        return True
    return (
        session.scalar(
            select(WorkflowFamily.id).where(
                WorkflowFamily.id == family_id,
                WorkflowFamily.enabled.is_(True),
                WorkflowFamily.archived.is_(False),
            )
        )
        is not None
    )
