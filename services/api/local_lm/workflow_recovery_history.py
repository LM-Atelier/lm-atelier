"""Remove workflow execution content while retaining immutable historical identity."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from .db import Base
from .models import RecoveryItem, WorkflowDefinition, WorkflowRevision
from .recovery_previews import RecoveryPreviewConflict

HISTORICAL_WORKFLOW_CONSUMERS = frozenset(
    {
        "runs",
        "work_steps",
        "jobs",
        "run_context_snapshots",
        "generation_experiment_arms",
        "setup_verifications",
    }
)


def retain_workflow_history(session: Session, family_id: str, now: datetime) -> None:
    """Minimize canonical rows only inside an authorized permanent-deletion transaction."""
    item = session.scalar(
        select(RecoveryItem).where(
            RecoveryItem.kind == "workflow_family", RecoveryItem.subject_id == family_id
        )
    )
    if item is None or item.state != "purging":
        raise RecoveryPreviewConflict("recovery-item-not-purging")
    definitions = select(WorkflowDefinition.id).where(WorkflowDefinition.family_id == family_id)
    revisions = select(WorkflowRevision.id).where(WorkflowRevision.workflow_id.in_(definitions))
    for name in (
        "workflow_dependency_bindings",
        "workflow_dependency_slots",
        "workflow_revision_reviews",
        "workflow_trust_attestations",
    ):
        table = Base.metadata.tables[name]
        session.execute(delete(table).where(table.c.workflow_revision_id.in_(revisions)))
    for name in ("workflow_preferences", "workflow_profile_compatibility"):
        table = Base.metadata.tables[name]
        session.execute(delete(table).where(table.c.workflow_family_id == family_id))
    activations = Base.metadata.tables["workflow_activations"]
    session.execute(
        update(activations)
        .where(activations.c.workflow_revision_id.in_(revisions))
        .values(
            is_active=False,
            state="disabled",
            last_validated_at=None,
            invalidated_at=now,
            invalidation_code="workflow-deleted",
            invalidation_reason="This workflow was permanently deleted.",
            details_json={},
        )
    )
    session.execute(
        update(WorkflowRevision)
        .where(WorkflowRevision.workflow_id.in_(definitions))
        .values(
            engine_version=None,
            ui_graph_json={},
            api_graph_json={},
            input_schema_json={},
            capabilities_json=[],
            dependencies_json={},
            dependency_contract_sha256=None,
            trusted=False,
        )
    )
    session.execute(
        update(WorkflowDefinition)
        .where(WorkflowDefinition.family_id == family_id)
        .values(description="", current_revision_id=None)
    )
    families = Base.metadata.tables["workflow_families"]
    session.execute(
        update(families)
        .where(families.c.id == family_id)
        .values(description="", use_case="", tags_json=[], enabled=False, archived=True)
    )
