"""Record what a workflow revision executes wherever that was never written.

Every way of creating a revision now stores its artifact identity, but for a
while one did not: a workflow brought in with a project archive arrived without
it. Such a revision can be reviewed, yet cannot be activated, cannot follow a
byte-identical recompile, and takes no part in capability evidence, all of which
look a revision up by that identity.

The identity is a pure function of what the revision stores, so it can be
written afterwards without asking anybody anything. Only a missing value is
written. An existing one is never recomputed, because a stored identity may
already be what other records point at, and rewriting it would detach them.

Nothing about approval changes: a review is judged against the revision's
execution-bearing content, never against this column.

A row whose stored content cannot be read as a workflow is left without one. This
runs at startup, and a workspace that refused to open over one damaged row would
be a far worse outcome than that row staying as unusable as it already was.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .model_planner import workflow_artifact_contract
from .models import WorkflowDefinition, WorkflowRevision


def reconcile_missing_artifact_identity(session: Session) -> int:
    """Write the artifact identity of every readable revision that lacks one, and say how many."""

    rows = session.execute(
        select(WorkflowRevision, WorkflowDefinition.operation)
        .join(WorkflowDefinition, WorkflowDefinition.id == WorkflowRevision.workflow_id)
        .where(WorkflowRevision.artifact_sha256.is_(None))
        .order_by(WorkflowRevision.id)
    ).all()
    written = 0
    for revision, operation in rows:
        fields = (revision.api_graph_json, revision.input_schema_json, revision.dependencies_json)
        if not all(isinstance(value, dict) for value in fields):
            continue
        try:
            identity = workflow_artifact_contract(
                operation=operation,
                engine=revision.engine,
                api_graph=revision.api_graph_json,
                input_schema=revision.input_schema_json,
                dependencies=revision.dependencies_json,
            )
        except (TypeError, ValueError):
            # Stored JSON the canonical form refuses, such as a non-finite number.
            continue
        revision.artifact_sha256 = identity
        written += 1
    session.flush()
    return written
