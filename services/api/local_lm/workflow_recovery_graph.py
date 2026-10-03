"""Compare workflow definitions and their consumers under the recovery writer."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session

from .chat_recovery_graph import ChatRecoveryGraphError, _GraphReader
from .db import Base
from .models import RecoveryItem
from .recovery_previews import RecoveryPreviewConflict, RecoverySnapshot, _utc
from .recovery_v1 import RecoveryAction, RecoveryConflict, RecoveryCountsV1, RecoveryKind
from .workflow_recovery_history import HISTORICAL_WORKFLOW_CONSUMERS
from .workflow_recovery_schema import (
    JSON_WORKFLOW_COLUMNS,
    WORKFLOW_IDENTITY_KEYS,
    workflow_consumer_sql,
)

_TERMINAL = frozenset({"complete", "failed", "cancelled"})
_INTERNAL = frozenset(
    {
        "workflow_definitions",
        "workflow_revisions",
        "workflow_preferences",
        "workflow_dependency_slots",
        "workflow_activations",
        "workflow_dependency_bindings",
        "workflow_revision_reviews",
        "workflow_profile_compatibility",
        "workflow_trust_attestations",
    }
)


def _references(value: object, identities: frozenset[str]) -> bool:
    """Compare structural identity values without interpreting any stored text."""
    budget = 100_000

    def visit(current: object, depth: int) -> bool:
        nonlocal budget
        budget -= 1
        if budget < 0 or depth > 16:
            raise ChatRecoveryGraphError()
        if isinstance(current, Mapping):
            if len(current) > 4_096:
                raise ChatRecoveryGraphError()
            found = False
            for key, child in current.items():
                if not isinstance(key, str):
                    raise ChatRecoveryGraphError()
                if key in WORKFLOW_IDENTITY_KEYS and isinstance(child, str) and child in identities:
                    found = True
                if isinstance(child, (Mapping, list)):
                    found = visit(child, depth + 1) or found
            return found
        if isinstance(current, list):
            if len(current) > 4_096:
                raise ChatRecoveryGraphError()
            return any([visit(child, depth + 1) for child in current])
        return False

    if value is None:
        return False
    if not isinstance(value, (Mapping, list)):
        raise ChatRecoveryGraphError()
    return visit(value, 0)


def inspect_workflow_recovery(session: Session, family_id: str, now: datetime) -> RecoverySnapshot:
    """Bind the preview to canonical identity, authority and bounded consumer rows."""
    try:
        return _snapshot(session, family_id, now)
    except ChatRecoveryGraphError:
        raise RecoveryPreviewConflict("workflow-recovery-graph-invalid") from None


def _snapshot(session: Session, family_id: str, now: datetime) -> RecoverySnapshot:
    reader = _GraphReader(session)
    families = Base.metadata.tables["workflow_families"]
    if not reader.read("workflow_families", families.c.id == family_id):
        raise RecoveryPreviewConflict("recovery-subject-not-found")
    definitions = Base.metadata.tables["workflow_definitions"]
    definition_rows = reader.read("workflow_definitions", definitions.c.family_id == family_id)
    definition_ids = frozenset(row["id"] for row in definition_rows)
    revisions = Base.metadata.tables["workflow_revisions"]
    revision_rows = reader.read("workflow_revisions", revisions.c.workflow_id.in_(definition_ids))
    revision_ids = frozenset(row["id"] for row in revision_rows)
    identities = frozenset({family_id, *definition_ids, *revision_ids})
    targets = {
        "workflow_families": frozenset({family_id}),
        "workflow_definitions": definition_ids,
        "workflow_revisions": revision_ids,
    }
    external: dict[str, tuple[Mapping[str, Any], ...]] = {}
    internal: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for name, table in sorted(Base.metadata.tables.items()):
        if name in {"workflow_families", "workflow_definitions", "workflow_revisions"}:
            continue
        predicates = [
            table.c[key.parent.name].in_(targets[key.column.table.name])
            for key in table.foreign_keys
            if key.column.table.name in targets
        ]
        if not predicates:
            continue
        rows = reader.read(name, or_(*predicates))
        (internal if name in _INTERNAL else external)[name] = rows
    arms = Base.metadata.tables["generation_experiment_arms"]
    external["generation_experiment_arms"] = reader.read(
        "generation_experiment_arms", arms.c.workflow_revision_id.in_(revision_ids)
    )
    identities = identities.union(
        row["id"]
        for rows in (
            internal.get("workflow_activations", ()),
            external.get("workflow_install_offers", ()),
        )
        for row in rows
    )
    recoveries = Base.metadata.tables["recovery_items"]
    reader.read(
        "recovery_items",
        (recoveries.c.kind == RecoveryKind.WORKFLOW_FAMILY.value)
        & (recoveries.c.subject_id == family_id),
    )
    item = session.scalar(
        select(RecoveryItem).where(
            RecoveryItem.kind == RecoveryKind.WORKFLOW_FAMILY.value,
            RecoveryItem.subject_id == family_id,
        )
    )
    matched: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for name, json_columns in JSON_WORKFLOW_COLUMNS.items():
        table = Base.metadata.tables[name]
        columns = tuple(
            column
            for column in (
                "id",
                "run_id",
                "status",
                "workflow_revision_id",
                "enabled",
                "is_default",
                *json_columns,
            )
            if column in table.c
        )
        rows = reader.read(
            name,
            text(workflow_consumer_sql(name, name, ":family_id")),
            {"family_id": family_id},
            columns=columns,
        )
        matched[name] = tuple(
            row
            for row in rows
            if (
                row.get("workflow_revision_id") in revision_ids
                or any(_references(row[column], identities) for column in json_columns)
            )
        )
    runs_table = Base.metadata.tables["runs"]
    captured_run_ids = {row["run_id"] for row in matched["run_context_snapshots"]}
    captured_runs = reader.read(
        "runs",
        runs_table.c.id.in_(captured_run_ids),
        columns=("id", "status", "workflow_revision_id"),
    )
    runs = {row["id"]: row for row in (*matched["runs"], *captured_runs)}
    if len(captured_runs) != len(captured_run_ids):
        raise ChatRecoveryGraphError()
    jobs_table = Base.metadata.tables["jobs"]
    linked_jobs = reader.read(
        "jobs", jobs_table.c.run_id.in_(runs), columns=("id", "run_id", "status")
    )
    jobs = {row["id"]: row for row in (*matched["jobs"], *linked_jobs)}
    active = sum(row["status"] not in _TERMINAL for row in runs.values())
    active += sum(row["status"] not in _TERMINAL for row in matched["work_steps"])
    active += sum(row["status"] not in _TERMINAL for row in jobs.values())
    active += sum(row["status"] == "queued" for row in external.get("workflow_install_offers", ()))
    active += sum(
        row["state"] in {"queued", "running"} for row in external.get("setup_verifications", ())
    )
    selections = sum(
        len(external.get(name, ()))
        for name in (
            "chat_workflow_selections",
            "project_workflow_selections",
            "projects",
        )
    )
    selections += sum(bool(row["is_default"]) for row in internal.get("workflow_preferences", ()))
    presets = matched["workflow_use_case_presets"]
    preset_ids = {row["id"] for row in presets}
    selections += sum(bool(row["is_default"]) for row in presets)
    for name in ("chat_workflow_use_case_selections", "project_workflow_use_case_selections"):
        table = Base.metadata.tables[name]
        external[name] = reader.read(name, table.c.preset_id.in_(preset_ids))
        selections += len(external[name])
    conflicts: tuple[RecoveryConflict, ...] = tuple(
        conflict
        for count, conflict in (
            (active, RecoveryConflict.ACTIVE_WORK),
            (selections, RecoveryConflict.ACTIVE_SELECTION),
        )
        if count
    )
    referenced: dict[str, set[tuple[object, ...]]] = {}
    for group in (external, matched, {"runs": tuple(runs.values()), "jobs": tuple(jobs.values())}):
        for name, rows in group.items():
            keys = tuple(column.name for column in Base.metadata.tables[name].primary_key.columns)
            referenced.setdefault(name, set()).update(
                tuple(row[key] for key in keys) for row in rows
            )
    references = sum(len(values) for values in referenced.values())
    actions: tuple[RecoveryAction, ...] = ()
    if not conflicts:
        if item is None:
            actions = (RecoveryAction.TRASH,)
        elif item.state == "recoverable":
            actions = (RecoveryAction.RESTORE,) if _utc(now) < _utc(item.purge_after) else ()
            if not any(
                values
                for name, values in referenced.items()
                if name not in HISTORICAL_WORKFLOW_CONSUMERS
            ):
                actions += (RecoveryAction.PURGE,)
    return RecoverySnapshot(
        fingerprint=hashlib.sha256(
            b"workflow-recovery-canonical-v1\0" + reader.digest.digest()
        ).hexdigest(),
        counts=RecoveryCountsV1(
            workflow_families=1,
            workflow_definitions=len(definition_rows),
            workflow_revisions=len(revision_rows),
            runs=len(runs),
            jobs=len(jobs),
            active_work=active,
            references=references,
        ),
        conflicts=conflicts,
        available_actions=actions,
    )
