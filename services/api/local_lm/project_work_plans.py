"""Portable historical work-plan graphs; imported plans carry no queued jobs."""

from __future__ import annotations

import copy
from collections import defaultdict, deque
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .domain import new_id
from .models import Chat, Message, Run, WorkPlan, WorkStep, WorkStepDependency
from .project_dependencies import ImportedDependencies
from .schemas import WorkPlanOut

_TERMINAL = {"complete", "failed", "cancelled", "interrupted"}
_PLAN_KEYS = {"plan_id", "work_plan_id"}
_STEP_KEYS = {"step_id", "work_step_id", "source_step_id", "depends_on_step_id"}
_RUN_KEYS = {"run_id", "source_run_id"}
_MESSAGE_KEYS = {
    "message_id",
    "user_message_id",
    "assistant_message_id",
    "source_message_id",
    "branch_head_message_id",
    "context_head_message_id",
}


def export_work_plans(
    session: Session, chat_ids: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    plans = list(
        session.scalars(
            select(WorkPlan)
            .options(selectinload(WorkPlan.steps))
            .where(WorkPlan.chat_id.in_(chat_ids))
            .order_by(WorkPlan.transcript_sequence, WorkPlan.id)
        )
    )
    step_ids = [step.id for plan in plans for step in plan.steps]
    edges = list(
        session.scalars(select(WorkStepDependency).where(WorkStepDependency.step_id.in_(step_ids)))
    )
    return (
        [WorkPlanOut.model_validate(plan).model_dump(mode="json") for plan in plans],
        [
            {"step_id": edge.step_id, "depends_on_step_id": edge.depends_on_step_id}
            for edge in edges
        ],
    )


def validate_work_plans(manifest: dict[str, Any]) -> list[WorkPlanOut]:
    if manifest["version"] < 7:
        return []
    records = manifest.get("work_plans")
    edges = manifest.get("work_step_dependencies")
    if not isinstance(records, list) or len(records) > 100_000:
        raise ValueError("project manifest has invalid work plans")
    if not isinstance(edges, list) or len(edges) > 100_000:
        raise ValueError("project manifest has invalid work dependencies")
    plans = [WorkPlanOut.model_validate(record) for record in records]
    chats = {chat["id"] for chat in manifest["chats"]}
    messages = {
        message["id"]: chat["id"] for chat in manifest["chats"] for message in chat["messages"]
    }
    runs = {run["id"]: run for run in manifest["runs"]}
    seen_plans: set[str] = set()
    seen_steps: dict[str, str] = {}
    seen_runs: set[str] = set()
    sequences: set[tuple[str, int]] = set()
    for plan in plans:
        if (
            not 1 <= len(plan.id) <= 40
            or plan.id in seen_plans
            or plan.chat_id not in chats
            or plan.transcript_sequence < 1
            or (plan.chat_id, plan.transcript_sequence) in sequences
            or len(plan.source_action) > 32
            or len(plan.planner_version) > 32
            or (
                len(plan.failure_policy) > 32
                and plan.failure_policy != "preserve_completed_block_dependents"
            )
            or len(plan.status) > 16
            or not plan.steps
            or len(plan.steps) > 10_000
            or (
                plan.context_head_message_id is not None
                and messages.get(plan.context_head_message_id) != plan.chat_id
            )
        ):
            raise ValueError("project manifest has an invalid work plan graph")
        seen_plans.add(plan.id)
        sequences.add((plan.chat_id, plan.transcript_sequence))
        ordinals: set[int] = set()
        for step in plan.steps:
            if (
                not 1 <= len(step.id) <= 40
                or step.id in seen_steps
                or step.plan_id != plan.id
                or step.ordinal < 1
                or step.ordinal in ordinals
            ):
                raise ValueError("project manifest has an invalid work step graph")
            seen_steps[step.id] = plan.chat_id
            ordinals.add(step.ordinal)
            if step.run_id is not None:
                run = runs.get(step.run_id)
                if (
                    run is None
                    or step.run_id in seen_runs
                    or run["chat_id"] != plan.chat_id
                    or run.get("work_plan_id") != plan.id
                    or run.get("work_step_id") != step.id
                    or run["operation"] != step.operation
                    or run.get("profile_id") != step.profile_id
                    or run.get("workflow_revision_id") != step.workflow_revision_id
                ):
                    raise ValueError("project manifest has an invalid work step run")
                seen_runs.add(step.run_id)
        source = plan.summary_json.get("edit_source")
        if source is not None:
            if not isinstance(source, dict):
                raise ValueError("project manifest has an invalid edit source")
            source_message = source.get("source_message_id")
            source_run_id = source.get("source_run_id")
            if not all(
                isinstance(value, str) and 1 <= len(value) <= 40
                for value in (source_message, source_run_id)
            ):
                raise ValueError("project manifest has an invalid edit source")
            if source_message in messages and messages[source_message] != plan.chat_id:
                raise ValueError("project edit source belongs to a different chat")
            source_run = runs.get(source_run_id)
            if source_run is not None and (
                source_run["chat_id"] != plan.chat_id
                or source_run["user_message_id"] != source_message
            ):
                raise ValueError("project edit source references a different run")
            last = max(plan.steps, key=lambda step: step.ordinal)
            final_run = runs.get(last.run_id or "")
            if (
                final_run is None
                or plan.summary_json.get("branch_head_message_id")
                != final_run["assistant_message_id"]
            ):
                raise ValueError("project edit has an invalid branch head")
    for run in runs.values():
        if run.get("work_plan_id") is not None and run["id"] not in seen_runs:
            raise ValueError("project run references a missing work step")
    seen_edges: set[tuple[str, str]] = set()
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        if not isinstance(edge, dict) or set(edge) != {"step_id", "depends_on_step_id"}:
            raise ValueError("project manifest has an invalid work dependency")
        target, source = edge["step_id"], edge["depends_on_step_id"]
        if (
            not isinstance(target, str)
            or not isinstance(source, str)
            or target == source
            or target not in seen_steps
            or source not in seen_steps
            or seen_steps[target] != seen_steps[source]
            or (target, source) in seen_edges
        ):
            raise ValueError("project manifest has an invalid work dependency")
        seen_edges.add((target, source))
        adjacency.setdefault(target, set()).add(source)
    # Reject cycles without recursive traversal of a potentially deep imported graph.
    counts = {step: len(adjacency.get(step, set())) for step in seen_steps}
    consumers: dict[str, list[str]] = defaultdict(list)
    for target, sources in adjacency.items():
        for source in sources:
            consumers[source].append(target)
    ready = deque(step for step, count in counts.items() if count == 0)
    visited = 0
    while ready:
        visited += 1
        for target in consumers[ready.popleft()]:
            counts[target] -= 1
            if counts[target] == 0:
                ready.append(target)
    if visited != len(seen_steps):
        raise ValueError("project work dependencies contain a cycle")
    return plans


def remap_work_references(
    value: Any,
    *,
    plans: dict[str, str],
    steps: dict[str, str],
    runs: dict[str, str],
    messages: dict[str, str],
    missing: dict[str, str],
    key: str = "",
) -> Any:
    mappings = (
        plans
        if key in _PLAN_KEYS
        else steps
        if key in _STEP_KEYS
        else runs
        if key in _RUN_KEYS
        else messages
        if key in _MESSAGE_KEYS
        else None
    )
    if isinstance(value, str) and mappings is not None:
        if value in mappings:
            return mappings[value]
        # An absent original remains unavailable, never a reference into another imported project.
        if value not in missing:
            missing[value] = new_id("missing")
        return missing[value]
    if isinstance(value, dict):
        return {
            field: remap_work_references(
                child,
                plans=plans,
                steps=steps,
                runs=runs,
                messages=messages,
                missing=missing,
                key=field,
            )
            for field, child in value.items()
            if field not in {"job_id", "job_ids"}
        }
    if isinstance(value, list):
        singular = key[:-1] if key.endswith("_ids") else key
        return [
            remap_work_references(
                child,
                plans=plans,
                steps=steps,
                runs=runs,
                messages=messages,
                missing=missing,
                key=singular,
            )
            for child in value
        ]
    return copy.deepcopy(value)


def import_work_plans(
    session: Session,
    records: list[WorkPlanOut],
    edges: list[dict[str, str]],
    chats: dict[str, Chat],
    messages: dict[str, Message],
    runs: dict[str, Run],
    dependencies: ImportedDependencies | None,
) -> tuple[dict[str, str], dict[str, str]]:
    plan_map: dict[str, str] = {}
    step_map: dict[str, str] = {}
    stored: dict[str, WorkPlan] = {}
    for record in records:
        plan = WorkPlan(
            chat_id=chats[record.chat_id].id,
            idempotency_key=None,
            source_action=record.source_action,
            persistence_scope="durable",
            status=record.status if record.status in _TERMINAL else "failed",
            context_head_message_id=messages[record.context_head_message_id].id
            if record.context_head_message_id
            else None,
            transcript_sequence=record.transcript_sequence,
            priority=record.priority,
            planner_version=record.planner_version,
            failure_policy=record.failure_policy,
            summary_json={},
            created_at=record.created_at,
        )
        session.add(plan)
        session.flush()
        plan_map[record.id] = plan.id
        stored[record.id] = plan
        for item in record.steps:
            run = runs.get(item.run_id or "")
            role = (
                "chat"
                if item.operation == "text"
                else "video"
                if "video" in item.operation
                else "image"
            )
            step = WorkStep(
                plan_id=plan.id,
                run_id=run.id if run else None,
                ordinal=item.ordinal,
                display_group=item.display_group,
                operation=item.operation,
                status=item.status if item.status in _TERMINAL else "failed",
                prompt=item.prompt,
                profile_id=dependencies.profile(item.profile_id, role) if dependencies else None,
                workflow_revision_id=dependencies.revision(
                    item.workflow_revision_id, {item.operation}
                )
                if dependencies
                else None,
                settings_json=item.settings_json,
                input_bindings_json=[],
                output_contract_json=[],
                queue_class=item.queue_class,
                error=item.error
                if item.status in _TERMINAL
                else "Imported while generation was incomplete.",
                created_at=item.created_at,
            )
            session.add(step)
            session.flush()
            step_map[item.id] = step.id
            if run is not None:
                run.work_plan_id = plan.id
                run.work_step_id = step.id
    missing: dict[str, str] = {}

    run_ids = {key: run.id for key, run in runs.items()}
    message_ids = {key: message.id for key, message in messages.items()}

    def remap(value: Any) -> Any:
        return remap_work_references(
            value,
            plans=plan_map,
            steps=step_map,
            runs=run_ids,
            messages=message_ids,
            missing=missing,
        )

    for record in records:
        stored[record.id].summary_json = remap(record.summary_json)
        for item in record.steps:
            stored_step = session.get(WorkStep, step_map[item.id])
            assert stored_step is not None
            stored_step.input_bindings_json = remap(item.input_bindings_json)
            stored_step.output_contract_json = remap(item.output_contract_json)
    for edge in edges:
        session.add(
            WorkStepDependency(
                step_id=step_map[edge["step_id"]],
                depends_on_step_id=step_map[edge["depends_on_step_id"]],
            )
        )
    for run in runs.values():
        run.provenance_json = remap(run.provenance_json)
    return plan_map, step_map
