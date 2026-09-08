from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import Chat, WorkPlan, WorkStep
from local_lm.project_work_plans import import_work_plans, validate_work_plans
from local_lm.schemas import WorkPlanOut, WorkStepOut

_STEP_STATUSES = (
    "queued",
    "running",
    "paused",
    "complete",
    "failed",
    "cancelled",
    "interrupted",
    "blocked",
)
_PLAN_STATUSES = (*_STEP_STATUSES, "partial")


def _step(status: str = "queued") -> dict[str, Any]:
    return {
        "id": "step-one",
        "plan_id": "plan-one",
        "run_id": None,
        "ordinal": 1,
        "display_group": None,
        "operation": "text",
        "status": status,
        "prompt": "Describe a paper boat",
        "profile_id": None,
        "workflow_revision_id": None,
        "settings_json": {},
        "input_bindings_json": [],
        "output_contract_json": [],
        "queue_class": "interactive_compute",
        "error": None,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


def _plan(status: str = "queued", step_status: str = "queued") -> dict[str, Any]:
    return {
        "id": "plan-one",
        "chat_id": "chat-one",
        "idempotency_key": None,
        "source_action": "turn",
        "persistence_scope": "durable",
        "status": status,
        "context_head_message_id": None,
        "transcript_sequence": 1,
        "priority": 0,
        "planner_version": "legacy-turn-v1",
        "failure_policy": "stop_dependents",
        "summary_json": {},
        "steps": [_step(step_status)],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


@pytest.mark.parametrize("status", _STEP_STATUSES)
def test_work_step_response_accepts_each_emitted_status(status: str) -> None:
    assert WorkStepOut.model_validate(_step(status)).model_dump(mode="json")["status"] == status


@pytest.mark.parametrize("status", _PLAN_STATUSES)
def test_work_plan_response_accepts_each_emitted_status(status: str) -> None:
    assert WorkPlanOut.model_validate(_plan(status)).model_dump(mode="json")["status"] == status


@pytest.mark.parametrize("status", ["future-status", "", "COMPLETE"])
def test_work_step_response_refuses_an_undeclared_status(status: str) -> None:
    with pytest.raises(ValidationError):
        WorkStepOut.model_validate(_step(status))


@pytest.mark.parametrize("status", ["future-status", "", "COMPLETE"])
def test_work_plan_response_refuses_an_undeclared_status(status: str) -> None:
    with pytest.raises(ValidationError):
        WorkPlanOut.model_validate(_plan(status))


def test_partial_is_a_plan_outcome_and_not_a_step_outcome() -> None:
    with pytest.raises(ValidationError):
        WorkPlanOut.model_validate(_plan("partial", "partial"))


@pytest.mark.parametrize(
    "status",
    [
        "future-status",
        "",
        "queued",
        "blocked",
        "partial",
        "complete",
        "failed",
        "cancelled",
        "interrupted",
    ],
)
def test_archive_status_acceptance_and_normalization_are_preserved(
    settings: Settings, status: str
) -> None:
    manifest = {
        "version": 7,
        "work_plans": [_plan(status, status)],
        "work_step_dependencies": [],
        "chats": [{"id": "chat-one", "messages": []}],
        "runs": [],
    }
    records = validate_work_plans(manifest)
    assert records[0].status == records[0].steps[0].status == status
    expected = status if status in {"complete", "failed", "cancelled", "interrupted"} else "failed"
    with SessionLocal() as session:
        chat = Chat(title="Imported paper boats")
        session.add(chat)
        session.flush()
        plans, steps = import_work_plans(session, records, [], {"chat-one": chat}, {}, {}, None)
        session.flush()
        plan = session.get(WorkPlan, plans["plan-one"])
        step = session.get(WorkStep, steps["step-one"])
        assert plan is not None and step is not None
        assert plan.status == step.status == expected
        assert step.error == (
            None if expected == status else "Imported while generation was incomplete."
        )
        assert WorkPlanOut.model_validate(plan).model_dump(mode="json")["status"] == expected
        assert WorkStepOut.model_validate(step).model_dump(mode="json")["status"] == expected
