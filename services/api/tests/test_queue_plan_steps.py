from __future__ import annotations

import json

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event
from sqlalchemy.orm import Session
from test_user_queue_activity import client as client
from test_user_queue_activity import plan
from test_user_queue_activity import session as session

from local_lm.models import WorkStep, WorkStepDependency


async def test_plan_steps_are_bounded_metadata_with_prerequisite_state(
    session: Session,
    client: AsyncClient,
) -> None:
    plan(session, "selected")
    plan(session, "other")
    for index, status in enumerate(["complete", "running", "queued", "cancelled"]):
        session.add(
            WorkStep(
                id=f"step-{index}",
                plan_id="selected",
                ordinal=index,
                operation="image",
                status=status,
                prompt="neutral-private-prompt",
                settings_json={"token": "neutral-private-setting"},
                input_bindings_json=[{"path": "neutral-private-input"}],
                output_contract_json=[{"path": "neutral-private-output"}],
                error="neutral-private-error",
            )
        )
    session.add(
        WorkStep(id="foreign", plan_id="other", ordinal=0, operation="video", status="queued")
    )
    session.flush()
    session.add_all(
        [
            WorkStepDependency(step_id="step-2", depends_on_step_id="step-0"),
            WorkStepDependency(step_id="step-2", depends_on_step_id="step-1"),
            WorkStepDependency(step_id="step-3", depends_on_step_id="step-1"),
        ]
    )
    session.commit()
    session.expunge_all()
    existing = await client.get("/api/queue/activity")
    assert existing.status_code == 200
    loaded: list[object] = []

    def track(_session: Session, instance: object) -> None:
        loaded.append(instance)

    event.listen(session, "loaded_as_persistent", track)
    try:
        response = await client.get("/api/queue/plans/selected/steps", params={"limit": 2})
    finally:
        event.remove(session, "loaded_as_persistent", track)
    assert response.status_code == 200
    value = response.json()
    assert value["plan_id"] == "selected" and value["total"] == 4 and value["next_offset"] == 2
    assert [row["id"] for row in value["items"]] == ["step-0", "step-1"]
    assert [row["status"] for row in value["items"]] == ["complete", "running"]
    assert all(row["label"] == "Image generation" for row in value["items"])
    assert loaded == [] and "neutral-private" not in json.dumps(value)
    page = await client.get("/api/queue/plans/selected/steps", params={"limit": 2, "offset": 2})
    assert page.status_code == 200
    value = page.json()
    assert value["next_offset"] is None
    assert [row["id"] for row in value["items"]] == ["step-2", "step-3"]
    assert value["items"][0]["status"] == "blocked" and value["items"][0]["blocked_by"] == 1
    assert value["items"][1]["status"] == "cancelled" and value["items"][1]["blocked_by"] == 0


async def test_step_pages_do_not_expose_unknown_operations_or_flush_pending_rows(
    session: Session,
    client: AsyncClient,
) -> None:
    plan(session, "selected")
    session.add(
        WorkStep(
            id="unknown",
            plan_id="selected",
            ordinal=0,
            operation="neutral-private-operation",
            status="queued",
        )
    )
    session.commit()
    session.add(WorkStep(id="pending", plan_id="selected", ordinal=1, operation="image"))
    response = await client.get("/api/queue/plans/selected/steps")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["label"] == "Work step"
    assert "neutral-private-operation" not in response.text and len(session.new) == 1


async def test_empty_and_missing_plans_are_distinct(session: Session, client: AsyncClient) -> None:
    plan(session, "empty")
    session.commit()
    empty = await client.get("/api/queue/plans/empty/steps")
    assert empty.status_code == 200
    assert empty.json()["items"] == [] and empty.json()["total"] == 0
    missing = await client.get("/api/queue/plans/missing/steps")
    assert missing.status_code == 404
    assert missing.json()["code"] == "queue-plan-not-found"


async def test_unknown_stored_step_status_fails_with_a_fixed_error(
    session: Session,
    client: AsyncClient,
) -> None:
    plan(session, "selected")
    session.add(
        WorkStep(
            id="invalid",
            plan_id="selected",
            ordinal=0,
            operation="image",
            status="neutral-private-status",
        )
    )
    session.commit()
    response = await client.get("/api/queue/plans/selected/steps")
    assert response.status_code == 409
    assert response.json()["code"] == "queue-step-state-invalid"
    assert "neutral-private-status" not in response.text


@pytest.mark.parametrize(
    "params",
    [
        {"limit": "0"},
        {"limit": "101"},
        {"offset": "-1"},
        {"offset": str(2**63)},
        {"limit": "invalid"},
    ],
)
async def test_step_paging_inputs_are_bounded(client: AsyncClient, params: dict[str, str]) -> None:
    response = await client.get("/api/queue/plans/example/steps", params=params)
    assert response.status_code == 422
