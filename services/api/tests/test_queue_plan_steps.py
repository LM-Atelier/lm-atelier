from __future__ import annotations

import json

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event
from sqlalchemy.orm import Session
from test_user_queue_activity import client as client
from test_user_queue_activity import plan
from test_user_queue_activity import session as session

from local_lm.models import Job, WorkStep, WorkStepDependency


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


@pytest.mark.parametrize(
    "reported, expected",
    [
        (0.42, 0.42),
        (0.0, 0.0),
        (1.0, 1.0),
        (-0.1, None),
        (1.1, None),
        (True, None),
        ("0.42", None),
        (None, None),
    ],
)
async def test_step_progress_is_a_bounded_report_from_its_running_job(
    session: Session, client: AsyncClient, reported: object, expected: float | None
) -> None:
    plan(session, "selected")
    session.add(
        WorkStep(id="active", plan_id="selected", ordinal=1, operation="image", status="running")
    )
    session.flush()
    session.add(
        Job(
            id="active-job",
            kind="image",
            status="running",
            work_plan_id="selected",
            work_step_id="active",
            progress=0.9,
            progress_json={
                "version": 2,
                "indeterminate": False,
                "stage_progress": reported,
                "overall_progress": None,
            },
            phase="neutral-private-phase",
            payload_json={"prompt": "neutral-private-payload"},
        )
    )
    session.commit()
    session.expunge_all()
    loaded: list[object] = []
    statements: list[str] = []

    def track(_session: Session, instance: object) -> None:
        loaded.append(instance)

    def query(
        _connection: object,
        _cursor: object,
        sql: str,
        _parameters: object,
        _context: object,
        _many: bool,
    ) -> None:
        if sql.lstrip().upper().startswith("SELECT"):
            statements.append(sql)

    engine = session.get_bind()
    event.listen(session, "loaded_as_persistent", track)
    event.listen(engine, "before_cursor_execute", query)
    try:
        response = await client.get("/api/queue/plans/selected/steps")
    finally:
        event.remove(session, "loaded_as_persistent", track)
        event.remove(engine, "before_cursor_execute", query)
    assert response.status_code == 200
    value = response.json()
    assert value["items"][0]["progress"] == expected
    assert value["items"][0]["progress_scope"] == ("stage" if expected is not None else None)
    assert len(statements) == 3 and loaded == []
    assert "neutral-private" not in response.text


@pytest.mark.parametrize(
    "case", ["missing", "hidden", "other-plan", "duplicate", "queued-job", "completed-step"]
)
async def test_step_progress_does_not_borrow_unrelated_or_ambiguous_job_evidence(
    session: Session, client: AsyncClient, case: str
) -> None:
    plan(session, "selected")
    plan(session, "other")
    session.add(
        WorkStep(
            id="active",
            plan_id="selected",
            ordinal=1,
            operation="image",
            status="complete" if case == "completed-step" else "running",
        )
    )
    session.flush()
    if case != "missing":
        session.add(
            Job(
                id="candidate",
                kind="edit_verify" if case == "hidden" else "image",
                status="queued" if case == "queued-job" else "running",
                work_plan_id="other" if case == "other-plan" else "selected",
                work_step_id="active",
                progress=0.9,
                progress_json={"version": 2, "indeterminate": False, "stage_progress": 0.42},
            )
        )
    if case == "duplicate":
        session.add(
            Job(
                id="second",
                kind="image",
                status="running",
                work_plan_id="selected",
                work_step_id="active",
                progress=0.8,
                progress_json={},
            )
        )
    session.commit()
    response = await client.get("/api/queue/plans/selected/steps")
    assert response.status_code == 200
    assert response.json()["items"][0]["progress"] is None


@pytest.mark.parametrize(
    "snapshot, expected, scope",
    [
        (
            {"version": 2, "indeterminate": False, "stage_progress": 0.8, "overall_progress": 0.3},
            0.3,
            "overall",
        ),
        ({"version": 2, "indeterminate": True, "stage_progress": 0.8}, None, None),
        ({"version": 1, "indeterminate": False, "stage_progress": 0.8}, None, None),
        ({"version": 2, "stage_progress": 0.8}, None, None),
        ({"version": 2, "indeterminate": 0, "stage_progress": 0.8}, None, None),
        (
            {"version": 2, "indeterminate": False, "stage_progress": 0.8, "overall_progress": 1.5},
            None,
            None,
        ),
        ({}, None, None),
    ],
)
async def test_step_progress_uses_only_current_snapshot_metadata(
    session: Session,
    client: AsyncClient,
    snapshot: dict[str, object],
    expected: float | None,
    scope: str | None,
) -> None:
    plan(session, "selected")
    session.add(
        WorkStep(id="active", plan_id="selected", ordinal=1, operation="image", status="running")
    )
    session.flush()
    session.add(
        Job(
            id="active-job",
            kind="image",
            status="running",
            work_plan_id="selected",
            work_step_id="active",
            progress=0.9,
            progress_json=snapshot,
        )
    )
    session.commit()
    response = await client.get("/api/queue/plans/selected/steps")
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert (item["progress"], item["progress_scope"]) == (expected, scope)


@pytest.mark.parametrize("indeterminate", [False, True])
async def test_step_progress_follows_real_writer_across_stage_changes(
    session: Session,
    client: AsyncClient,
    indeterminate: bool,
) -> None:
    from local_lm.progress import update_job_progress

    plan(session, "selected")
    session.add(
        WorkStep(id="active", plan_id="selected", ordinal=1, operation="image", status="running")
    )
    session.flush()
    job = Job(
        id="active-job",
        kind="image",
        status="running",
        work_plan_id="selected",
        work_step_id="active",
        progress=0.0,
    )
    session.add(job)
    update_job_progress(job, stage="first", stage_progress=0.8)
    update_job_progress(job, stage="second", stage_progress=0.1, indeterminate=indeterminate)
    session.commit()
    assert job.progress == 0.8
    response = await client.get("/api/queue/plans/selected/steps")
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["progress"] == (None if indeterminate else 0.1)
    assert item["progress_scope"] == (None if indeterminate else "stage")
