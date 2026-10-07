"""A deleted workflow retains accepted comparison history but refuses another choice."""

import copy
from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError
from test_chat_recovery import _command, _impact
from test_generation_experiment_preflight import PREFLIGHT, _request, _two_choices
from test_generation_experiment_records import CREATE, _accept, _records

from local_lm import generation_experiment_api
from local_lm.db import SessionLocal
from local_lm.generation_experiment_preflight import (
    ExperimentResolution,
    resolve_generation_experiment,
)
from local_lm.models import GenerationExperimentArm, WorkflowDefinition, WorkflowRevision


def _family_id(revision_id: str) -> str:
    with SessionLocal() as session:
        family_id: str = session.scalar(
            select(WorkflowDefinition.family_id)
            .join(WorkflowRevision, WorkflowRevision.workflow_id == WorkflowDefinition.id)
            .where(WorkflowRevision.id == revision_id)
        )
        return family_id


async def _trash(client: AsyncClient, revision_id: str) -> dict[str, Any]:
    family_id = _family_id(revision_id)
    preview = await _impact(client, f"/api/workflow-families/{family_id}/deletion-impact")
    assert preview["available_actions"] == ["trash"]
    response = await client.post(
        f"/api/workflow-families/{family_id}/trash",
        json=_command(preview, "trash-compared-workflow"),
    )
    assert response.status_code == 200, response.text
    deleted: dict[str, Any] = response.json()
    return deleted


@pytest.mark.parametrize("operation", ["insert", "change", "missing"])
async def test_new_comparison_choices_require_a_live_workflow_and_keep_existing_history(
    client: AsyncClient, operation: str
) -> None:
    first, second = _two_choices()
    accepted = await _accept(client, _request(first, second))
    revision_id = first["workflow_revision_id"]
    item = await _trash(client, revision_id)
    impact = await _impact(client, f"/api/recovery-items/{item['deletion_id']}/impact")
    assert impact["counts"]["references"] == 1
    assert impact["available_actions"] == ["restore", "purge"]
    with SessionLocal() as session:
        arms = session.scalars(
            select(GenerationExperimentArm)
            .where(GenerationExperimentArm.experiment_id == accepted["id"])
            .order_by(GenerationExperimentArm.ordinal)
        ).all()
        original = [
            {
                column.name: copy.deepcopy(getattr(arm, column.name))
                for column in arm.__table__.columns
            }
            for arm in arms
        ]
        session.execute(
            update(GenerationExperimentArm)
            .where(GenerationExperimentArm.id == arms[0].id)
            .values(workflow_revision_id=revision_id)
        )
        session.commit()
        with pytest.raises(IntegrityError, match="workflow-recovery-write-refused"):
            if operation == "change":
                session.execute(
                    update(GenerationExperimentArm)
                    .where(GenerationExperimentArm.id == arms[1].id)
                    .values(workflow_revision_id=revision_id)
                )
            else:
                candidate = {
                    **original[0],
                    "id": "choice-after-trash",
                    "ordinal": 3,
                    "label": "New",
                }
                if operation == "missing":
                    candidate["workflow_revision_id"] = "revision-no-longer-present"
                session.execute(insert(GenerationExperimentArm).values(**candidate))
        session.rollback()
        session.expire_all()
        assert [
            {column.name: getattr(arm, column.name) for column in arm.__table__.columns}
            for arm in arms
        ] == original
        assert session.get(GenerationExperimentArm, "choice-after-trash") is None
    read = await client.get(f"{CREATE}/{accepted['id']}")
    assert read.status_code == 200 and read.json() == accepted


async def test_a_workflow_trashed_after_comparison_resolution_returns_a_coded_refusal(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _two_choices()
    body = _request(first, second)
    preflight = await client.post(PREFLIGHT, json=body)
    assert preflight.status_code == 200 and preflight.json()["outcome"] == "compatible"
    original_resolve = resolve_generation_experiment

    async def resolve_then_trash(*args: Any, **kwargs: Any) -> ExperimentResolution:
        resolution = await original_resolve(*args, **kwargs)
        assert resolution.compatible
        await _trash(client, first["workflow_revision_id"])
        return resolution

    monkeypatch.setattr(
        generation_experiment_api, "resolve_generation_experiment", resolve_then_trash
    )
    before = _records()
    response = await client.post(
        CREATE,
        json={
            **body,
            "idempotency_key": "comparison-racing-workflow-trash",
            "preflight_sha256": preflight.json()["preflight_sha256"],
        },
    )
    assert response.status_code == 409
    assert response.json()["code"] == "generation-experiment-preflight-changed"
    assert _records() == before
