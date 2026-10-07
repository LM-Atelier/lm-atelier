"""Workflow expiry respects original deadlines, history and atomic rollback."""

import asyncio
import copy
from datetime import UTC, datetime, timedelta
from typing import NoReturn

import pytest
from fastapi import FastAPI
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from test_media_recovery_restart import _open
from test_recovery_expiry import REAL_MAINTENANCE
from test_recovery_expiry import manual_expiry as manual_expiry
from test_workflow_recovery_api import _seed
from test_workflow_recovery_graph import _consumer
from test_workflow_recovery_history import _history

from local_lm import main, recovery_maintenance
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.events import EventBroker
from local_lm.models import (
    Job,
    RecoveryItem,
    RecoveryOperation,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
    WorkflowUseCasePreset,
)
from local_lm.recovery_previews import RecoveryPreviewConflict
from local_lm.recovery_v1 import PurgeRecoveryV1, RecoveryCommandV1
from local_lm.schemas import EventOut
from local_lm.workflow_recovery import preview_workflow_trash, trash_workflow

pytestmark = pytest.mark.usefixtures("manual_expiry")
NOW = datetime(2026, 10, 2, tzinfo=UTC)
DEADLINE = NOW + timedelta(days=30)


def _trash(
    now: datetime = NOW, *, history: bool = False, saved: bool = False
) -> tuple[str, str, str, str]:
    family_id, definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        if history:
            _consumer(session, "run", revision_id, "complete")
        if saved:
            session.add(
                WorkflowUseCasePreset(
                    name="Garden layout configuration",
                    use_case="image_generation",
                    settings_json={"workflow_revision_id": revision_id},
                    is_default=False,
                )
            )
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.artifact_sha256 = "a" * 64
        session.commit()
        preview = preview_workflow_trash(session, family_id, now)
        item = trash_workflow(
            session,
            family_id,
            RecoveryCommandV1(
                expected_revision=preview.revision,
                impact_sha256=preview.impact_sha256,
                operation_key="trash-expiring-workflow",
            ),
            now,
        )
        session.commit()
        return item.deletion_id, family_id, definition_id, revision_id


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.usefixtures("client")
async def test_workflow_expiry_uses_the_original_deadline_and_retains_completed_identity(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, history: bool
) -> None:
    deletion_id, family_id, definition_id, revision_id = _trash(history=history)
    before = _history()
    events: list[tuple[str, str | None, dict[str, object] | None]] = []
    broker: EventBroker = app.state.services.events
    original = broker.publish

    async def observed(
        event_type: str, entity_id: str | None = None, payload: dict[str, object] | None = None
    ) -> EventOut:
        if event_type == "recovery.updated":
            with SessionLocal() as session:
                item = session.get(RecoveryItem, entity_id)
                assert item is not None
                assert item.state == "purged"
            events.append((event_type, entity_id, payload))
        return await original(event_type, entity_id, payload)

    monkeypatch.setattr(app.state.services.events, "publish", observed)
    early = await recovery_maintenance.expire_recovery_batch(
        app.state.services, DEADLINE - timedelta(microseconds=1)
    )
    assert early.examined == early.purged == early.deferred == 0
    assert events == []
    result = await recovery_maintenance.expire_recovery_batch(app.state.services, DEADLINE)
    assert result.examined == result.purged == 1 and result.deferred == 0
    assert events == [("recovery.updated", deletion_id, {})]
    assert _history() == before
    with SessionLocal() as session:
        item = session.get(RecoveryItem, deletion_id)
        assert item is not None
        assert item.state == "purged" and item.purge_after.replace(tzinfo=UTC) == DEADLINE
        assert item.deleted_at.replace(tzinfo=UTC) == NOW
        assert session.scalar(select(func.count()).select_from(Job)) == 0
        revision = session.get(WorkflowRevision, revision_id)
        if history:
            assert revision is not None and revision.workflow_id == definition_id
            assert revision.artifact_sha256 == "a" * 64 and revision.api_graph_json == {}
            assert not revision.trusted
            family = session.get(WorkflowFamily, family_id)
            assert family is not None
            assert not family.enabled
        else:
            assert revision is None
            assert session.get(WorkflowDefinition, definition_id) is None
            assert session.get(WorkflowFamily, family_id) is None
        operations = session.scalar(select(func.count()).select_from(RecoveryOperation))
    repeated = await recovery_maintenance.expire_recovery_batch(app.state.services, DEADLINE)
    assert repeated.examined == repeated.purged == repeated.deferred == 0
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(RecoveryOperation)) == operations


@pytest.mark.usefixtures("client")
async def test_a_failed_workflow_expiry_rolls_back_history_minimization_and_authority_removal(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    deletion_id, _family_id, _definition_id, _revision_id = _trash(history=True)
    models = (
        RecoveryItem,
        RecoveryOperation,
        WorkflowFamily,
        WorkflowDefinition,
        WorkflowRevision,
        WorkflowPreference,
    )

    def rows() -> dict[str, list[dict[str, object]]]:
        with SessionLocal() as session:
            return {
                model.__tablename__: copy.deepcopy(
                    [dict(row) for row in session.execute(select(model.__table__)).mappings()]
                )
                for model in models
            }

    before = rows()
    original = recovery_maintenance._purge

    def fail_during_purge(candidate: recovery_maintenance._Candidate, now: datetime) -> bool:
        # Raise inside the lifecycle transaction after the real minimization.
        from local_lm.workflow_recovery import purge_workflow

        def interrupted(
            session: Session, deletion_id: str, command: PurgeRecoveryV1, now: datetime
        ) -> NoReturn:
            purge_workflow(session, deletion_id, command, now)
            raise RecoveryPreviewConflict("recovery-impact-stale")

        with monkeypatch.context() as patch:
            patch.setattr(recovery_maintenance, "purge_workflow", interrupted)
            return original(candidate, now)

    monkeypatch.setattr(recovery_maintenance, "_purge", fail_during_purge)
    result = await recovery_maintenance.expire_recovery_batch(app.state.services, DEADLINE)
    assert result.examined == result.deferred == 1 and result.purged == 0
    assert rows() == before
    with SessionLocal() as session:
        item = session.get(RecoveryItem, deletion_id)
        assert item is not None
        assert item.state == "recoverable"


@pytest.mark.usefixtures("client")
async def test_saved_workflow_configuration_defers_expiry_without_extending_its_deadline(
    app: FastAPI,
) -> None:
    deletion_id, family_id, _definition_id, revision_id = _trash(saved=True)
    result = await recovery_maintenance.expire_recovery_batch(
        app.state.services, DEADLINE + timedelta(days=7)
    )
    assert result.examined == result.deferred == 1 and result.purged == 0
    with SessionLocal() as session:
        item = session.get(RecoveryItem, deletion_id)
        assert item is not None
        assert item.state == "recoverable" and item.purge_after.replace(tzinfo=UTC) == DEADLINE
        assert session.get(WorkflowFamily, family_id) is not None
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        assert revision.api_graph_json


async def test_startup_expires_workflows_while_the_generation_lane_remains_paused(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _open(settings) as (_app, client):
        deletion_id, family_id, _definition_id, _revision_id = _trash(
            datetime.now(UTC) - timedelta(days=31)
        )
        policy = (await client.get("/api/queue/lanes/generation")).json()
        paused = await client.post(
            "/api/queue/lanes/generation/pause-after-current",
            json={
                "expected_revision": policy["revision"],
                "idempotency_key": "pause-workflow-expiry",
            },
        )
        assert paused.status_code == 200 and paused.json()["dispatch_state"] == "paused"
    monkeypatch.setattr(main, "maintain_recovery_expiry", REAL_MAINTENANCE)
    async with _open(settings) as (app, client):
        for _ in range(100):
            with SessionLocal() as session:
                item = session.get(RecoveryItem, deletion_id)
                assert item is not None
                state = item.state
            if state == "purged":
                break
            await asyncio.sleep(0.01)
        assert state == "purged"
        assert not app.state.recovery_maintenance.done()
        assert (await client.get("/api/queue/lanes/generation")).json()[
            "dispatch_state"
        ] == "paused"
        with SessionLocal() as session:
            assert session.get(WorkflowFamily, family_id) is None
            assert session.scalar(select(func.count()).select_from(Job)) == 0
    assert app.state.recovery_maintenance.done()
