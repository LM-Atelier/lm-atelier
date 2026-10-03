"""Accepted workflow consumers and historical snapshots constrain recovery independently."""

from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient
from test_workflow_recovery_api import _seed

from local_lm.db import SessionLocal
from local_lm.models import (
    Chat,
    Job,
    Message,
    RecoveryItem,
    Run,
    RunContextSnapshot,
    WorkflowActivation,
    WorkflowInstallOffer,
    WorkPlan,
    WorkStep,
)
from local_lm.recovery_previews import RecoveryPreviewConflict
from local_lm.recovery_v1 import RecoveryAction, RecoveryConflict
from local_lm.workflow_recovery_graph import inspect_workflow_recovery


def _consumer(session, kind: str, revision_id: str, status: str):
    if kind == "job":
        row = Job(
            id="job-workflow",
            status=status,
            payload_json={"accepted": {"workflow_revision_id": revision_id}},
        )
        session.add(row)
        return row
    session.add(Chat(id="chat-workflow", title="Garden notes"))
    session.flush()
    if kind == "step":
        session.add(
            WorkPlan(
                id="plan-workflow", chat_id="chat-workflow", transcript_sequence=1, status=status
            )
        )
        session.flush()
        row = WorkStep(
            id="step-workflow",
            plan_id="plan-workflow",
            ordinal=0,
            operation="text_to_image",
            status=status,
            settings_json={"workflow_revision_id": revision_id},
        )
        session.add(row)
        return row
    session.add_all(
        [
            Message(id="user-workflow", chat_id="chat-workflow"),
            Message(id="assistant-workflow", chat_id="chat-workflow", role="assistant"),
        ]
    )
    session.flush()
    row = Run(
        id="run-workflow",
        chat_id="chat-workflow",
        user_message_id="user-workflow",
        assistant_message_id="assistant-workflow",
        status=status,
        workflow_revision_id=revision_id if kind == "run" else None,
    )
    session.add(row)
    session.flush()
    if kind == "snapshot":
        session.add(
            RunContextSnapshot(
                run_id=row.id,
                sha256="a" * 64,
                payload_json={"accepted": {"workflow_revision_id": revision_id}},
            )
        )
    return row


@pytest.mark.parametrize("kind", ["job", "step", "run", "snapshot"])
@pytest.mark.parametrize("status", ["queued", "paused", "running", "complete"])
async def test_workflow_recovery_finds_consumers_in_identity_and_accepted_json(
    client: AsyncClient,
    kind: str,
    status: str,
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    now = datetime.now(UTC)
    with SessionLocal() as session:
        _consumer(session, kind, revision_id, status)
        session.flush()
        snapshot = inspect_workflow_recovery(session, family_id, now)
        assert snapshot.counts.references > 0
        assert (
            snapshot.counts.workflow_families
            == snapshot.counts.workflow_definitions
            == snapshot.counts.workflow_revisions
            == 1
        )
        if status == "complete":
            assert snapshot.counts.active_work == 0
            assert snapshot.available_actions == (RecoveryAction.TRASH,)
            session.add(
                RecoveryItem(
                    kind="workflow_family",
                    subject_id=family_id,
                    display_label="Garden layout",
                    deleted_at=now,
                    purge_after=now.replace(year=now.year + 1),
                    state="recoverable",
                    subject_revision="a" * 64,
                )
            )
            session.flush()
            historical = inspect_workflow_recovery(session, family_id, now)
            assert historical.available_actions == (RecoveryAction.RESTORE, RecoveryAction.PURGE)
        else:
            assert snapshot.counts.active_work > 0
            assert snapshot.conflicts == (RecoveryConflict.ACTIVE_WORK,)
            assert not snapshot.available_actions


async def test_workflow_private_revision_changes_without_changing_public_counts(
    client: AsyncClient,
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    now = datetime.now(UTC)
    with SessionLocal() as session:
        job = _consumer(session, "job", revision_id, "complete")
        session.flush()
        first = inspect_workflow_recovery(session, family_id, now)
        job.payload_json = {"accepted": {"workflow_revision_id": revision_id, "output_count": 2}}
        session.flush()
        second = inspect_workflow_recovery(session, family_id, now)
        assert first.counts == second.counts
        assert first.available_actions == second.available_actions
        assert first.fingerprint != second.fingerprint


async def test_workflow_recovery_does_not_interpret_description_as_identity(
    client: AsyncClient,
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        session.add(Job(id="job-other", status="running", payload_json={"prompt": revision_id}))
        session.flush()
        snapshot = inspect_workflow_recovery(session, family_id, datetime.now(UTC))
        assert snapshot.counts.active_work == snapshot.counts.references == 0
        assert snapshot.available_actions == (RecoveryAction.TRASH,)


async def test_workflow_graph_refuses_excessive_canonical_data(
    client: AsyncClient, monkeypatch
) -> None:
    from local_lm import chat_recovery_graph

    family_id, _definition_id, _revision_id, _graph = _seed()
    monkeypatch.setattr(chat_recovery_graph, "MAX_GRAPH_BYTES", 1)
    with (
        SessionLocal() as session,
        pytest.raises(RecoveryPreviewConflict, match="workflow-recovery-graph-invalid"),
    ):
        inspect_workflow_recovery(session, family_id, datetime.now(UTC))


async def test_unrelated_history_does_not_change_a_workflow_recovery_preview(
    client: AsyncClient,
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    now = datetime.now(UTC)
    with SessionLocal() as session:
        job = Job(id="unrelated-job", status="running", payload_json={"prompt": revision_id})
        session.add(job)
        session.flush()
        first = inspect_workflow_recovery(session, family_id, now)
        job.payload_json = {"prompt": "Keep the garden paths wide", "output_count": 2}
        session.flush()
        second = inspect_workflow_recovery(session, family_id, now)
        assert first == second


@pytest.mark.parametrize("kind", ["job", "step", "run", "snapshot"])
async def test_workflow_consumers_are_counted_once_by_their_canonical_identity(
    client: AsyncClient, kind: str
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        _consumer(session, kind, revision_id, "complete")
        session.flush()
        snapshot = inspect_workflow_recovery(session, family_id, datetime.now(UTC))
        assert snapshot.counts.references == (2 if kind == "snapshot" else 1)


@pytest.mark.parametrize("kind", ["activation", "offer"])
async def test_runless_queued_work_names_the_required_activation_or_install_offer(
    client: AsyncClient, kind: str
) -> None:
    family_id, _definition_id, revision_id, _graph = _seed()
    with SessionLocal() as session:
        if kind == "activation":
            owner = WorkflowActivation(
                workflow_revision_id=revision_id,
                resolver_version="1",
                dependency_contract_sha256="a" * 64,
                binding_sha256="b" * 64,
                state="disabled",
                is_active=False,
            )
            key = "workflow_activation_id"
        else:
            owner = WorkflowInstallOffer(
                workflow_revision_id=revision_id,
                workflow_artifact_sha256="a" * 64,
                dependency_contract_sha256="b" * 64,
                binding_plan_sha256="c" * 64,
                offer_sha256="d" * 64,
                plan_count=1,
                total_bytes=1,
                status="invalidated",
            )
            key = "workflow_install_offer_id"
        session.add(owner)
        session.flush()
        session.add(Job(status="queued", payload_json={"accepted": {key: owner.id}}))
        session.flush()
        snapshot = inspect_workflow_recovery(session, family_id, datetime.now(UTC))
        assert snapshot.counts.active_work == 1
        assert snapshot.conflicts == (RecoveryConflict.ACTIVE_WORK,)
        assert not snapshot.available_actions
