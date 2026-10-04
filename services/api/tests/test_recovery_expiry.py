"""Expiry resumes from original deadlines and never creates generation work."""

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from test_media_recovery import CONTENT, _seed
from test_media_recovery_restart import _open

from local_lm import main, recovery_maintenance
from local_lm.chat_recovery import (
    preview_chat_recovery,
    preview_chat_trash,
    restore_chat,
    trash_chat,
)
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.events import EventBroker
from local_lm.media_recovery import preview_media_trash, purge_media, trash_media
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    Job,
    RecoveryItem,
    RecoveryOperation,
)
from local_lm.recovery_maintenance import ExpiryCursor, _Candidate
from local_lm.recovery_previews import RecoveryPreviewConflict
from local_lm.recovery_v1 import (
    PurgeRecoveryV1,
    RecoveryCommandV1,
    RecoveryResultV1,
    RestoreRecoveryV1,
    TrashChatV1,
)
from local_lm.schemas import EventOut

NOW = datetime(2026, 10, 2, tzinfo=UTC)
DEADLINE = NOW + timedelta(days=30)
REAL_MAINTENANCE = recovery_maintenance.maintain_recovery_expiry


@pytest.fixture(autouse=True)
def manual_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    async def dormant(*_args: object, **_kwargs: object) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(main, "maintain_recovery_expiry", dormant)


async def _trash(app: FastAPI, kind: str, now: datetime = NOW) -> tuple[str, str, str | None]:
    await asyncio.wait_for(app.state.retention_sweep, 30)
    with SessionLocal() as session:
        if kind == "chat":
            chat = Chat(title="Garden expiry")
            session.add(chat)
            session.commit()
            subject_id = chat.id
            preview = preview_chat_trash(session, subject_id, now)
            chat_command = TrashChatV1(
                expected_revision=preview.revision,
                impact_sha256=preview.impact_sha256,
                operation_key="trash-expiring-chat",
            )
            item = trash_chat(session, subject_id, chat_command, now)
            artifact_id = None
        else:
            artifact, entry, _collection, _tag = _seed(app.state.services.artifacts, session)
            subject_id, artifact_id = entry.id, artifact.id
            preview = preview_media_trash(session, subject_id, now)
            command = RecoveryCommandV1(
                expected_revision=preview.revision,
                impact_sha256=preview.impact_sha256,
                operation_key="trash-expiring-media",
            )
            item = trash_media(session, subject_id, command, now)
        session.commit()
        return item.deletion_id, subject_id, artifact_id


@pytest.mark.parametrize("kind", ["chat", "media_library_entry"])
@pytest.mark.usefixtures("client")
async def test_expiry_uses_the_exact_original_deadline_and_publishes_after_commit(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    deletion_id, subject_id, artifact_id = await _trash(app, kind)
    events: list[tuple[str, str | None, dict[str, Any] | None]] = []
    broker: EventBroker = app.state.services.events
    publish = broker.publish

    async def observed(
        event_type: str, entity_id: str | None = None, payload: dict[str, Any] | None = None
    ) -> EventOut:
        if event_type == "recovery.updated":
            with SessionLocal() as session:
                item = session.get(RecoveryItem, entity_id)
                assert item is not None and item.state == "purged"
            events.append((event_type, entity_id, payload))
        return await publish(event_type, entity_id, payload)

    monkeypatch.setattr(app.state.services.events, "publish", observed)
    early = await recovery_maintenance.expire_recovery_batch(
        app.state.services, DEADLINE - timedelta(microseconds=1)
    )
    assert early.examined == early.purged == early.deferred == 0
    assert events == []
    result = await recovery_maintenance.expire_recovery_batch(app.state.services, DEADLINE)
    assert result.examined == result.purged == 1 and result.deferred == 0
    assert events == [("recovery.updated", deletion_id, {})]
    with SessionLocal() as session:
        item = session.get(RecoveryItem, deletion_id)
        assert item is not None and item.purge_after.replace(tzinfo=UTC) == DEADLINE
        assert session.scalar(select(func.count()).select_from(Job)) == 0
        if kind == "chat":
            assert session.get(Chat, subject_id) is None
        else:
            assert session.get(ArtifactLibraryEntry, subject_id) is None
            artifact = session.get(Artifact, artifact_id)
            assert artifact is not None and not artifact.favorite
            assert app.state.services.artifacts.resolve(artifact).read_bytes() == CONTENT
        operations = session.scalar(select(func.count()).select_from(RecoveryOperation))
    repeated = await recovery_maintenance.expire_recovery_batch(app.state.services, DEADLINE)
    assert repeated.examined == repeated.purged == repeated.deferred == 0
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(RecoveryOperation)) == operations


@pytest.mark.usefixtures("client")
async def test_expiry_rolls_back_a_failed_item_and_seeks_past_it(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_id, entry_id, artifact_id = await _trash(
        app, "media_library_entry", NOW - timedelta(days=1)
    )
    second_id, chat_id, _ = await _trash(app, "chat")
    original = purge_media

    def fail_after_transition(
        session: Session, deletion_id: str, command: PurgeRecoveryV1, now: datetime
    ) -> RecoveryResultV1:
        original(session, deletion_id, command, now)
        raise RecoveryPreviewConflict("recovery-impact-stale")

    monkeypatch.setattr(recovery_maintenance, "purge_media", fail_after_transition)
    first = await recovery_maintenance.expire_recovery_batch(app.state.services, DEADLINE, limit=1)
    assert first.examined == first.deferred == 1 and first.purged == 0
    assert first.next_cursor is not None and first.next_cursor.deletion_id == first_id
    with SessionLocal() as session:
        item = session.get(RecoveryItem, first_id)
        entry = session.get(ArtifactLibraryEntry, entry_id)
        artifact = session.get(Artifact, artifact_id)
        assert item is not None and item.state == "recoverable"
        assert entry is not None and entry.state == "trashed" and entry.favorite
        assert artifact is not None and artifact.favorite
        assert app.state.services.artifacts.resolve(artifact).read_bytes() == CONTENT
        assert session.scalar(select(func.count()).select_from(RecoveryOperation)) == 2
    second = await recovery_maintenance.expire_recovery_batch(
        app.state.services, DEADLINE, after=first.next_cursor, limit=1
    )
    assert second.examined == second.purged == 1 and second.deferred == 0
    with SessionLocal() as session:
        second_item = session.get(RecoveryItem, second_id)
        assert second_item is not None
        assert second_item.state == "purged"
        assert session.get(Chat, chat_id) is None


@pytest.mark.usefixtures("client")
async def test_an_expiry_candidate_is_revalidated_after_waiting_for_the_chat_lock(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deletion_id, chat_id, _ = await _trash(app, "chat")
    selected = asyncio.Event()
    loop = asyncio.get_running_loop()
    original = recovery_maintenance._candidates

    def observe(now: datetime, after: ExpiryCursor | None, limit: int) -> list[_Candidate]:
        rows = original(now, after, limit)
        loop.call_soon_threadsafe(selected.set)
        return rows

    monkeypatch.setattr(recovery_maintenance, "_candidates", observe)
    guard = app.state.services.orchestrator.chat_guard(chat_id)
    await guard.acquire()
    operation = asyncio.create_task(
        recovery_maintenance.expire_recovery_batch(app.state.services, DEADLINE)
    )
    try:
        await asyncio.wait_for(selected.wait(), 3)
        assert not operation.done()
        with SessionLocal() as session:
            preview = preview_chat_recovery(session, deletion_id, NOW)
            restore_chat(
                session,
                deletion_id,
                RestoreRecoveryV1(
                    expected_revision=preview.revision,
                    impact_sha256=preview.impact_sha256,
                    operation_key="restore-before-expiry-lock",
                ),
                NOW,
            )
            session.commit()
    finally:
        guard.release()
    result = await operation
    assert result.examined == result.deferred == 1 and result.purged == 0
    with SessionLocal() as session:
        assert session.get(Chat, chat_id) is not None
        assert session.get(RecoveryItem, deletion_id) is None


@pytest.mark.usefixtures("client")
async def test_shutdown_keeps_the_chat_lock_until_the_database_thread_finishes(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _deletion_id, chat_id, _ = await _trash(app, "chat")
    started, release = threading.Event(), threading.Event()
    original = recovery_maintenance._purge

    def pause(candidate: _Candidate, now: datetime) -> bool:
        started.set()
        assert release.wait(3)
        return original(candidate, now)

    monkeypatch.setattr(recovery_maintenance, "_purge", pause)
    operation = asyncio.create_task(
        recovery_maintenance.expire_recovery_batch(app.state.services, DEADLINE)
    )
    try:
        assert await asyncio.to_thread(started.wait, 3)
        operation.cancel()
        await asyncio.sleep(0)
        operation.cancel()
        await asyncio.sleep(0)
        assert not operation.done()
        assert app.state.services.orchestrator.chat_guard(chat_id).locked()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert not app.state.services.orchestrator.chat_guard(chat_id).locked()


@pytest.mark.parametrize("kind", ["chat", "media_library_entry"])
async def test_application_restart_runs_due_expiry_without_generation_jobs(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    async with _open(settings) as (app, client):
        deletion_id, subject_id, _ = await _trash(app, kind, datetime.now(UTC) - timedelta(days=31))
        policy = await client.get("/api/queue/lanes/generation")
        assert policy.status_code == 200
        paused = await client.post(
            "/api/queue/lanes/generation/pause-after-current",
            json={
                "expected_revision": policy.json()["revision"],
                "idempotency_key": "pause-before-expiry-restart",
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
        policy = await client.get("/api/queue/lanes/generation")
        assert policy.status_code == 200 and policy.json()["dispatch_state"] == "paused"
        with SessionLocal() as session:
            assert session.scalar(select(func.count()).select_from(Job)) == 0
            model = Chat if kind == "chat" else ArtifactLibraryEntry
            assert session.get(model, subject_id) is None
    assert app.state.recovery_maintenance.done()


@pytest.mark.parametrize("limit", [0, 21])
@pytest.mark.usefixtures("client")
async def test_expiry_refuses_unbounded_batches_before_any_mutation(
    app: FastAPI, limit: int
) -> None:
    deletion_id, _, _ = await _trash(app, "chat")
    with pytest.raises(ValueError, match="one and twenty"):
        await recovery_maintenance.expire_recovery_batch(app.state.services, DEADLINE, limit=limit)
    with SessionLocal() as session:
        item = session.get(RecoveryItem, deletion_id)
        assert item is not None
        assert item.state == "recoverable"
