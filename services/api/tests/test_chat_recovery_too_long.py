"""A deleted chat too large to check whole stays listed and can only be permanently deleted."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session
from test_chat_deletion import _image_exchange
from test_chat_recovery import _chat, _command, _history, _impact
from test_chat_recovery_graph import _artifact_id, _graph
from test_chat_recovery_graph import graph_session as graph_session
from test_chat_recovery_graph_bounds import _rows
from test_chat_recovery_organization import _admitted_with_the_least_room, _tag, _trash
from test_chat_recovery_transitions import NOW, _fields
from test_chat_recovery_transitions import _trash as _trash_garden
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm import chat_recovery_graph, recovery_maintenance
from local_lm.artifact_library import ensure_library_entry
from local_lm.chat_media_purge import generated_media_ids
from local_lm.chat_recovery import preview_chat_recovery, purge_chat
from local_lm.chat_recovery_graph import ChatRecoveryGraph
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.events import EventBroker
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    Job,
    MessagePart,
    RecoveryItem,
    RecoveryOperation,
    RecoveryPreviewRecord,
    Run,
)
from local_lm.recovery_previews import reserve_recovery_write
from local_lm.recovery_v1 import PurgeRecoveryV1, RecoveryAction, RecoveryCountsV1
from local_lm.schemas import EventOut

PICTURES = 8


@dataclass(frozen=True)
class LongChat:
    chat_id: str
    # The library entry of the picture its answer generated.
    generated: str
    # Pictures its answer also shows that have no library entry yet.
    pictures: tuple[str, ...]


@dataclass(frozen=True)
class Deleted:
    first: dict[str, Any]
    long: dict[str, Any]
    last: dict[str, Any]
    chat: LongChat
    room: int


async def _long_chat(client: AsyncClient, app: FastAPI) -> LongChat:
    chat_id = await _chat(client)
    await _image_exchange(client, chat_id, "Create a picture of garden beds")
    parts = _history(chat_id)["parts"]
    part = next(part for part in parts if part[-1])
    position = max(item[2] for item in parts if item[1] == part[1]) + 1
    store = app.state.services.artifacts
    pictures: list[str] = []
    with SessionLocal() as session:
        picture = session.get(Artifact, part[-1])
        assert picture is not None
        entry = ensure_library_entry(session, picture)
        assert entry is not None
        generated = entry.id
        content = store.resolve(picture).read_bytes()
        for index in range(PICTURES):
            added = store.ingest_bytes(
                session,
                content + f"garden bed {index}".encode(),
                kind=ArtifactKind.IMAGE,
                media_type="image/png",
            )
            session.add(
                MessagePart(
                    message_id=part[1],
                    position=position + index,
                    type="image",
                    artifact_id=added.id,
                )
            )
            pictures.append(added.id)
        session.commit()
        assert _entries(session, pictures) == 0
    return LongChat(chat_id, generated, tuple(pictures))


def _entries(session: Session, artifact_ids: list[str] | tuple[str, ...]) -> int | None:
    return session.scalar(
        select(func.count()).where(ArtifactLibraryEntry.artifact_id.in_(artifact_ids))
    )


async def _deleted_in_order(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    *,
    generated_media: bool = False,
) -> Deleted:
    """Delete a plain chat, the long chat with the least room it may enter with, then another."""
    first = await _trash(client, await _chat(client), "trash-first")
    chat = await _long_chat(client, app)
    room = await _admitted_with_the_least_room(client, monkeypatch, chat.chat_id)
    assert 0 < room <= len(chat.pictures)
    long = await _trash(client, chat.chat_id, "trash-long", generated_media=generated_media)
    last = await _trash(client, await _chat(client), "trash-last")
    return Deleted(first, long, last, chat, room)


async def _grow_past_its_bound(client: AsyncClient, chat: LongChat, room: int) -> tuple[str, ...]:
    """Favorite `room` of its pictures: each gains a library entry that the whole check reads."""
    for artifact_id in chat.pictures[:room]:
        favored = await client.patch(f"/api/artifacts/{artifact_id}", json={"favorite": True})
        assert favored.status_code == 200, favored.text
    with SessionLocal() as session:
        entries = tuple(
            session.scalars(
                select(ArtifactLibraryEntry.id).where(
                    ArtifactLibraryEntry.artifact_id.in_(chat.pictures)
                )
            )
        )
        assert len(entries) == room
        # With its deletion record, these entries put the whole check one row past its bound.
        reserve_recovery_write(session)
        with pytest.raises(chat_recovery_graph.ChatRecoveryGraphTooLarge):
            chat_recovery_graph.inspect_chat_recovery_graph(session, chat.chat_id)
    return entries


def _previews() -> set[str]:
    with SessionLocal() as session:
        return set(session.scalars(select(RecoveryPreviewRecord.revision)))


async def test_a_deleted_chat_too_large_to_check_is_listed_among_the_rest(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    deleted = await _deleted_in_order(client, app, monkeypatch)
    await _grow_past_its_bound(client, deleted.chat, deleted.room)
    previews = _previews()

    page = await client.get("/api/recovery-items")
    assert page.status_code == 200, page.text
    items = page.json()["items"]
    assert items == [
        deleted.last,
        {**deleted.long, "restore_conflicts": ["too_large"]},
        deleted.first,
    ]
    history = _history(deleted.chat.chat_id)
    assert (items[1]["counts"]["messages"], items[1]["counts"]["message_parts"]) == (
        len(history["messages"]),
        len(history["parts"]),
    )
    seen: list[str] = []
    params: dict[str, Any] = {"limit": 1}
    for _ in range(3):
        one = await client.get("/api/recovery-items", params=params)
        assert one.status_code == 200, one.text
        seen += [item["deletion_id"] for item in one.json()["items"]]
        params = {"limit": 1, "cursor": one.json()["next_cursor"]}
    assert seen == [item["deletion_id"] for item in items] and params["cursor"] is None
    recoverable = await client.get(
        "/api/recovery-items", params={"state": "recoverable", "kind": "chat"}
    )
    assert recoverable.status_code == 200, recoverable.text
    assert recoverable.json()["items"] == items
    assert _previews() == previews


@pytest.mark.parametrize("saved", ["missing", "invalid", "another_item"])
async def test_a_too_large_chat_without_its_saved_result_lists_zero_counts(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, saved: str
) -> None:
    deleted = await _deleted_in_order(client, app, monkeypatch)
    await _grow_past_its_bound(client, deleted.chat, deleted.room)
    with SessionLocal() as session:
        operation = session.scalar(
            select(RecoveryOperation).where(
                RecoveryOperation.deletion_id == deleted.long["deletion_id"]
            )
        )
        assert operation is not None
        if saved == "missing":
            session.delete(operation)
        elif saved == "invalid":
            operation.response_json = {**deleted.long, "counts": {"messages": -1}}
        else:
            operation.response_json = {**deleted.long, "deletion_id": deleted.first["deletion_id"]}
        session.commit()

    page = await client.get("/api/recovery-items")
    assert page.status_code == 200, page.text
    zero = RecoveryCountsV1().model_dump(mode="json")
    assert deleted.long["counts"] != zero
    assert page.json()["items"] == [
        deleted.last,
        {**deleted.long, "counts": zero, "restore_conflicts": ["too_large"]},
        deleted.first,
    ]


@pytest.mark.parametrize("media", ["keep_media", "delete_generated_media"])
async def test_a_deleted_chat_too_large_to_check_can_be_permanently_deleted_but_not_restored(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, media: str
) -> None:
    released = media == "delete_generated_media"
    deleted = await _deleted_in_order(client, app, monkeypatch, generated_media=released)
    chat_id, deletion_id = deleted.chat.chat_id, deleted.long["deletion_id"]
    early = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    favored = await _grow_past_its_bound(client, deleted.chat, deleted.room)
    history = _history(chat_id)
    assert history["jobs"]

    stale = await client.post(
        f"/api/recovery-items/{deletion_id}/purge",
        json={**_command(early, "purge-before-growth"), "acknowledgement": "permanently-delete"},
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()["code"] == "recovery-impact-stale"
    impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    assert impact["available_actions"] == ["purge"]
    assert impact["counts"] == deleted.long["counts"]
    assert (
        impact["counts"]["messages"],
        impact["counts"]["message_parts"],
        impact["counts"]["runs"],
    ) == (len(history["messages"]), len(history["parts"]), len(history["runs"]))
    refused = await client.post(
        f"/api/recovery-items/{deletion_id}/restore", json=_command(impact, "restore-long")
    )
    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "chat-recovery-graph-too-large"
    assert (await client.get(f"/api/chats/{chat_id}")).status_code == 404
    with SessionLocal() as session:
        item = session.get(RecoveryItem, deletion_id)
        assert item is not None and item.state == "recoverable"

    purged = await client.post(
        f"/api/recovery-items/{deletion_id}/purge",
        json={**_command(impact, "purge-long"), "acknowledgement": "permanently-delete"},
    )
    assert purged.status_code == 200, purged.text
    with SessionLocal() as session:
        assert session.get(Chat, chat_id) is None
        jobs = [job[0] for job in history["jobs"]]
        assert not list(session.scalars(select(Job.id).where(Job.id.in_(jobs))))
        assert all(session.get(ArtifactLibraryEntry, entry) is not None for entry in favored)
        assert (session.get(ArtifactLibraryEntry, deleted.chat.generated) is None) == released
    page = await client.get("/api/recovery-items")
    assert page.status_code == 200, page.text
    assert page.json()["items"] == [deleted.last, deleted.first]


async def test_expiry_deletes_a_chat_too_large_to_check_that_was_never_listed(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = await _long_chat(client, app)
    room = await _admitted_with_the_least_room(client, monkeypatch, chat.chat_id)
    long = await _trash(client, chat.chat_id, "trash-long")
    await _grow_past_its_bound(client, chat, room)
    events: list[tuple[str | None, str | None]] = []
    broker: EventBroker = app.state.services.events
    publish = broker.publish

    async def observed(
        event_type: str, entity_id: str | None = None, payload: dict[str, Any] | None = None
    ) -> EventOut:
        if event_type == "recovery.updated":
            with SessionLocal() as session:
                item = session.get(RecoveryItem, entity_id)
                events.append((entity_id, item.state if item else None))
        return await publish(event_type, entity_id, payload)

    monkeypatch.setattr(app.state.services.events, "publish", observed)
    deadline = datetime.fromisoformat(long["purge_after"])
    expired = await recovery_maintenance.expire_recovery_batch(app.state.services, deadline)
    assert (expired.examined, expired.purged, expired.deferred) == (1, 1, 0)
    assert events == [(long["deletion_id"], "purged")]
    with SessionLocal() as session:
        assert session.get(Chat, chat.chat_id) is None
    again = await recovery_maintenance.expire_recovery_batch(app.state.services, deadline)
    assert again.examined == 0


async def test_a_batch_deletes_a_chat_too_large_to_check_and_will_not_restore_it(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    deleted = await _deleted_in_order(client, app, monkeypatch)
    await _grow_past_its_bound(client, deleted.chat, deleted.room)
    first, long = deleted.first["deletion_id"], deleted.long["deletion_id"]

    restore = await client.post(
        "/api/recovery-items/batches", json={"deletion_ids": [first, long], "action": "restore"}
    )
    assert restore.status_code == 200, restore.text
    assert restore.json()["available"] is False
    assert {
        member["deletion_id"]: member["impact"]["available_actions"]
        for member in restore.json()["items"]
    } == {first: ["restore", "purge"], long: ["purge"]}
    refused = await client.post(
        f"/api/recovery-items/batches/{restore.json()['batch_id']}/apply",
        json=_command(restore.json(), "restore-both"),
    )
    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "recovery-batch-unavailable"

    purge = await client.post(
        "/api/recovery-items/batches", json={"deletion_ids": [first, long], "action": "purge"}
    )
    assert purge.status_code == 200, purge.text
    assert purge.json()["available"] is True
    applied = await client.post(
        f"/api/recovery-items/batches/{purge.json()['batch_id']}/apply",
        json={**_command(purge.json(), "purge-both"), "acknowledgement": "permanently-delete"},
    )
    assert applied.status_code == 200, applied.text
    with SessionLocal() as session:
        assert session.get(Chat, deleted.first["subject_id"]) is None
        assert session.get(Chat, deleted.chat.chat_id) is None
    page = await client.get("/api/recovery-items")
    assert page.status_code == 200, page.text
    assert page.json()["items"] == [deleted.last]


async def test_a_deleted_chat_with_damaged_references_still_refuses_rather_than_offering_deletion(
    client: AsyncClient, app: FastAPI
) -> None:
    """Only a chat too large to check is described from what permanent deletion reads.

    Damage is not size. The deletion read never reads the artifact rows a chat
    reaches, so describing a damaged chat from it would offer to delete what the
    whole check could not vouch for.
    """
    chat = await _long_chat(client, app)
    deletion_id = (await _trash(client, chat.chat_id, "trash-damaged"))["deletion_id"]
    with SessionLocal() as session:
        session.execute(text("DROP TRIGGER IF EXISTS artifacts_artifact_reference_update_guard"))
        session.execute(
            update(Artifact).where(Artifact.id == chat.pictures[0]).values(metadata_json=[])
        )
        session.commit()
        # The damage is something the deletion read alone would never see.
        assert chat_recovery_graph.inspect_chat_purge_graph(session, chat.chat_id)
        session.rollback()

    page = await client.get("/api/recovery-items")
    assert page.status_code == 409, page.text
    assert page.json()["code"] == "chat-recovery-graph-invalid"
    impact = await client.get(f"/api/recovery-items/{deletion_id}/impact")
    assert impact.status_code == 409, impact.text
    assert impact.json()["code"] == "chat-recovery-graph-invalid"


def _shared_jobs(session: Session) -> None:
    """Add jobs the conversation owns alone beside jobs another conversation also owns."""
    session.add_all(
        [
            Job(id="job-source", status="complete", payload_json={"source_run_id": "run-garden"}),
            Job(
                id="job-other",
                run_id="run-other",
                status="complete",
                payload_json={"source_run_id": "run-garden"},
            ),
            Job(
                id="job-shared",
                run_id="run-other",
                status="complete",
                payload_json={"chat_id": "chat-garden"},
            ),
            Job(
                id="job-elsewhere",
                run_id="run-garden",
                status="complete",
                payload_json={"chat_id": "chat-other"},
            ),
        ]
    )
    session.commit()


def _tagged(session: Session) -> None:
    _tag(session, 1)


def _renamed(session: Session) -> None:
    session.execute(update(ArtifactLibraryEntry).values(display_name="Garden preview", version=2))


def _studio(session: Session) -> None:
    session.add(
        Chat(
            id="chat-studio",
            title="Garden studio",
            scope="studio",
            origin_json={"source_chat_id": "chat-garden"},
        )
    )


def test_permanent_deletion_reads_only_rows_that_cannot_grow_after_deletion(
    graph_session: Session,
) -> None:
    from local_lm.chat_recovery_graph import inspect_chat_purge_graph

    _shared_jobs(graph_session)
    graph_session.add(
        RecoveryItem(
            kind="chat",
            subject_id="chat-garden",
            display_label="Garden notes",
            deleted_at=NOW,
            purge_after=NOW + timedelta(days=30),
            subject_revision="0" * 64,
        )
    )
    graph_session.commit()

    def read() -> tuple[ChatRecoveryGraph, ChatRecoveryGraph]:
        whole = _graph(graph_session)
        purge = inspect_chat_purge_graph(graph_session, "chat-garden")
        graph_session.commit()
        return whole, purge

    whole, purge = read()
    assert purge.owned_job_ids == whole.owned_job_ids == ("job-garden", "job-source")
    assert generated_media_ids(purge) == generated_media_ids(whole) == {_artifact_id(1)}
    owned = (
        "messages",
        "message_parts",
        "response_revisions",
        "runs",
        "jobs",
        "work_plans",
        "references",
        "active_work",
    )
    assert [getattr(purge.snapshot.counts, name) for name in owned] == [
        getattr(whole.snapshot.counts, name) for name in owned
    ]
    changes: tuple[Callable[[Session], None], ...] = (_tagged, _renamed, _studio)
    for change in changes:
        change(graph_session)
        graph_session.commit()
        changed, unchanged = read()
        assert changed.snapshot.fingerprint != whole.snapshot.fingerprint
        assert unchanged.snapshot.fingerprint == purge.snapshot.fingerprint
        whole = changed
    graph_session.execute(
        update(Run).where(Run.id == "run-other").values(duration_ms=7, updated_at=Run.updated_at)
    )
    graph_session.commit()
    changed, also_changed = read()
    assert changed.snapshot.fingerprint != whole.snapshot.fingerprint
    assert also_changed.snapshot.fingerprint != purge.snapshot.fingerprint


def test_permanent_deletion_of_a_too_large_chat_removes_exactly_its_own_jobs(
    graph_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _shared_jobs(graph_session)
    item, _ = _trash_garden(graph_session)
    rows = _rows(graph_session, monkeypatch)
    # Purge's own read sits at least six rows lower: it reads no artifact or library entry.
    monkeypatch.setattr(chat_recovery_graph, "MAX_GRAPH_ROWS", rows - 1)
    with pytest.raises(chat_recovery_graph.ChatRecoveryGraphTooLarge):
        _graph(graph_session)
    graph_session.rollback()

    preview = preview_chat_recovery(graph_session, item.deletion_id, NOW)
    graph_session.commit()
    assert preview.available_actions == (RecoveryAction.PURGE,)
    command = PurgeRecoveryV1.model_validate(
        {**_fields(preview, "purge-garden"), "acknowledgement": "permanently-delete"}
    )
    purge_chat(graph_session, item.deletion_id, command, NOW)
    graph_session.commit()
    assert graph_session.get(Chat, "chat-garden") is None
    assert set(graph_session.scalars(select(Job.id))) == {
        "job-other",
        "job-shared",
        "job-elsewhere",
    }
