"""Recovery pages seek through stable identities and expose only current structural facts."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_chat_recovery import _command, _impact
from test_empty_chat_deletion import session as session

from local_lm.chat_recovery import chat_recovery_page
from local_lm.db import SessionLocal
from local_lm.models import Chat, Message, MessagePart, RecoveryItem, RecoveryPreviewRecord
from local_lm.recovery_previews import RecoveryPreviewConflict
from local_lm.recovery_v1 import RecoveryConflict, RecoveryState

NOW = datetime(2026, 10, 2, tzinfo=UTC)


def _item(
    session: Session,
    identity: str,
    *,
    state: str = "recoverable",
    kind: str = "chat",
    scope: str = "standard",
    date: datetime = NOW,
) -> None:
    session.add(Chat(id=f"chat-{identity}", title="Garden notes", scope=scope))
    session.flush()
    session.add(
        RecoveryItem(
            deletion_id=f"deleted-{identity}",
            kind=kind,
            subject_id=f"chat-{identity}",
            display_label="Garden notes",
            deleted_at=date,
            purge_after=date + timedelta(days=30),
            state=state,
            subject_revision="a" * 64,
        )
    )
    session.commit()


def test_seek_pages_use_ties_without_hidden_or_purged_items_spending_the_limit(
    session: Session,
) -> None:
    for identity in ("a", "b", "c"):
        _item(session, identity)
    _item(session, "z-purged", state="purged")
    _item(session, "z-hidden", scope="prompt_expansion")
    _item(session, "z-project", kind="project")
    first = chat_recovery_page(session, NOW, limit=1)
    session.commit()
    assert [item.deletion_id for item in first.items] == ["deleted-c"]
    assert first.next_cursor is not None
    _item(session, "new", date=NOW + timedelta(seconds=1))
    second = chat_recovery_page(session, NOW, limit=1, cursor=first.next_cursor)
    session.commit()
    assert [item.deletion_id for item in second.items] == ["deleted-b"]
    assert second.next_cursor is not None
    third = chat_recovery_page(session, NOW, limit=1, cursor=second.next_cursor)
    session.commit()
    assert [item.deletion_id for item in third.items] == ["deleted-a"]
    assert third.next_cursor is None
    assert session.scalar(select(RecoveryPreviewRecord)) is None


def test_state_and_date_filters_run_before_paging_and_bind_the_cursor(session: Session) -> None:
    for identity in ("a", "b"):
        _item(session, identity, state="blocked")
    _item(session, "c")
    _item(session, "old", state="blocked", date=NOW - timedelta(days=1))
    page = chat_recovery_page(session, NOW, limit=1, state=RecoveryState.BLOCKED, deleted_since=NOW)
    session.commit()
    assert [item.deletion_id for item in page.items] == ["deleted-b"]
    assert page.next_cursor is not None
    with pytest.raises(RecoveryPreviewConflict, match="recovery-cursor-invalid"):
        chat_recovery_page(session, NOW, cursor=page.next_cursor, deleted_since=NOW)
    with pytest.raises(RecoveryPreviewConflict, match="recovery-cursor-invalid"):
        chat_recovery_page(session, NOW, cursor=page.next_cursor, state=RecoveryState.BLOCKED)
    next_page = chat_recovery_page(
        session,
        NOW,
        limit=1,
        cursor=page.next_cursor,
        state=RecoveryState.BLOCKED,
        deleted_since=NOW,
    )
    session.commit()
    assert [item.deletion_id for item in next_page.items] == ["deleted-a"]
    assert next_page.next_cursor is None


@pytest.mark.parametrize("cursor", ["%%%", "W10=", "bnVsbA==", "x" * 2049])
def test_malformed_cursors_fail_without_creating_previews(session: Session, cursor: str) -> None:
    with pytest.raises(RecoveryPreviewConflict, match="recovery-cursor-invalid"):
        chat_recovery_page(session, NOW, cursor=cursor)
    assert session.scalar(select(RecoveryPreviewRecord)) is None


def test_listing_reads_canonical_counts_and_reports_a_missing_original_project(
    session: Session,
) -> None:
    session.add(Chat(id="chat-counts", title="Garden notes"))
    session.flush()
    session.add(Message(id="message-counts", chat_id="chat-counts", role="user", status="complete"))
    session.flush()
    session.add(
        MessagePart(message_id="message-counts", position=0, type="text", text="Garden spacing")
    )
    session.flush()
    session.add(
        RecoveryItem(
            deletion_id="deleted-counts",
            kind="chat",
            subject_id="chat-counts",
            display_label="Garden notes",
            original_project_id="project-missing",
            original_project_label="Garden",
            deleted_at=NOW,
            purge_after=NOW + timedelta(days=30),
            state="recoverable",
            subject_revision="a" * 64,
        )
    )
    session.commit()
    page = chat_recovery_page(session, NOW)
    session.commit()
    assert len(page.items) == 1
    item = page.items[0]
    assert item.counts.messages == 1 and item.counts.message_parts == 1
    assert item.restore_conflicts == (RecoveryConflict.ORIGINAL_PROJECT_MISSING,)
    assert item.original_location.project_label == "Garden"
    assert item.counts.reclaimable_bytes == 0
    assert item.deleted_at.tzinfo == UTC
    assert "Garden spacing" not in page.model_dump_json()
    assert session.scalar(select(RecoveryPreviewRecord)) is None


async def test_http_list_pages_deleted_chats_and_refuses_a_changed_filter(
    client: AsyncClient,
) -> None:
    identities = []
    for number in range(3):
        created = await client.post("/api/chats", json={"title": f"Garden notes {number}"})
        assert created.status_code == 201
        identity = created.json()["id"]
        impact = await _impact(client, f"/api/chats/{identity}/deletion-impact")
        trashed = await client.post(
            f"/api/chats/{identity}/trash", json=_command(impact, f"trash-{number}")
        )
        assert trashed.status_code == 200
        identities.append(trashed.json()["deletion_id"])
    first = await client.get("/api/recovery-items", params={"limit": 1, "state": "recoverable"})
    assert first.status_code == 200
    assert first.headers["cache-control"] == "no-store"
    assert [item["deletion_id"] for item in first.json()["items"]] == [identities[-1]]
    cursor = first.json()["next_cursor"]
    refused = await client.get("/api/recovery-items", params={"cursor": cursor})
    assert refused.status_code == 400 and refused.json()["code"] == "recovery-cursor-invalid"
    second = await client.get(
        "/api/recovery-items", params={"limit": 2, "cursor": cursor, "state": "recoverable"}
    )
    assert second.status_code == 200
    assert [item["deletion_id"] for item in second.json()["items"]] == identities[-2::-1]
    assert second.json()["next_cursor"] is None
    assert (await client.get("/api/recovery-items", params={"limit": 21})).status_code == 422


@pytest.mark.parametrize("action", ["restore", "purge"])
async def test_membership_events_follow_committed_transitions_without_content(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    observed: list[tuple[str, str | None]] = []
    publish = app.state.services.events.publish

    async def observe(event_type: str, entity_id: str | None = None, payload: dict | None = None):
        if event_type == "recovery.updated":
            assert entity_id is not None and payload == {}
            with SessionLocal() as current:
                membership = current.get(RecoveryItem, entity_id)
                observed.append((entity_id, membership.state if membership else None))
        return await publish(event_type, entity_id, payload)

    monkeypatch.setattr(app.state.services.events, "publish", observe)
    created = await client.post("/api/chats", json={"title": "Garden events"})
    assert created.status_code == 201
    chat_id = created.json()["id"]
    impact = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    stale = _command(impact, "stale-trash-events") | {"expected_revision": "a" * 64}
    assert (await client.post(f"/api/chats/{chat_id}/trash", json=stale)).status_code == 409
    assert observed == []
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash", json=_command(impact, "trash-events")
    )
    assert trashed.status_code == 200
    deletion_id = trashed.json()["deletion_id"]
    assert observed == [(deletion_id, "recoverable")]
    impact = await _impact(client, f"/api/recovery-items/{deletion_id}/impact")
    command = _command(impact, f"{action}-events")
    if action == "purge":
        command["acknowledgement"] = "permanently-delete"
    result = await client.post(f"/api/recovery-items/{deletion_id}/{action}", json=command)
    assert result.status_code == 200
    assert observed == [
        (deletion_id, "recoverable"),
        (deletion_id, None if action == "restore" else "purged"),
    ]
