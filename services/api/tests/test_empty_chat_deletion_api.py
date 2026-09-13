"""Previewing and deleting empty chats over HTTP.

The service tests cover the rules. These cover what only the surface can get
wrong: the typed refusals and their status codes, a payload that never carries
anything somebody wrote, and the chat lifecycle guard - held before the
selection is revalidated, and taken without cancelling anything.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy.orm import Session

from local_lm.db import SessionLocal
from local_lm.models import Chat, Message, MessageRole

pytestmark = pytest.mark.asyncio

PREVIEW = "/api/maintenance/empty-chats/preview"
EXECUTE = "/api/maintenance/empty-chats/execute"


def _chat(session: Session, **values: object) -> str:
    row = Chat(**{"created_at": datetime.now(UTC) - timedelta(hours=48), **values})
    session.add(row)
    session.flush()
    return row.id


def _exists(chat_id: str) -> bool:
    with SessionLocal() as session:
        return session.get(Chat, chat_id) is not None


async def _preview(
    client: AsyncClient, chat_ids: list[str], **filters: object
) -> dict[str, object]:
    response = await client.post(PREVIEW, json={"chat_ids": chat_ids, **filters})
    assert response.status_code == 200, response.text
    return response.json()


def _execute_body(
    preview: dict[str, object], chat_ids: list[str], operation_id: str, **overrides: object
) -> dict[str, object]:
    return {
        "operation_id": operation_id,
        "preview_id": preview["preview_id"],
        "digest": preview["digest"],
        "acknowledged_count": int(str(preview["strict_count"]))
        + int(str(preview["configured_count"])),
        "chat_ids": chat_ids,
        **overrides,
    }


async def test_preview_then_execute_deletes_once_and_a_retry_replays(client: AsyncClient) -> None:
    with SessionLocal() as session:
        first, second = _chat(session), _chat(session)
        session.commit()
    ids = [first, second]

    preview = await _preview(client, ids)
    assert (preview["strict_count"], preview["configured_count"], preview["conflicts"]) == (
        2,
        0,
        [],
    )
    assert str(preview["expires_at"]).endswith("Z")

    deleted = await client.post(EXECUTE, json=_execute_body(preview, ids, "op-http"))
    assert deleted.status_code == 200, deleted.text
    body = deleted.json()
    assert (sorted(body["deleted_ids"]), body["replayed"]) == (sorted(ids), False)
    assert not _exists(first) and not _exists(second)

    retried = await client.post(EXECUTE, json=_execute_body(preview, ids, "op-http"))
    assert retried.status_code == 200, retried.text
    assert retried.json() == {**body, "replayed": True}


async def test_a_drifted_selection_is_refused_with_a_typed_code_and_nothing_deleted(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        kept, changed = _chat(session), _chat(session)
        session.commit()
    ids = [kept, changed]
    preview = await _preview(client, ids)
    with SessionLocal() as session:
        row = session.get(Chat, changed)
        assert row is not None
        row.title = "A title somebody typed"
        session.commit()

    refused = await client.post(EXECUTE, json=_execute_body(preview, ids, "op-drift"))

    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "empty-chat-selection-drifted"
    assert refused.json()["chat_ids"] == [changed]
    assert "A title somebody typed" not in refused.text
    assert _exists(kept) and _exists(changed)


@pytest.mark.parametrize(
    ("override", "status", "code"),
    [
        ({"preview_id": "emptyprev_invented"}, 409, "empty-chat-preview-unknown"),
        ({"acknowledged_count": 5}, 422, "empty-chat-count-mismatch"),
    ],
)
async def test_every_refusal_carries_its_own_code_and_deletes_nothing(
    client: AsyncClient, override: dict[str, object], status: int, code: str
) -> None:
    with SessionLocal() as session:
        blank = _chat(session)
        session.commit()
    preview = await _preview(client, [blank])

    refused = await client.post(
        EXECUTE, json=_execute_body(preview, [blank], f"op-{code}", **override)
    )

    assert refused.status_code == status, refused.text
    assert refused.json()["code"] == code
    assert _exists(blank)


async def test_configured_chats_are_deleted_only_with_their_own_acknowledgement(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        titled = _chat(session, title="Somebody named this", draft_prompt="half a thought")
        session.commit()
    preview = await _preview(client, [titled], include_configured=True)
    assert preview["configured_count"] == 1

    refused = await client.post(
        EXECUTE, json=_execute_body(preview, [titled], "op-ack", include_configured=True)
    )
    assert refused.status_code == 422, refused.text
    assert refused.json()["code"] == "empty-chat-configured-not-acknowledged"
    assert _exists(titled)

    deleted = await client.post(
        EXECUTE,
        json=_execute_body(
            preview,
            [titled],
            "op-ack-confirmed",
            include_configured=True,
            acknowledged_configured=True,
        ),
    )
    assert deleted.status_code == 200, deleted.text
    for text in (preview, deleted.json()):
        assert "Somebody named this" not in str(text)
        assert "half a thought" not in str(text)
    assert not _exists(titled)


async def test_a_selection_is_bounded(client: AsyncClient) -> None:
    too_many = [f"chat_{index:04d}" for index in range(201)]

    for body in ({"chat_ids": []}, {"chat_ids": too_many}, {"chat_ids": ["c" * 41]}):
        response = await client.post(PREVIEW, json=body)
        assert response.status_code == 422, response.text


async def test_the_guard_is_held_before_the_selection_is_revalidated(
    app: FastAPI, client: AsyncClient
) -> None:
    """A turn that holds the chat when the deletion arrives is waited for, then seen.

    The deletion must not revalidate until it holds the guard. If it checked
    first and locked afterwards, the message written while the turn held the
    guard would arrive after a check that had already approved the delete.
    """

    with SessionLocal() as session:
        blank = _chat(session)
        session.commit()
    preview = await _preview(client, [blank])
    guard = app.state.services.orchestrator.chat_guard(blank)

    await guard.acquire()
    try:
        pending = asyncio.create_task(
            client.post(EXECUTE, json=_execute_body(preview, [blank], "op-guarded"))
        )
        await asyncio.sleep(0.2)
        assert not pending.done(), "the deletion did not wait for the chat's guard"
        with SessionLocal() as session:
            session.add(Message(chat_id=blank, role=MessageRole.USER.value))
            session.commit()
    finally:
        guard.release()

    refused = await asyncio.wait_for(pending, timeout=10)
    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "empty-chat-selection-drifted"
    assert _exists(blank)


async def test_deleting_cancels_nothing_in_a_chat_that_gained_work(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Somebody started a turn after the preview; the refusal leaves it running.

    Deleting a single chat cancels its generations first, which is right when
    somebody asked for that chat to go. Here nobody did: the chat was chosen
    because it was empty, and it is not any more.
    """

    with SessionLocal() as session:
        blank = _chat(session)
        session.commit()
    preview = await _preview(client, [blank])
    with SessionLocal() as session:
        session.add(Message(chat_id=blank, role=MessageRole.USER.value))
        session.commit()
    orchestrator = app.state.services.orchestrator
    cancelled: list[str] = []

    async def record(chat_id: str) -> None:
        cancelled.append(chat_id)

    monkeypatch.setattr(orchestrator, "_cancel_chat_runs", record)

    refused = await client.post(EXECUTE, json=_execute_body(preview, [blank], "op-no-cancel"))

    assert refused.status_code == 409, refused.text
    assert cancelled == []
    assert _exists(blank)
