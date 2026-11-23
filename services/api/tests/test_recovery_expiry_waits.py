"""A due item that keeps failing waits longer before each new try.

Each try takes SQLite's writer for as long as the item takes to check, so an
item that cannot be purged must not take it every minute for as long as it
stays stuck: other writers that wait past the busy timeout fail.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy.orm import Session
from test_recovery_expiry import DEADLINE, NOW, _trash

from local_lm import main, recovery_maintenance
from local_lm.chat_recovery import purge_chat
from local_lm.db import SessionLocal
from local_lm.models import Chat
from local_lm.recovery_maintenance import (
    EXPIRY_RETRY_FIRST_SECONDS,
    EXPIRY_RETRY_LONGEST_SECONDS,
    ExpiryBatch,
    ExpiryCursor,
    ExpiryWait,
)
from local_lm.recovery_previews import RecoveryPreviewConflict
from local_lm.recovery_v1 import PurgeRecoveryV1, RecoveryResultV1

FIRST = timedelta(seconds=EXPIRY_RETRY_FIRST_SECONDS)


@pytest.fixture(autouse=True)
def manual_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    async def dormant(*_args: object, **_kwargs: object) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(main, "maintain_recovery_expiry", dormant)


class _RefusedPurges:
    """Refuse the chat purges of chosen items, as a stale impact would, and count the tries."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *refused: str) -> None:
        self.refused = set(refused)
        self.tries: list[str] = []

        def purge(
            session: Session, deletion_id: str, command: PurgeRecoveryV1, now: datetime
        ) -> RecoveryResultV1:
            self.tries.append(deletion_id)
            if deletion_id in self.refused:
                raise RecoveryPreviewConflict("recovery-impact-stale")
            return purge_chat(session, deletion_id, command, now)

        monkeypatch.setattr(recovery_maintenance, "purge_chat", purge)


@pytest.mark.usefixtures("client")
async def test_an_item_that_keeps_failing_is_tried_again_only_after_a_wait_that_doubles(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    deletion_id, chat_id, _ = await _trash(app, "chat")
    purges = _RefusedPurges(monkeypatch, deletion_id)
    services = app.state.services
    waits: dict[str, ExpiryWait] = {}

    first = await recovery_maintenance.expire_recovery_batch(services, DEADLINE, waits=waits)
    assert (first.examined, first.deferred, first.waiting) == (1, 1, 0)
    assert waits[deletion_id] == ExpiryWait(1, DEADLINE + FIRST)

    early = await recovery_maintenance.expire_recovery_batch(
        services, DEADLINE + FIRST - timedelta(seconds=1), waits=waits
    )
    assert (early.examined, early.deferred, early.waiting) == (0, 0, 1)
    assert purges.tries == [deletion_id]

    second_at = DEADLINE + FIRST
    second = await recovery_maintenance.expire_recovery_batch(services, second_at, waits=waits)
    assert (second.examined, second.deferred, second.waiting) == (1, 1, 0)
    assert purges.tries == [deletion_id, deletion_id]
    assert waits[deletion_id] == ExpiryWait(2, second_at + 2 * FIRST)

    # Once the item can be purged, the next due try purges it and forgets the wait.
    purges.refused.clear()
    third = await recovery_maintenance.expire_recovery_batch(
        services, second_at + 2 * FIRST, waits=waits
    )
    assert (third.examined, third.purged, third.deferred) == (1, 1, 0)
    assert waits == {}
    with SessionLocal() as session:
        assert session.get(Chat, chat_id) is None


@pytest.mark.usefixtures("client")
async def test_the_wait_stops_growing_at_its_longest(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    deletion_id, _chat_id, _ = await _trash(app, "chat")
    _RefusedPurges(monkeypatch, deletion_id)
    waits = {deletion_id: ExpiryWait(20, DEADLINE)}

    result = await recovery_maintenance.expire_recovery_batch(
        app.state.services, DEADLINE, waits=waits
    )

    assert result.deferred == 1
    assert waits[deletion_id] == ExpiryWait(
        21, DEADLINE + timedelta(seconds=EXPIRY_RETRY_LONGEST_SECONDS)
    )


@pytest.mark.usefixtures("client")
async def test_a_waiting_item_does_not_hold_up_the_items_after_it(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    stuck_id, _stuck_chat, _ = await _trash(app, "chat", NOW - timedelta(days=1))
    next_id, next_chat, _ = await _trash(app, "chat")
    purges = _RefusedPurges(monkeypatch, stuck_id)
    services = app.state.services
    waits = {stuck_id: ExpiryWait(1, DEADLINE + timedelta(hours=1))}

    first = await recovery_maintenance.expire_recovery_batch(
        services, DEADLINE, limit=1, waits=waits
    )
    assert (first.examined, first.waiting) == (0, 1)
    assert first.next_cursor is not None and first.next_cursor.deletion_id == stuck_id
    second = await recovery_maintenance.expire_recovery_batch(
        services, DEADLINE, after=first.next_cursor, limit=1, waits=waits
    )

    assert (second.examined, second.purged) == (1, 1)
    assert purges.tries == [next_id]
    with SessionLocal() as session:
        assert session.get(Chat, next_chat) is None


@pytest.mark.usefixtures("client")
async def test_the_reason_an_item_waits_is_logged_without_anything_it_holds(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    deletion_id, _chat_id, _ = await _trash(app, "chat")
    _RefusedPurges(monkeypatch, deletion_id)

    with caplog.at_level(logging.INFO, logger="local_lm"):
        await recovery_maintenance.expire_recovery_batch(app.state.services, DEADLINE, waits={})

    messages = [record.getMessage() for record in caplog.records if record.name == "local_lm"]
    assert (
        f"Recovery expiry will try {deletion_id} again in "
        f"{EXPIRY_RETRY_FIRST_SECONDS:.0f} s (recovery-impact-stale)"
    ) in messages
    assert not any("Garden" in message for message in messages)


async def test_the_expiry_loop_keeps_one_set_of_waits_across_its_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, ExpiryWait] | None] = []
    three = asyncio.Event()

    async def batch(
        _services: Any,
        _now: datetime,
        *,
        after: ExpiryCursor | None = None,
        limit: int = 10,
        waits: dict[str, ExpiryWait] | None = None,
    ) -> ExpiryBatch:
        seen.append(waits)
        if len(seen) >= 3:
            three.set()
        return ExpiryBatch(0, 0, 0, None)

    monkeypatch.setattr(recovery_maintenance, "expire_recovery_batch", batch)
    services: Any = object()
    loop = asyncio.create_task(
        recovery_maintenance.maintain_recovery_expiry(services, interval_seconds=0.01)
    )
    try:
        await asyncio.wait_for(three.wait(), 3)
    finally:
        loop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop
    assert seen[0] is not None
    assert all(waits is seen[0] for waits in seen)
