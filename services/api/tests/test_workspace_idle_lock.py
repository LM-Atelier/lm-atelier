"""The workspace locking itself after a quiet spell: what counts as use, and what does not."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.main import create_app
from local_lm.models import AppSetting
from local_lm.workspace_lock import WorkspaceLock, lock_when_idle
from local_lm.workspace_lock_policy import (
    SavedWorkspaceLock,
    WorkspaceLockSettingInvalid,
    read_policy,
)

PIN = "4826"


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _unlocked(clock: _Clock, idle_minutes: int | None = 5) -> WorkspaceLock:
    lock = WorkspaceLock(clock=clock)
    lock.settle(SavedWorkspaceLock(enabled=True, revision=1, idle_lock_minutes=idle_minutes))
    lock.unlock()
    return lock


async def test_an_unused_workspace_locks_once_its_spell_runs_out() -> None:
    clock = _Clock()
    lock = _unlocked(clock)
    epoch = lock.status().lock_epoch
    signal = lock.lock_signal()

    clock.now += 299
    assert lock.lock_if_idle() is False
    assert lock.status().locked is False

    clock.now += 1
    assert lock.lock_if_idle() is True

    state = lock.status()
    assert state.locked is True
    # A lock like any other: a new epoch, and live connections told to close.
    assert state.lock_epoch != epoch
    assert signal.is_set()
    assert lock.lock_if_idle() is False


async def test_only_a_reported_use_starts_the_spell_again() -> None:
    clock = _Clock()
    lock = _unlocked(clock)

    clock.now += 200
    lock.note_use()
    clock.now += 299
    # Reading the status, and the requests a page makes on its own, are not use.
    lock.status()
    assert lock.refusal_code(None) is None
    assert lock.lock_if_idle() is False

    clock.now += 1
    assert lock.lock_if_idle() is True


async def test_a_use_reported_while_locked_changes_nothing() -> None:
    clock = _Clock()
    lock = _unlocked(clock)
    lock.lock()

    clock.now += 1_000
    lock.note_use()
    lock.unlock()
    # Unlocking is a use, so the spell starts from it, not from the report.
    clock.now += 299
    assert lock.lock_if_idle() is False
    clock.now += 1
    assert lock.lock_if_idle() is True


@pytest.mark.parametrize(
    "saved",
    [
        SavedWorkspaceLock(enabled=True, revision=1, idle_lock_minutes=None),
        SavedWorkspaceLock(enabled=False, revision=1, idle_lock_minutes=5),
        None,
    ],
    ids=["no spell chosen", "lock off", "setting unreadable"],
)
async def test_nothing_locks_for_being_unused_without_a_spell_and_the_lock_on(
    saved: SavedWorkspaceLock | None,
) -> None:
    clock = _Clock()
    lock = WorkspaceLock(clock=clock)
    lock.settle(saved)
    if lock.status().locked:
        lock.unlock()

    clock.now += 10 * 24 * 3600

    assert lock.lock_if_idle() is False
    assert lock.status().idle_lock_seconds is None


async def test_a_new_setting_takes_effect_at_once_and_counts_as_use() -> None:
    clock = _Clock()
    lock = _unlocked(clock, idle_minutes=None)

    clock.now += 3_600
    lock.note_policy(enabled=True, require_pin=False, idle_lock_minutes=1)
    assert lock.status().idle_lock_seconds == 60
    clock.now += 59
    assert lock.lock_if_idle() is False
    clock.now += 1
    assert lock.lock_if_idle() is True


async def test_the_service_locks_by_itself_once_the_spell_runs_out() -> None:
    clock = _Clock()
    lock = _unlocked(clock)
    watcher = asyncio.create_task(lock_when_idle(lock, every=0.01))
    try:
        await asyncio.sleep(0.05)
        assert lock.status().locked is False
        clock.now += 300
        for _ in range(200):
            if lock.status().locked:
                break
            await asyncio.sleep(0.01)
        assert lock.status().locked is True
    finally:
        watcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watcher


def test_a_setting_saved_before_spells_existed_still_reads(settings: Settings) -> None:
    with SessionLocal() as session:
        session.add(
            AppSetting(
                key="workspace_lock", value_json={"enabled": True, "revision": 1, "pin": None}
            )
        )
        session.commit()
        assert read_policy(session).idle_lock_minutes is None


@pytest.mark.parametrize("minutes", [0, 24 * 60 + 1, "5", True])
def test_a_saved_spell_outside_its_bounds_makes_the_setting_unreadable(
    settings: Settings, minutes: object
) -> None:
    document: dict[str, Any] = {
        "enabled": True,
        "revision": 1,
        "pin": None,
        "idle_lock_minutes": minutes,
    }
    with SessionLocal() as session:
        session.add(AppSetting(key="workspace_lock", value_json=document))
        session.commit()
        with pytest.raises(WorkspaceLockSettingInvalid):
            read_policy(session)


@asynccontextmanager
async def _open(settings: Settings) -> AsyncIterator[tuple[FastAPI, AsyncClient]]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            session = await client.post("/api/session")
            client.headers["x-local-lm-csrf"] = session.json()["csrf_token"]
            yield app, client


@pytest.fixture
def quick_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    class QuickArgon2id:
        def __init__(
            self, *, salt: bytes, length: int, iterations: int, lanes: int, memory_cost: int
        ) -> None:
            self._salt = salt
            self._length = length

        def derive(self, key_material: bytes) -> bytes:
            return hashlib.sha256(self._salt + key_material).digest()[: self._length]

    monkeypatch.setattr("local_lm.workspace_lock_policy.Argon2id", QuickArgon2id)


async def test_a_spell_is_chosen_kept_and_reported(settings: Settings) -> None:
    async with _open(settings) as (_app, client):
        chosen = await client.put(
            "/api/privacy/policy",
            json={"expected_revision": 0, "enabled": True, "idle_lock_minutes": 15},
        )
        assert chosen.status_code == 200, chosen.text
        assert chosen.json()["idle_lock_minutes"] == 15
        assert (await client.get("/api/privacy/status")).json()["idle_lock_seconds"] == 15 * 60

        # A change that does not name the spell keeps it.
        kept = await client.put(
            "/api/privacy/policy", json={"expected_revision": 1, "enabled": True}
        )
        assert kept.json()["idle_lock_minutes"] == 15

        never = await client.put(
            "/api/privacy/policy",
            json={"expected_revision": 2, "enabled": True, "idle_lock_minutes": None},
        )
        assert never.json()["idle_lock_minutes"] is None
        with SessionLocal() as session:
            stored = session.get(AppSetting, "workspace_lock")
            assert stored is not None
            # Without a spell the saved document is the one earlier versions read.
            assert set(stored.value_json) == {"enabled", "revision", "pin"}


async def test_a_spell_outside_its_bounds_is_refused(settings: Settings) -> None:
    async with _open(settings) as (_app, client):
        for minutes in (0, 24 * 60 + 1):
            refused = await client.put(
                "/api/privacy/policy",
                json={"expected_revision": 0, "enabled": True, "idle_lock_minutes": minutes},
            )
            assert refused.status_code == 422
        assert (await client.get("/api/privacy/policy")).json()["revision"] == 0


async def test_with_a_pin_only_staying_open_longer_asks_for_it(
    settings: Settings, quick_pins: None
) -> None:
    async with _open(settings) as (_app, client):
        first = await client.put(
            "/api/privacy/policy",
            json={"expected_revision": 0, "enabled": True, "new_pin": PIN, "idle_lock_minutes": 15},
        )
        assert first.status_code == 200, first.text

        shorter = await client.put(
            "/api/privacy/policy",
            json={"expected_revision": 1, "enabled": True, "idle_lock_minutes": 5},
        )
        assert (shorter.status_code, shorter.json()["idle_lock_minutes"]) == (200, 5)

        for longer in (30, None):
            refused = await client.put(
                "/api/privacy/policy",
                json={"expected_revision": 2, "enabled": True, "idle_lock_minutes": longer},
            )
            assert (refused.status_code, refused.json()["code"]) == (403, "workspace-pin-refused")
        assert (await client.get("/api/privacy/policy")).json()["idle_lock_minutes"] == 5

        allowed = await client.put(
            "/api/privacy/policy",
            json={
                "expected_revision": 2,
                "enabled": True,
                "idle_lock_minutes": None,
                "current_pin": PIN,
            },
        )
        assert (allowed.status_code, allowed.json()["idle_lock_minutes"]) == (200, None)


async def test_a_use_is_reported_only_while_unlocked(settings: Settings) -> None:
    async with _open(settings) as (app, client):
        clock = _Clock()
        lock = app.state.services.workspace_lock
        lock._clock = clock
        await client.put(
            "/api/privacy/policy",
            json={"expected_revision": 0, "enabled": True, "idle_lock_minutes": 1},
        )

        clock.now += 50
        assert (await client.post("/api/privacy/activity")).status_code == 204
        clock.now += 50
        # Ordinary requests are not use, so only the report above holds the lock off.
        assert (await client.get("/api/chats")).status_code == 200
        assert lock.lock_if_idle() is False
        clock.now += 10
        assert lock.lock_if_idle() is True

        refused = await client.post("/api/privacy/activity")
        assert (refused.status_code, refused.json()["code"]) == (423, "workspace-locked")
        assert lock.status().locked is True


async def test_the_running_service_locks_by_itself(
    settings: Settings,
) -> None:
    async with _open(settings) as (app, client):
        clock = _Clock()
        lock = app.state.services.workspace_lock
        lock._clock = clock
        chosen = await client.put(
            "/api/privacy/policy",
            json={"expected_revision": 0, "enabled": True, "idle_lock_minutes": 1},
        )
        assert chosen.status_code == 200, chosen.text

        clock.now += 60
        for _ in range(60):
            status = (await client.get("/api/privacy/status")).json()
            if status["locked"]:
                break
            await asyncio.sleep(0.1)

        # Locked by the service's own watch, not by any request this test made.
        assert status["locked"] is True
        refused = await client.get("/api/chats")
        assert (refused.status_code, refused.json()["code"]) == (423, "workspace-locked")
