"""The workspace lock's state, its pacing of PIN checks, and the gate in front of the API."""

from __future__ import annotations

import json

import pytest
from starlette.types import Message, Receive, Scope, Send

from local_lm.workspace_lock import (
    LOCK_EPOCH_HEADER,
    UnlockAttempts,
    WorkspaceLock,
    WorkspaceLockChanged,
    WorkspaceLockDisabled,
    WorkspaceLockMiddleware,
    WorkspaceLockState,
    wait_after,
)
from local_lm.workspace_lock_policy import PinVerifier, SavedWorkspaceLock

ON = SavedWorkspaceLock(enabled=True, revision=1, pin=None)
OFF = SavedWorkspaceLock(enabled=False, revision=1, pin=None)
LOCKED = {"detail": "LM Atelier is locked.", "code": "workspace-locked"}
CHANGED = {
    "detail": "The workspace lock changed. Reload to continue.",
    "code": "workspace-lock-changed",
}


def _protected() -> SavedWorkspaceLock:
    verifier = PinVerifier(
        algorithm="argon2id-v19",
        iterations=1,
        lanes=1,
        memory_kib=65536,
        salt="ab" * 16,
        hash="cd" * 32,
    )
    return SavedWorkspaceLock(enabled=True, revision=1, pin=verifier)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def test_five_wrong_pins_are_free_and_each_one_after_doubles_the_wait() -> None:
    assert [wait_after(count) for count in range(1, 6)] == [0, 0, 0, 0, 1]
    assert [wait_after(count) for count in range(6, 11)] == [2, 4, 8, 16, 32]
    assert [wait_after(count) for count in range(11, 16)] == [64, 128, 256, 512, 900]
    assert wait_after(100_000) == 900


def test_a_paced_refusal_neither_counts_nor_lengthens_the_wait() -> None:
    clock = _Clock()
    attempts = UnlockAttempts(clock=clock)
    waits = []
    for _ in range(5):
        assert attempts.admit() is None
        waits.append(attempts.fail())
        attempts.release()
    assert waits == [0, 0, 0, 0, 1]

    assert attempts.admit() == 1
    clock.now += 0.5
    assert attempts.admit() == 1
    assert attempts.consecutive_failures == 5

    clock.now += 0.5
    assert attempts.admit() is None
    assert attempts.fail() == 2
    attempts.release()
    clock.now += 1.5
    assert attempts.admit() == 1
    clock.now += 0.5
    assert attempts.admit() is None


def test_only_one_check_runs_at_a_time() -> None:
    attempts = UnlockAttempts(clock=_Clock())

    assert attempts.admit() is None
    assert attempts.admit() == 1
    assert attempts.consecutive_failures == 0
    attempts.release()
    assert attempts.admit() is None


def test_the_right_pin_clears_the_count_and_the_wait() -> None:
    attempts = UnlockAttempts(clock=_Clock())
    for _ in range(5):
        assert attempts.admit() is None
        attempts.fail()
        attempts.release()
    assert attempts.admit() == 1

    attempts.succeed()

    assert attempts.admit() is None
    assert attempts.fail() == 0


async def test_until_startup_settles_it_the_workspace_is_locked() -> None:
    lock = WorkspaceLock()

    state = lock.status()
    assert (state.locked, state.enabled, state.require_pin) == (True, True, True)
    assert isinstance(state.lock_epoch, str) and state.lock_epoch
    assert lock.lock_signal().is_set()
    assert not lock.setting_readable


async def test_startup_follows_the_saved_setting() -> None:
    off = WorkspaceLock()
    off.settle(OFF)
    assert off.status() == WorkspaceLockState(
        locked=False, enabled=False, require_pin=False, lock_epoch=None
    )
    assert not off.lock_signal().is_set()
    assert off.setting_readable

    on = WorkspaceLock()
    on.settle(_protected())
    state = on.status()
    assert (state.locked, state.enabled, state.require_pin) == (True, True, True)
    assert isinstance(state.lock_epoch, str) and state.lock_epoch
    assert on.lock_signal().is_set()
    assert on.setting_readable

    unreadable = WorkspaceLock()
    unreadable.settle(None)
    state = unreadable.status()
    assert (state.locked, state.enabled, state.require_pin) == (True, True, True)
    assert unreadable.lock_signal().is_set()
    assert not unreadable.setting_readable


async def test_a_lock_mints_an_epoch_and_an_unlock_keeps_it() -> None:
    lock = WorkspaceLock()
    lock.settle(ON)
    locked_signal = lock.lock_signal()
    locked_epoch = lock.status().lock_epoch

    unlocked = lock.unlock()
    assert (unlocked.locked, unlocked.lock_epoch) == (False, locked_epoch)
    open_signal = lock.lock_signal()
    assert open_signal is not locked_signal
    assert locked_signal.is_set() and not open_signal.is_set()

    relocked = lock.lock()
    assert relocked.locked
    assert relocked.lock_epoch != locked_epoch
    assert open_signal.is_set()
    # Locking again changes nothing a page already holds.
    assert lock.lock() == relocked


async def test_locking_needs_the_lock_turned_on() -> None:
    lock = WorkspaceLock()
    lock.settle(OFF)

    with pytest.raises(WorkspaceLockDisabled):
        lock.lock()
    assert not lock.status().locked

    lock.note_policy(enabled=True, require_pin=False)
    assert lock.lock().locked


async def _gate(
    lock: WorkspaceLock,
    path: str,
    *,
    method: str = "GET",
    epoch: str | None = None,
    scope_type: str = "http",
) -> list[Message] | None:
    """What the gate sent, or None when it passed the request on untouched."""

    passed: list[Scope] = []

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        passed.append(scope)

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    headers = [(b"host", b"testserver")]
    if epoch is not None:
        headers.append((LOCK_EPOCH_HEADER.encode(), epoch.encode()))
    scope = {
        "type": scope_type,
        "method": method,
        "path": path,
        "headers": headers,
        "query_string": b"",
    }
    middleware = WorkspaceLockMiddleware(
        downstream, lock=lock, public_paths=frozenset({"/api/session", "/api/ready"})
    )
    await middleware(scope, receive, send)
    if passed:
        assert sent == []
        return None
    return sent


def _refusal(sent: list[Message] | None) -> tuple[int, dict[str, str], dict[bytes, bytes]]:
    assert sent is not None, "the gate let the request through"
    start, body = sent
    return start["status"], json.loads(body["body"]), dict(start["headers"])


async def test_a_locked_gate_refuses_every_api_request_but_the_few_a_locked_page_needs() -> None:
    lock = WorkspaceLock()

    assert await _gate(lock, "/api/session", method="POST") is None
    assert await _gate(lock, "/api/ready") is None
    assert await _gate(lock, "/api/privacy/status") is None
    assert await _gate(lock, "/api/privacy/unlock", method="POST") is None
    assert await _gate(lock, "/", method="GET") is None
    assert await _gate(lock, "/assets/index.js") is None
    assert await _gate(lock, "/api/events", scope_type="websocket") is None

    for method, path in (
        ("GET", "/api/chats"),
        ("GET", "/api/privacy/unlock"),
        ("POST", "/api/privacy/status"),
        ("GET", "/api/privacy/status/"),
        ("POST", "/api/privacy/lock"),
        ("PUT", "/api/privacy/policy"),
        ("GET", "/api/unknown"),
        ("GET", "/apiary"),
    ):
        status, body, headers = _refusal(await _gate(lock, path, method=method))
        assert status == 423, f"{method} {path}"
        assert body == LOCKED
        assert headers[b"cache-control"] == b"no-store"


async def test_an_exempt_name_cannot_carry_a_different_path() -> None:
    # The session check reads the path through a URL, which drops this
    # fragment; the router matches the raw path, which keeps it.
    status, _, _ = _refusal(await _gate(WorkspaceLock(), "/api/session#/chats", method="POST"))

    assert status == 423


async def test_an_epoch_from_before_the_last_lock_is_refused_only_while_the_lock_is_on() -> None:
    lock = WorkspaceLock()
    lock.settle(ON)
    current = lock.unlock().lock_epoch
    assert current is not None

    assert await _gate(lock, "/api/chats", epoch=current) is None
    assert await _gate(lock, "/api/chats") is None
    status, body, _ = _refusal(await _gate(lock, "/api/chats", epoch="an-older-epoch"))
    assert (status, body) == (423, CHANGED)
    assert await _gate(lock, "/api/privacy/status", epoch="an-older-epoch") is None
    assert await _gate(lock, "/api/session", method="POST", epoch="an-older-epoch") is None

    lock.note_policy(enabled=False, require_pin=False)
    assert lock.status().lock_epoch is None
    assert await _gate(lock, "/api/chats", epoch="an-older-epoch") is None


async def test_an_unlock_decided_for_an_earlier_lock_opens_nothing() -> None:
    lock = WorkspaceLock()
    lock.settle(ON)
    admitted = lock.status().lock_epoch
    lock.unlock(admitted_epoch=admitted)
    newer = lock.lock().lock_epoch

    with pytest.raises(WorkspaceLockChanged):
        lock.unlock(admitted_epoch=admitted)

    assert lock.status().locked
    assert lock.status().lock_epoch == newer
    assert lock.unlock(admitted_epoch=newer).locked is False


async def _answer(lock: WorkspaceLock, path: str) -> dict[bytes, bytes]:
    """The headers an admitted request's response leaves the gate with."""

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"cache-control", b"private, max-age=31536000, immutable")],
            }
        )
        await send({"type": "http.response.body", "body": b"neutral"})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [(b"host", b"testserver")],
        "query_string": b"",
    }
    middleware = WorkspaceLockMiddleware(
        downstream, lock=lock, public_paths=frozenset({"/api/session"})
    )
    await middleware(scope, receive, send)
    return dict(sent[0]["headers"])


async def test_responses_are_not_kept_while_the_lock_is_on() -> None:
    lock = WorkspaceLock()
    lock.settle(OFF)
    kept = await _answer(lock, "/api/artifacts/neutral/content")
    lock.settle(ON)
    lock.unlock()

    assert kept[b"cache-control"] == b"private, max-age=31536000, immutable"
    assert (await _answer(lock, "/api/artifacts/neutral/content"))[b"cache-control"] == b"no-store"
    assert (await _answer(lock, "/api/session"))[b"cache-control"] == b"no-store"
    assert (await _answer(lock, "/assets/index.js"))[b"cache-control"] == (
        b"private, max-age=31536000, immutable"
    )
