"""The workspace lock through the running application: what stays shut, and what opens it."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from run_waits import wait_for_terminal_status
from starlette.types import Message as ASGIMessage

from local_lm import workspace_lock_api
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.instance_identity import INSTANCE_ID_HEADER
from local_lm.main import create_app
from local_lm.models import AppSetting, Message, Run

LOCKED = {"detail": "LM Atelier is locked.", "code": "workspace-locked"}
CHANGED = {
    "detail": "The workspace lock changed. Reload to continue.",
    "code": "workspace-lock-changed",
}
EPOCH = "x-local-lm-lock-epoch"
PIN = "4826"
WRONG = "1357"
VERIFIER_FIELDS = {"algorithm", "iterations", "lanes", "memory_kib", "salt", "hash"}
UNREADABLE = {"enabled": False, "revision": 1}


@asynccontextmanager
async def _open(settings: Settings) -> AsyncIterator[tuple[FastAPI, AsyncClient]]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            session = await client.post("/api/session")
            assert session.status_code == 200
            client.headers["x-local-lm-csrf"] = session.json()["csrf_token"]
            yield app, client


@dataclass
class _Derivations:
    """A quick stand-in for Argon2id that counts, and can hold or fail on request."""

    count: int = 0
    started: threading.Event = field(default_factory=threading.Event)
    hold: threading.Event | None = None
    error: BaseException | None = None


@pytest.fixture
def derivations(monkeypatch: pytest.MonkeyPatch) -> _Derivations:
    record = _Derivations()

    class QuickArgon2id:
        def __init__(
            self, *, salt: bytes, length: int, iterations: int, lanes: int, memory_cost: int
        ) -> None:
            self._salt = salt
            self._length = length

        def derive(self, key_material: bytes) -> bytes:
            record.count += 1
            record.started.set()
            if record.hold is not None:
                record.hold.wait(timeout=30)
            if record.error is not None:
                raise record.error
            return hashlib.sha256(self._salt + key_material).digest()[: self._length]

    monkeypatch.setattr("local_lm.workspace_lock_policy.Argon2id", QuickArgon2id)
    return record


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _paced(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Put the PIN pacing on a clock the test moves by hand."""

    clock = _Clock()
    monkeypatch.setattr(app.state.services.workspace_lock.attempts, "clock", clock)
    return clock


async def _choose(client: AsyncClient, revision: int, **change: Any) -> Any:
    body = {"expected_revision": revision, "enabled": True, **change}
    return await client.put("/api/privacy/policy", json=body)


async def _lock(client: AsyncClient) -> str:
    response = await client.post("/api/privacy/lock")
    assert response.status_code == 200
    assert response.json()["locked"] is True
    epoch = response.json()["lock_epoch"]
    assert isinstance(epoch, str) and epoch
    return epoch


async def _unlock(client: AsyncClient, pin: str | None = None) -> Any:
    return await client.post("/api/privacy/unlock", json={} if pin is None else {"pin": pin})


async def test_the_lock_starts_off_and_every_change_names_the_revision_it_read(
    client: AsyncClient,
) -> None:
    assert (await client.get("/api/privacy/policy")).json() == {
        "enabled": False,
        "require_pin": False,
        "revision": 0,
        "idle_lock_minutes": None,
    }
    assert (await client.get("/api/privacy/status")).json() == {
        "locked": False,
        "enabled": False,
        "require_pin": False,
        "lock_epoch": None,
        "idle_lock_seconds": None,
    }
    session = (await client.post("/api/session")).json()
    assert (session["workspace_locked"], session["lock_epoch"]) == (False, None)

    turned_on = await _choose(client, 0)
    assert turned_on.status_code == 200
    assert turned_on.json() == {
        "enabled": True,
        "require_pin": False,
        "revision": 1,
        "idle_lock_minutes": None,
    }

    late = await _choose(client, 0, enabled=False)
    assert late.status_code == 409
    assert late.json()["code"] == "workspace-lock-setting-stale"
    assert late.json()["current_revision"] == 1

    status = (await client.get("/api/privacy/status")).json()
    assert (status["locked"], status["enabled"], status["require_pin"]) == (False, True, False)
    assert isinstance(status["lock_epoch"], str) and status["lock_epoch"]


async def test_a_pin_is_kept_only_as_a_verifier(
    client: AsyncClient, derivations: _Derivations
) -> None:
    chosen = await _choose(client, 0, new_pin=PIN)

    assert chosen.json() == {
        "enabled": True,
        "require_pin": True,
        "revision": 1,
        "idle_lock_minutes": None,
    }
    assert (await client.get("/api/privacy/policy")).json() == chosen.json()
    status = (await client.get("/api/privacy/status")).json()
    assert set(status) == {"locked", "enabled", "require_pin", "lock_epoch", "idle_lock_seconds"}
    assert status["require_pin"] is True
    with SessionLocal() as session:
        stored = session.get(AppSetting, "workspace_lock")
        assert stored is not None
        document = stored.value_json
    assert set(document) == {"enabled", "revision", "pin"}
    assert set(document["pin"]) == VERIFIER_FIELDS
    assert PIN not in str(document)


async def test_a_saved_setting_that_cannot_be_read_is_refused_without_echoing_it(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        session.add(AppSetting(key="workspace_lock", value_json={"enabled": "sometimes"}))
        session.commit()

    read = await client.get("/api/privacy/policy")
    write = await _choose(client, 0)

    for response in (read, write):
        assert response.status_code == 409
        assert response.json() == {
            "detail": "The saved workspace lock setting cannot be read.",
            "code": "workspace-lock-setting-invalid",
        }


async def test_locking_needs_the_lock_turned_on(client: AsyncClient) -> None:
    refused = await client.post("/api/privacy/lock")

    assert refused.status_code == 409
    assert refused.json()["code"] == "workspace-lock-disabled"
    assert (await client.get("/api/chats")).status_code == 200


async def test_while_locked_the_api_answers_only_what_a_locked_page_needs(
    client: AsyncClient,
) -> None:
    assert (await _choose(client, 0)).status_code == 200
    await _lock(client)

    for method, path in (
        ("GET", "/api/chats"),
        ("POST", "/api/projects"),
        ("GET", "/api/x"),
        ("GET", "/api/privacy/status/"),
        ("GET", "/api/privacy/unlock"),
        ("GET", "/api/privacy/policy"),
        ("PUT", "/api/privacy/policy"),
        ("POST", "/api/privacy/lock"),
        ("GET", "/api/artifacts/art_missing/content"),
        ("GET", "/api/queue/activity"),
        ("GET", "/api/jobs/activity"),
    ):
        body = None if method == "GET" else {"name": "Refused"}
        response = await client.request(method, path, json=body)
        assert (response.status_code, response.json()) == (423, LOCKED), f"{method} {path}"
        assert response.headers["cache-control"] == "no-store", f"{method} {path}"
        assert response.headers["x-frame-options"] == "DENY", f"{method} {path}"

    session = await client.post("/api/session")
    assert session.status_code == 200
    assert session.json()["workspace_locked"] is True
    assert (await client.get("/api/health")).status_code == 200
    ready = await client.get("/api/ready")
    assert ready.status_code == 200
    assert ready.headers[INSTANCE_ID_HEADER]
    status = await client.get("/api/privacy/status")
    assert status.status_code == 200
    assert status.json()["locked"] is True

    unlocked = await client.post("/api/privacy/unlock")
    assert unlocked.status_code == 200
    assert unlocked.json()["locked"] is False
    assert (await client.get("/api/chats")).status_code == 200


async def test_a_caller_without_a_session_hears_that_before_hearing_of_the_lock(
    settings: Settings,
) -> None:
    outside_development = Settings(
        data_dir=settings.data_dir, dev=False, chat_engine="mock", media_engine="mock"
    )
    app = create_app(outside_development)
    project = {"name": "Refused"}
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            csrf = (await client.post("/api/session")).json()["csrf_token"]
            with_csrf = {"x-local-lm-csrf": csrf}
            policy = await client.put(
                "/api/privacy/policy",
                json={"expected_revision": 0, "enabled": True},
                headers=with_csrf,
            )
            assert policy.status_code == 200
            assert (await client.post("/api/privacy/lock", headers=with_csrf)).status_code == 200

            no_csrf = await client.post("/api/projects", json=project)
            assert no_csrf.status_code == 403
            assert no_csrf.json()["code"] == "csrf-invalid"

            elsewhere = await client.get(
                "/api/chats", headers={"origin": "https://elsewhere.example"}
            )
            assert elsewhere.status_code == 403
            assert elsewhere.json()["code"] == "origin-untrusted"

            locked = await client.post("/api/projects", json=project, headers=with_csrf)
            assert (locked.status_code, locked.json()) == (423, LOCKED)
            assert (await client.get("/api/chats")).status_code == 423

        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as stranger:
            no_session = await stranger.get("/api/chats")
            assert no_session.status_code == 401
            assert no_session.json()["code"] == "session-required"


async def test_wrong_pins_are_paced_and_the_right_one_opens_the_workspace(
    app: FastAPI,
    client: AsyncClient,
    derivations: _Derivations,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _paced(app, monkeypatch)
    assert (await _choose(client, 0, new_pin=PIN)).status_code == 200
    await _lock(client)

    # A wrong PIN, no PIN, and text that could never be the PIN each count once.
    for attempt in ({"pin": WRONG}, {}, {"pin": "482"}, {"pin": "4" * 65}):
        refused = await client.post("/api/privacy/unlock", json=attempt)
        assert refused.status_code == 403, attempt
        assert refused.json() == {
            "detail": "The workspace could not be unlocked.",
            "code": "workspace-pin-refused",
            "retry_after_seconds": 0,
        }
        assert "retry-after" not in refused.headers

    fifth = await _unlock(client, WRONG)
    assert fifth.status_code == 403
    assert fifth.json()["retry_after_seconds"] == 1
    assert fifth.headers["retry-after"] == "1"

    derived = derivations.count
    early = await _unlock(client, PIN)
    assert early.status_code == 429
    assert early.json()["code"] == "workspace-pin-throttled"
    assert early.json()["retry_after_seconds"] == 1
    assert early.headers["retry-after"] == "1"
    assert derivations.count == derived
    assert (await client.get("/api/chats")).status_code == 423

    clock.now += 1
    opened = await _unlock(client, PIN)
    assert opened.status_code == 200
    assert opened.json()["locked"] is False
    assert (await client.get("/api/chats")).status_code == 200

    # The right PIN cleared the count, so the next wrong one is the first again.
    await _lock(client)
    assert (await _unlock(client, WRONG)).json()["retry_after_seconds"] == 0


async def test_only_one_pin_check_runs_at_a_time(
    client: AsyncClient, derivations: _Derivations
) -> None:
    assert (await _choose(client, 0, new_pin=PIN)).status_code == 200
    await _lock(client)
    derivations.started.clear()
    derivations.hold = threading.Event()

    first = asyncio.create_task(_unlock(client, PIN))
    try:
        assert await asyncio.to_thread(derivations.started.wait, 10)
        second = await _unlock(client, PIN)
        assert second.status_code == 429
        assert second.json()["retry_after_seconds"] == 1
    finally:
        derivations.hold.set()
    assert (await first).status_code == 200


async def test_neither_an_unavailable_check_nor_an_unread_request_counts_as_a_wrong_pin(
    client: AsyncClient, derivations: _Derivations
) -> None:
    assert (await _choose(client, 0, new_pin=PIN)).status_code == 200
    await _lock(client)

    derivations.error = MemoryError()
    unavailable = await _unlock(client, PIN)
    assert unavailable.status_code == 503
    assert unavailable.json()["code"] == "workspace-pin-unavailable"
    derivations.error = None

    too_long = await _unlock(client, "4" * 257)
    assert too_long.status_code == 422
    assert too_long.json()["code"] == "request-validation-invalid"

    waits = [(await _unlock(client, WRONG)).json()["retry_after_seconds"] for _ in range(5)]
    assert waits == [0, 0, 0, 0, 1]


async def test_a_set_pin_guards_every_change_that_would_weaken_it(
    app: FastAPI,
    client: AsyncClient,
    derivations: _Derivations,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _paced(app, monkeypatch)
    protected = await _choose(client, 0, new_pin=PIN)
    assert protected.json() == {
        "enabled": True,
        "require_pin": True,
        "revision": 1,
        "idle_lock_minutes": None,
    }

    for change in (
        {"new_pin": "8642"},
        {"clear_pin": True, "current_pin": WRONG},
        {"enabled": False, "current_pin": WRONG},
    ):
        refused = await _choose(client, 1, **change)
        assert refused.status_code == 403, change
        assert refused.json()["code"] == "workspace-pin-refused", change

    # Those three count against the same pacing as unlocking.
    await _lock(client)
    assert (await _unlock(client, WRONG)).json()["retry_after_seconds"] == 0
    assert (await _unlock(client, WRONG)).json()["retry_after_seconds"] == 1
    clock.now += 1
    assert (await _unlock(client, PIN)).status_code == 200

    for pin in ("482", "4" * 65):
        invalid = await _choose(client, 1, new_pin=pin, current_pin=PIN)
        assert invalid.status_code == 422
        assert invalid.json()["code"] == "workspace-pin-invalid"
    both = await _choose(client, 1, new_pin="8642", clear_pin=True, current_pin=PIN)
    assert both.status_code == 422
    assert both.json()["code"] == "request-validation-invalid"

    changed = await _choose(client, 1, new_pin="8642", current_pin=PIN)
    assert changed.json() == {
        "enabled": True,
        "require_pin": True,
        "revision": 2,
        "idle_lock_minutes": None,
    }
    assert (await _choose(client, 2, clear_pin=True, current_pin=PIN)).status_code == 403
    removed = await _choose(client, 2, clear_pin=True, current_pin="8642")
    assert removed.json() == {
        "enabled": True,
        "require_pin": False,
        "revision": 3,
        "idle_lock_minutes": None,
    }
    assert (await _choose(client, 3, new_pin=PIN)).json()["require_pin"] is True
    turned_off = await _choose(client, 4, enabled=False, current_pin=PIN)
    assert turned_off.json() == {
        "enabled": False,
        "require_pin": True,
        "revision": 5,
        "idle_lock_minutes": None,
    }


async def test_a_request_from_before_the_latest_lock_is_refused_as_a_change(
    client: AsyncClient,
) -> None:
    assert (await _choose(client, 0)).status_code == 200
    before = (await client.post("/api/session")).json()["lock_epoch"]
    assert isinstance(before, str) and before

    current = await _lock(client)
    assert current != before
    unlocked = await _unlock(client)
    assert unlocked.json()["lock_epoch"] == current

    stale = await client.get("/api/chats", headers={EPOCH: before})
    assert (stale.status_code, stale.json()) == (423, CHANGED)
    assert stale.headers["cache-control"] == "no-store"
    assert (await client.get("/api/chats", headers={EPOCH: current})).status_code == 200
    assert (await client.get("/api/chats")).status_code == 200
    # A page holding the older epoch can still learn where things stand.
    assert (await client.get("/api/privacy/status", headers={EPOCH: before})).status_code == 200
    assert (await client.post("/api/session", headers={EPOCH: before})).status_code == 200

    assert (await _choose(client, 1, enabled=False)).status_code == 200
    assert (await client.get("/api/chats", headers={EPOCH: before})).status_code == 200
    session = (await client.post("/api/session")).json()
    assert (session["workspace_locked"], session["lock_epoch"]) == (False, None)


async def test_work_accepted_before_a_lock_finishes_and_is_there_after_unlocking(
    client: AsyncClient,
) -> None:
    assert (await _choose(client, 0)).status_code == 200
    chat = (await client.post("/api/chats", json={"title": "Lock while working"})).json()
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Explain local inference", "mode": "auto", "idempotency_key": "lock-1"},
    )
    assert accepted.status_code == 202
    run_id = accepted.json()["run"]["id"]
    reply_id = accepted.json()["assistant_message"]["id"]
    await _lock(client)
    assert (await client.get(f"/api/runs/{run_id}")).status_code == 423

    # The API is shut, so the wait reads the database directly.
    def stored_status(model: type[Run] | type[Message], key: str) -> dict[str, str] | None:
        with SessionLocal() as session:
            row = session.get(Run, key) if model is Run else session.get(Message, key)
            return None if row is None else {"status": str(row.status)}

    async def read_run() -> dict[str, str] | None:
        return stored_status(Run, run_id)

    async def read_reply() -> dict[str, str] | None:
        return stored_status(Message, reply_id)

    await wait_for_terminal_status(read_run, what=f"run {run_id} while locked")
    await wait_for_terminal_status(read_reply, what=f"reply {reply_id} while locked")
    assert (await _unlock(client)).status_code == 200

    transcript = (await client.get(f"/api/chats/{chat['id']}")).json()
    replies = [message for message in transcript["messages"] if message["role"] == "assistant"]
    assert [(reply["id"], reply["status"]) for reply in replies] == [(reply_id, "complete")]
    assert "Mock local response" in replies[0]["parts"][0]["text"]


async def test_a_lock_that_is_on_is_locked_again_after_a_restart(settings: Settings) -> None:
    async with _open(settings) as (_, client):
        assert (await _choose(client, 0)).status_code == 200

    async with _open(settings) as (_, client):
        assert (await client.get("/api/chats")).status_code == 423
        status = (await client.get("/api/privacy/status")).json()
        assert (status["locked"], status["enabled"], status["require_pin"]) == (True, True, False)
        assert (await _unlock(client)).status_code == 200
        assert (await _choose(client, 1, enabled=False)).status_code == 200

    async with _open(settings) as (_, client):
        assert (await client.get("/api/chats")).status_code == 200
        assert (await client.get("/api/privacy/policy")).json() == {
            "enabled": False,
            "require_pin": False,
            "revision": 2,
            "idle_lock_minutes": None,
        }


async def test_an_unreadable_setting_keeps_the_workspace_locked(settings: Settings) -> None:
    with SessionLocal() as session:
        session.add(AppSetting(key="workspace_lock", value_json=UNREADABLE))
        session.commit()

    async with _open(settings) as (_, client):
        status = (await client.get("/api/privacy/status")).json()
        assert (status["locked"], status["enabled"], status["require_pin"]) == (True, True, True)
        assert (await client.get("/api/chats")).status_code == 423
        refused = await _unlock(client, PIN)
        assert refused.status_code == 409
        assert refused.json()["code"] == "workspace-lock-setting-invalid"


async def test_the_reset_switch_clears_the_lock_at_startup(settings: Settings) -> None:
    with SessionLocal() as session:
        session.add(AppSetting(key="workspace_lock", value_json=UNREADABLE))
        session.commit()
    resetting = Settings(
        data_dir=settings.data_dir,
        dev=True,
        chat_engine="mock",
        media_engine="mock",
        reset_workspace_lock=True,
    )

    async with _open(resetting) as (_, client):
        assert (await client.get("/api/chats")).status_code == 200
        assert (await client.get("/api/privacy/status")).json()["locked"] is False

    with SessionLocal() as session:
        assert session.get(AppSetting, "workspace_lock") is None


async def test_nothing_is_served_before_startup_has_read_the_setting(app: FastAPI) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as raw:
        session = await raw.post("/api/session")
        assert session.status_code == 200
        assert session.json()["workspace_locked"] is True
        raw.headers["x-local-lm-csrf"] = session.json()["csrf_token"]

        chats = await raw.get("/api/chats")
        assert (chats.status_code, chats.json()) == (423, LOCKED)
        unlock = await raw.post("/api/privacy/unlock", json={})
        assert unlock.status_code == 409
        assert unlock.json()["code"] == "workspace-lock-setting-invalid"


async def test_the_page_still_loads_while_locked(settings: Settings, tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>LM Atelier</title>", encoding="utf-8")
    app = create_app(
        Settings(
            data_dir=settings.data_dir,
            web_dist_dir=dist,
            dev=True,
            chat_engine="mock",
            media_engine="mock",
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as raw:
        page = await raw.get("/")
        assert page.status_code == 200
        assert "LM Atelier" in page.text
        await raw.post("/api/session")
        assert (await raw.get("/api/chats")).status_code == 423


class _EventSocket:
    """One /api/events connection, driven through the application's own ASGI entry point."""

    def __init__(
        self,
        app: FastAPI,
        client: AsyncClient,
        *,
        query: bytes = b"after=0",
        headers: tuple[tuple[bytes, bytes], ...] = (),
    ) -> None:
        cookie = f"local_lm_session={client.cookies['local_lm_session']}".encode()
        self.inbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.outbound: asyncio.Queue[ASGIMessage] = asyncio.Queue()
        self.inbound.put_nowait({"type": "websocket.connect"})
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "scheme": "ws",
            "path": "/api/events",
            "raw_path": b"/api/events",
            "query_string": query,
            "root_path": "",
            "headers": [(b"host", b"testserver"), (b"cookie", cookie), *headers],
            "client": ("127.0.0.1", 123),
            "server": ("testserver", 80),
            "subprotocols": [],
        }
        self.task = asyncio.create_task(app(scope, self.inbound.get, self.outbound.put))

    async def next_sent(self) -> ASGIMessage:
        return await asyncio.wait_for(self.outbound.get(), timeout=10)

    async def ended(self) -> None:
        await asyncio.wait_for(self.task, timeout=10)

    async def leave(self) -> None:
        self.inbound.put_nowait({"type": "websocket.disconnect", "code": 1000})
        await self.ended()


async def test_a_locked_workspace_refuses_the_event_socket_before_accepting_it(
    app: FastAPI, client: AsyncClient
) -> None:
    assert (await _choose(client, 0)).status_code == 200
    await _lock(client)

    socket = _EventSocket(app, client)
    first = await socket.next_sent()
    await socket.ended()

    assert (first["type"], first["code"]) == ("websocket.close", 4423)
    assert socket.outbound.empty()


async def test_an_open_workspace_accepts_the_event_socket(
    app: FastAPI, client: AsyncClient
) -> None:
    assert (await _choose(client, 0)).status_code == 200

    socket = _EventSocket(app, client)
    first = await socket.next_sent()
    await socket.leave()

    assert first["type"] == "websocket.accept"


async def test_a_lock_closes_an_event_socket_that_is_already_open(
    app: FastAPI, client: AsyncClient
) -> None:
    assert (await _choose(client, 0)).status_code == 200
    socket = _EventSocket(app, client)
    assert (await socket.next_sent())["type"] == "websocket.accept"

    await _lock(client)
    # Events the stream was already sending may come first; nothing follows the close.
    closed = await socket.next_sent()
    while closed["type"] == "websocket.send":
        closed = await socket.next_sent()
    await socket.ended()

    assert closed == {"type": "websocket.close", "code": 4423, "reason": "workspace-locked"}


@pytest.mark.parametrize("carried", ["query", "header"])
async def test_an_event_socket_from_before_the_latest_lock_is_refused_before_accepting_it(
    app: FastAPI, client: AsyncClient, carried: str
) -> None:
    assert (await _choose(client, 0)).status_code == 200
    earlier = (await client.post("/api/session")).json()["lock_epoch"]
    current = await _lock(client)
    assert (await _unlock(client)).status_code == 200

    def socket(epoch: str) -> _EventSocket:
        if carried == "query":
            return _EventSocket(app, client, query=f"after=0&lock_epoch={epoch}".encode())
        return _EventSocket(app, client, headers=((EPOCH.encode(), epoch.encode()),))

    stale = socket(earlier)
    refused = await stale.next_sent()
    await stale.ended()
    fresh = socket(current)
    accepted = await fresh.next_sent()
    await fresh.leave()

    assert refused == {"type": "websocket.close", "code": 4423, "reason": ""}
    assert accepted["type"] == "websocket.accept"


async def test_a_pin_check_decided_for_an_earlier_lock_cannot_open_a_newer_one(
    app: FastAPI, client: AsyncClient, derivations: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (await _choose(client, 0, new_pin=PIN)).status_code == 200
    await _lock(client)
    check = workspace_lock_api._check_saved_pin
    decided = threading.Event()
    resume = threading.Event()

    def slow_to_answer(attempts: Any, pin: str | None) -> Any:
        # The check has finished and given up its slot; only its answer is late.
        answer = check(attempts, pin)
        if not decided.is_set():
            decided.set()
            assert resume.wait(timeout=15)
        return answer

    monkeypatch.setattr(workspace_lock_api, "_check_saved_pin", slow_to_answer)
    earlier = asyncio.create_task(_unlock(client, PIN))
    try:
        assert await asyncio.to_thread(decided.wait, 10)
        assert (await _unlock(client, PIN)).status_code == 200
        newer = await _lock(client)
        resume.set()
        late = await asyncio.wait_for(earlier, timeout=10)
    finally:
        resume.set()
        await asyncio.gather(earlier, return_exceptions=True)
    status = app.state.services.workspace_lock.status()

    assert late.status_code == 423
    assert late.json() == CHANGED
    assert status.locked
    assert status.lock_epoch == newer


async def test_what_the_api_sends_while_the_lock_is_on_is_not_kept_by_the_browser(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("neutral.bin", b"neutral lock fixture", "application/octet-stream")},
    )
    assert uploaded.status_code == 201
    content = f"/api/artifacts/{uploaded.json()['id']}/content"
    before = await client.get(content)

    assert (await _choose(client, 0)).status_code == 200
    after = await client.get(content)
    status = await client.get("/api/privacy/status")

    assert before.status_code == after.status_code == 200
    assert "no-store" not in before.headers["cache-control"]
    assert after.headers["cache-control"] == "no-store"
    assert status.headers["cache-control"] == "no-store"
