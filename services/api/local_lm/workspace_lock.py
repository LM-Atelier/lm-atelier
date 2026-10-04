"""Whether the workspace is locked right now, and the gate that keeps it shut.

The server enforces the lock, not the page. While it is locked, every API
request is refused with 423 except the few a locked page needs: a session, the
health and readiness probes, the lock's status, and unlocking. The event socket
is refused in its own handler, because no HTTP middleware sees a websocket.

Every lock mints a new epoch, an opaque token a page sends back with each
request once it knows one. A request carrying an epoch from before the latest
lock is refused too, so a page that slept through a lock and an unlock cannot
carry on as if nothing had happened. Epochs are only ever compared for equality.

The state lives in memory and starts locked. Startup settles it from the saved
setting before anything is served, and a setting that cannot be read leaves the
workspace locked.
"""

from __future__ import annotations

import asyncio
import math
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .workspace_lock_policy import SavedWorkspaceLock

LOCK_EPOCH_HEADER = "x-local-lm-lock-epoch"
#: How a live event socket is closed when the workspace locks: 4423 follows
#: HTTP's 423 the way 4401 follows 401.
LOCKED_CLOSE_CODE = 4423
#: What a locked page may still ask besides the public routes, by method and path.
REACHABLE_WHILE_LOCKED = frozenset(
    {("GET", "/api/privacy/status"), ("POST", "/api/privacy/unlock")}
)
#: Wrong PINs in a row that need no wait between them. From the last of these
#: on, each wrong PIN doubles the wait before the next check, up to the cap.
FREE_FAILURES = 5
MAX_WAIT_SECONDS = 900

WorkspaceLockRefusal = Literal["workspace-locked", "workspace-lock-changed"]
_REFUSAL_DETAILS: dict[WorkspaceLockRefusal, str] = {
    "workspace-locked": "LM Atelier is locked.",
    "workspace-lock-changed": "The workspace lock changed. Reload to continue.",
}


class WorkspaceLockDisabled(Exception):
    """Locking was asked for while the lock is turned off."""


class WorkspaceLockChanged(Exception):
    """The workspace was locked again while an unlock was being decided."""


@dataclass(frozen=True)
class WorkspaceLockState:
    locked: bool
    enabled: bool
    require_pin: bool
    #: None while the lock is off.
    lock_epoch: str | None


def wait_after(failures: int) -> int:
    """Seconds the next PIN check waits after this many wrong PINs in a row."""

    if failures < FREE_FAILURES:
        return 0
    # Capped before raising two to it, since the count itself has no bound.
    return min(1 << min(failures - FREE_FAILURES, 10), MAX_WAIT_SECONDS)


class UnlockAttempts:
    """Pace PIN checks for the whole installation.

    Every tab shares one session, so wrong PINs are counted for the installation
    rather than per caller. Five in a row are free; after that the wait before
    the next check doubles with each one, up to fifteen minutes. A right PIN, or
    a restart, clears the count. A check refused for pacing is not counted and
    does not lengthen the wait.

    Only one derivation runs at a time, for checks and new PINs alike, since
    each holds a quarter of a gigabyte while it runs.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.consecutive_failures = 0
        self.next_allowed_at: float | None = None
        self.in_flight = False
        self._guard = threading.Lock()

    def admit(self) -> int | None:
        """Take the one derivation slot, or say how many seconds to wait first."""

        with self._guard:
            if self.in_flight:
                return 1
            if self.next_allowed_at is not None:
                remaining = self.next_allowed_at - self.clock()
                if remaining > 0:
                    return max(1, math.ceil(remaining))
            self.in_flight = True
            return None

    def fail(self) -> int:
        """Count a wrong PIN, and return the wait that now applies, if any."""

        with self._guard:
            self.consecutive_failures += 1
            wait = wait_after(self.consecutive_failures)
            self.next_allowed_at = self.clock() + wait if wait else None
            return wait

    def succeed(self) -> None:
        with self._guard:
            self.consecutive_failures = 0
            self.next_allowed_at = None

    def release(self) -> None:
        with self._guard:
            self.in_flight = False


def _new_epoch() -> str:
    return secrets.token_urlsafe(16)


class WorkspaceLock:
    """The lock state for this run of the service.

    A thread lock guards it, because synchronous routes run in a thread pool.
    The signal that closes live event sockets is an asyncio event, so the calls
    that set or replace it, `settle`, `lock` and `unlock`, run on the event loop.
    """

    def __init__(self, attempts: UnlockAttempts | None = None) -> None:
        self.attempts = attempts or UnlockAttempts()
        self._guard = threading.Lock()
        # Locked, on and PIN-protected until startup has read the saved setting.
        self._locked = True
        self._enabled = True
        self._require_pin = True
        self._setting_readable = False
        self._epoch = _new_epoch()
        self._signal = asyncio.Event()
        self._signal.set()

    @property
    def setting_readable(self) -> bool:
        with self._guard:
            return self._setting_readable

    def settle(self, saved: SavedWorkspaceLock | None) -> None:
        """Start from the saved setting, or stay locked when it could not be read (None)."""

        with self._guard:
            if saved is None:
                self._setting_readable = False
                self._enabled = True
                self._require_pin = True
            else:
                self._setting_readable = True
                self._enabled = saved.enabled
                self._require_pin = saved.pin is not None
            self._locked = self._enabled
            if self._locked:
                self._signal.set()
            elif self._signal.is_set():
                self._signal = asyncio.Event()

    def lock(self) -> WorkspaceLockState:
        with self._guard:
            if not self._enabled:
                raise WorkspaceLockDisabled("The workspace lock is off.")
            if not self._locked:
                self._locked = True
                self._epoch = _new_epoch()
            self._signal.set()
        return self.status()

    def unlock(self, *, admitted_epoch: str | None = None) -> WorkspaceLockState:
        """Open the workspace. The epoch stays the one the lock minted.

        `admitted_epoch` is the epoch an unlock was decided against. A PIN
        check takes time; if the workspace was opened and locked again while it
        ran, its answer belongs to a lock that no longer exists, so it opens
        nothing and raises WorkspaceLockChanged instead.
        """

        with self._guard:
            if self._locked:
                if admitted_epoch is not None and admitted_epoch != self._epoch:
                    raise WorkspaceLockChanged("The workspace was locked again.")
                self._locked = False
                self._signal = asyncio.Event()
        return self.status()

    def note_policy(self, *, enabled: bool, require_pin: bool) -> None:
        """Take a setting that was just committed. Safe from any thread."""

        with self._guard:
            self._enabled = enabled
            self._require_pin = require_pin
            self._setting_readable = True

    def status(self) -> WorkspaceLockState:
        with self._guard:
            return WorkspaceLockState(
                locked=self._locked,
                enabled=self._enabled,
                require_pin=self._require_pin,
                lock_epoch=self._epoch if self._enabled else None,
            )

    def lock_signal(self) -> asyncio.Event:
        """Set when the workspace locks. An unlock replaces it rather than clearing it."""

        with self._guard:
            return self._signal

    def refusal_code(self, epoch: str | None) -> WorkspaceLockRefusal | None:
        """Why a gated request with this epoch header is refused, or None to let it through."""

        with self._guard:
            if self._locked:
                return "workspace-locked"
            if self._enabled and epoch is not None and epoch != self._epoch:
                return "workspace-lock-changed"
            return None


class WorkspaceLockMiddleware:
    """Refuse API requests while the workspace is locked.

    It sits inside the session checks, so a caller with no session or no CSRF
    token is told that first, and outside the body limits, so nothing is read
    before a refusal.
    """

    def __init__(self, app: ASGIApp, *, lock: WorkspaceLock, public_paths: frozenset[str]) -> None:
        self.app = app
        self.lock = lock
        self.public_paths = public_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # The path exactly as the session check reads it, so nothing that check
        # covers escapes this one.
        path = Request(scope).url.path
        if not path.startswith("/api"):
            await self.app(scope, receive, send)
            return
        if not self._always_reachable(scope, path):
            code = self.lock.refusal_code(Headers(scope=scope).get(LOCK_EPOCH_HEADER))
            if code is not None:
                response = JSONResponse(
                    {"detail": _REFUSAL_DETAILS[code], "code": code},
                    status_code=423,
                    headers={"Cache-Control": "no-store"},
                )
                await response(scope, receive, send)
                return
        if self.lock.status().enabled:
            # While the lock is on, nothing the API sends may be kept: a copy
            # in the browser's cache could be shown again after the next lock.
            send = _without_caching(send)
        await self.app(scope, receive, send)

    def _always_reachable(self, scope: Scope, path: str) -> bool:
        # The router matches the raw path. Exempt only a request whose raw path
        # is the one checked, so no other route can hide behind an exempt name.
        method: object = scope.get("method")
        raw_path: object = scope.get("path")
        listed = path in self.public_paths or (method, path) in REACHABLE_WHILE_LOCKED
        return listed and raw_path == path


def _without_caching(send: Send) -> Send:
    """Send each response marked so that neither the browser nor anything between keeps it."""

    async def sending(message: Message) -> None:
        if message["type"] == "http.response.start":
            MutableHeaders(scope=message)["cache-control"] = "no-store"
        await send(message)

    return sending
