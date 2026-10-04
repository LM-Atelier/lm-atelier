"""Keeping the computer from sleeping while durable work runs, and only then.

Each running job that should keep the machine awake holds the inhibitor under
its durable job id. The operating system is asked to stay awake when the
first holder arrives and told it may sleep when the last one leaves, however
the holders overlap, so the request starts and ends exactly once per busy
stretch. Turning the setting off lets the machine sleep at once without
touching the work; turning it back on asks again for the work still running.

Nothing here can fail the work it protects. A platform that cannot keep the
machine awake, or refuses to, is recorded and reported, and the job goes on.
The reason given to the operating system is one fixed sentence, never a
prompt, a file name or anything else about the work itself.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from .domain import utcnow

# The only text the operating system ever sees about why the machine is awake.
POWER_REASON = "LM Atelier is finishing work that is running now."


class SleepJobKind(StrEnum):
    """The kinds of durable work that may keep the machine awake."""

    GENERATION = "generation"
    DOWNLOAD = "download"
    RUNTIME_PREPARATION = "runtime_preparation"
    ARCHIVE = "archive"


class PowerBackend(Protocol):
    """One platform's way to keep the machine awake.

    ``acquire`` returns an opaque handle for one request, or raises OSError
    when the request cannot be made; ``release`` ends that request.
    """

    name: str
    supported: bool

    def acquire(self, reason: str) -> object: ...

    def release(self, handle: object) -> None: ...


class UnsupportedPowerBackend:
    """A platform with no way to keep the machine awake from here."""

    name = "unsupported"
    supported = False

    def acquire(self, reason: str) -> object:
        raise OSError("keeping the computer awake is not supported here")

    def release(self, handle: object) -> None:
        return None


@dataclass(frozen=True)
class PowerInhibitionStateV1:
    supported: bool
    backend: str
    enabled: bool
    active: bool
    holder_count: int
    kind_counts: dict[str, int] = field(default_factory=dict)
    since: datetime | None = None
    last_error: str | None = None


class PowerInhibitor:
    """Reference-counted permission for the operating system to keep the machine awake."""

    def __init__(
        self,
        backend: PowerBackend,
        *,
        enabled: bool = True,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._backend = backend
        self._enabled = enabled
        self._clock = clock
        self._lock = threading.Lock()
        self._holders: dict[str, SleepJobKind] = {}
        self._handle: object | None = None
        self._since: datetime | None = None
        self._last_error: str | None = None

    def acquire(self, job_id: str, kind: SleepJobKind) -> None:
        """Hold the machine awake for one job; holding again under the same id changes nothing."""

        with self._lock:
            self._holders.setdefault(job_id, kind)
            self._ask()

    def release(self, job_id: str) -> None:
        """Stop holding for one job; releasing a job that holds nothing changes nothing."""

        with self._lock:
            self._holders.pop(job_id, None)
            if not self._holders:
                self._let_sleep()

    @contextmanager
    def hold(self, job_id: str, kind: SleepJobKind) -> Iterator[None]:
        """Hold for the duration of a block, released however the block ends."""

        self.acquire(job_id, kind)
        try:
            yield
        finally:
            self.release(job_id)

    def set_enabled(self, enabled: bool) -> None:
        """Turn the setting on or off; running work is never cancelled either way."""

        with self._lock:
            self._enabled = enabled
            if enabled:
                self._ask()
            else:
                self._let_sleep()

    def state(self) -> PowerInhibitionStateV1:
        with self._lock:
            return PowerInhibitionStateV1(
                supported=self._backend.supported,
                backend=self._backend.name,
                enabled=self._enabled,
                active=self._handle is not None,
                holder_count=len(self._holders),
                kind_counts=dict(Counter(kind.value for kind in self._holders.values())),
                since=self._since,
                last_error=self._last_error,
            )

    def _ask(self) -> None:
        # Only the first holder, with the setting on, reaches the platform.
        if not self._enabled or not self._holders or self._handle is not None:
            return
        try:
            self._handle = self._backend.acquire(POWER_REASON)
        except OSError as error:
            self._last_error = _bounded(error)
            return
        self._since = self._clock()
        self._last_error = None

    def _let_sleep(self) -> None:
        handle, self._handle, self._since = self._handle, None, None
        if handle is None:
            return
        try:
            self._backend.release(handle)
        except OSError as error:
            self._last_error = _bounded(error)


def _bounded(error: OSError) -> str:
    # The platform's own words, cut short; they name a failure, never the work.
    return (str(error) or type(error).__name__)[:200]
