"""Keeping the computer from sleeping while durable work runs, and only then.

Each running attempt at a job that should keep the machine awake holds the
inhibitor under its own key; the scheduler uses the attempt's claim. The
operating system is asked to stay awake when the first holder arrives and told
it may sleep when the last one leaves, however the holders overlap, so the
request starts and ends exactly once per busy stretch. Turning the setting off
lets the machine sleep at once without touching the work; turning it back on
asks again for the work still running.

Nothing here can fail the work it protects. A platform that cannot keep the
machine awake, or refuses to, is recorded and reported, and the job goes on.
The reason given to the operating system is one fixed sentence, never a
prompt, a file name or anything else about the work itself.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, cast

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


class _DetailedReason(ctypes.Structure):
    _fields_ = (
        ("module", ctypes.c_void_p),
        ("reason_id", ctypes.c_ulong),
        ("count", ctypes.c_ulong),
        ("strings", ctypes.c_void_p),
    )


class _Reason(ctypes.Union):
    _fields_ = (("detailed", _DetailedReason), ("simple", ctypes.c_wchar_p))


class _ReasonContext(ctypes.Structure):
    """Windows' REASON_CONTEXT, used here only with its simple reason string."""

    _fields_ = (("version", ctypes.c_ulong), ("flags", ctypes.c_ulong), ("reason", _Reason))


_POWER_REQUEST_CONTEXT_VERSION = 0
_POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1
# PowerRequestSystemRequired: the system stays awake; the display may still sleep.
_POWER_REQUEST_SYSTEM_REQUIRED = 1
_INVALID_HANDLE = ctypes.c_void_p(-1).value


def _kernel32() -> Any:
    # Widened to Any: this module is type-checked for POSIX too, where ctypes
    # has no WinDLL at all.
    windows: Any = ctypes
    kernel = windows.WinDLL("kernel32", use_last_error=True)
    kernel.PowerCreateRequest.argtypes = (ctypes.POINTER(_ReasonContext),)
    kernel.PowerCreateRequest.restype = ctypes.c_void_p
    kernel.PowerSetRequest.argtypes = (ctypes.c_void_p, ctypes.c_int)
    kernel.PowerSetRequest.restype = ctypes.c_int
    kernel.PowerClearRequest.argtypes = (ctypes.c_void_p, ctypes.c_int)
    kernel.PowerClearRequest.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.restype = ctypes.c_int
    return kernel


def _last_error() -> int:
    windows: Any = ctypes
    return int(windows.get_last_error())


class WindowsPowerBackend:
    """A Windows power request asking that the system, not the display, stay awake.

    The request is a handle the process owns, so Windows ends it if the
    process dies; sleep the person asks for, and the lid and battery policy,
    still win.
    """

    name = "windows-power-request"
    supported = True

    def __init__(self, kernel: Any = None, last_error: Callable[[], int] = _last_error) -> None:
        self._kernel = kernel if kernel is not None else _kernel32()
        self._last_error = last_error

    def acquire(self, reason: str) -> object:
        context = _ReasonContext(
            _POWER_REQUEST_CONTEXT_VERSION,
            _POWER_REQUEST_CONTEXT_SIMPLE_STRING,
            _Reason(simple=reason),
        )
        handle = self._kernel.PowerCreateRequest(ctypes.byref(context))
        if not handle or handle == _INVALID_HANDLE:
            raise OSError(self._last_error(), "Windows refused to create a power request.")
        if not self._kernel.PowerSetRequest(handle, _POWER_REQUEST_SYSTEM_REQUIRED):
            error = self._last_error()
            self._kernel.CloseHandle(handle)
            raise OSError(error, "Windows refused to keep the computer awake.")
        # The reason's buffer is kept alive for as long as the request is.
        return (handle, context)

    def release(self, handle: object) -> None:
        request, _context = cast(tuple[object, object], handle)
        try:
            if not self._kernel.PowerClearRequest(request, _POWER_REQUEST_SYSTEM_REQUIRED):
                raise OSError(self._last_error(), "Windows refused to end the power request.")
        finally:
            self._kernel.CloseHandle(request)


def _logind_inhibit(reason: str) -> int:
    """Ask systemd-logind for an idle inhibitor and return the descriptor that holds it."""

    # Loaded by name: jeepney is installed only on Linux, and this module is
    # type-checked everywhere.
    jeepney: Any = importlib.import_module("jeepney")
    blocking: Any = importlib.import_module("jeepney.io.blocking")
    wrappers: Any = importlib.import_module("jeepney.wrappers")
    manager = jeepney.DBusAddress(
        "/org/freedesktop/login1",
        bus_name="org.freedesktop.login1",
        interface="org.freedesktop.login1.Manager",
    )
    # "idle" holds off the idle timer only; sleep the person asks for still wins.
    message = jeepney.new_method_call(
        manager, "Inhibit", "ssss", ("idle", "LM Atelier", reason, "block")
    )
    try:
        with blocking.open_dbus_connection(bus="SYSTEM", enable_fds=True) as connection:
            body = wrappers.unwrap_msg(connection.send_and_get_reply(message, timeout=5))
            return int(body[0].to_raw_fd())
    except OSError:
        raise
    except Exception as error:
        raise OSError("systemd-logind refused an idle inhibitor.") from error


class LinuxLogindPowerBackend:
    """A systemd-logind idle inhibitor, held as a descriptor this process owns.

    Closing the descriptor ends the inhibitor, and so does the process ending,
    so a crash can never leave the machine held awake.
    """

    name = "logind-idle-inhibitor"
    supported = True

    def __init__(self, inhibit: Callable[[str], int] = _logind_inhibit) -> None:
        self._inhibit = inhibit

    def acquire(self, reason: str) -> object:
        return self._inhibit(reason)

    def release(self, handle: object) -> None:
        os.close(cast(int, handle))


def default_power_backend() -> PowerBackend:
    """The way this platform keeps the computer awake, or an honest unsupported one."""

    if sys.platform == "win32":
        return WindowsPowerBackend()
    if sys.platform.startswith("linux"):
        return LinuxLogindPowerBackend()
    return UnsupportedPowerBackend()


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

    def acquire(self, holder: str, kind: SleepJobKind) -> None:
        """Hold the machine awake for one holder; holding again under one key changes nothing."""

        with self._lock:
            self._holders.setdefault(holder, kind)
            self._ask()

    def release(self, holder: str) -> None:
        """Stop holding for one holder; releasing one that holds nothing changes nothing."""

        with self._lock:
            self._holders.pop(holder, None)
            if not self._holders:
                self._let_sleep()

    @contextmanager
    def hold(self, holder: str, kind: SleepJobKind) -> Iterator[None]:
        """Hold for the duration of a block, released however the block ends."""

        self.acquire(holder, kind)
        try:
            yield
        finally:
            self.release(holder)

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
