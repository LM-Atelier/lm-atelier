"""The computer stays awake exactly while held work runs, and the work never depends on it."""

from __future__ import annotations

import ctypes
import sys
import threading
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from local_lm import power_inhibition
from local_lm.power_inhibition import (
    POWER_REASON,
    PowerInhibitor,
    SleepJobKind,
    UnsupportedPowerBackend,
)

NOON = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


class _Backend:
    """A platform that records every request, and can be told to refuse."""

    name = "test"
    supported = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.refuse_acquire = False
        self.refuse_release = False
        self._next = 0

    def acquire(self, reason: str) -> object:
        self.calls.append(("acquire", reason))
        if self.refuse_acquire:
            raise OSError("request refused")
        self._next += 1
        return self._next

    def release(self, handle: object) -> None:
        self.calls.append(("release", handle))
        if self.refuse_release:
            raise OSError("release refused")


def _inhibitor(backend: _Backend, *, enabled: bool = True) -> PowerInhibitor:
    return PowerInhibitor(backend, enabled=enabled, clock=lambda: NOON)


def test_overlapping_work_asks_once_and_lets_go_once() -> None:
    backend = _Backend()
    inhibitor = _inhibitor(backend)

    inhibitor.acquire("job_a", SleepJobKind.GENERATION)
    inhibitor.acquire("job_b", SleepJobKind.DOWNLOAD)
    inhibitor.release("job_a")
    assert backend.calls == [("acquire", POWER_REASON)]
    inhibitor.release("job_b")

    assert backend.calls == [("acquire", POWER_REASON), ("release", 1)]
    assert inhibitor.state().active is False and inhibitor.state().holder_count == 0


def test_holding_twice_or_releasing_nothing_changes_nothing() -> None:
    backend = _Backend()
    inhibitor = _inhibitor(backend)

    inhibitor.release("never_held")
    inhibitor.acquire("job_a", SleepJobKind.GENERATION)
    inhibitor.acquire("job_a", SleepJobKind.GENERATION)
    assert inhibitor.state().holder_count == 1
    inhibitor.release("job_a")
    inhibitor.release("job_a")

    assert backend.calls == [("acquire", POWER_REASON), ("release", 1)]


def test_a_held_block_lets_go_even_when_it_fails() -> None:
    backend = _Backend()
    inhibitor = _inhibitor(backend)

    with pytest.raises(RuntimeError), inhibitor.hold("job_a", SleepJobKind.GENERATION):
        assert inhibitor.state().active
        raise RuntimeError("the work failed")

    assert backend.calls == [("acquire", POWER_REASON), ("release", 1)]


def test_turning_the_setting_off_lets_the_machine_sleep_without_dropping_the_work() -> None:
    backend = _Backend()
    inhibitor = _inhibitor(backend, enabled=False)

    inhibitor.acquire("job_a", SleepJobKind.GENERATION)
    assert backend.calls == [] and inhibitor.state().holder_count == 1
    inhibitor.set_enabled(True)
    assert backend.calls == [("acquire", POWER_REASON)]
    inhibitor.set_enabled(False)
    assert backend.calls == [("acquire", POWER_REASON), ("release", 1)]

    state = inhibitor.state()
    assert (state.enabled, state.active, state.holder_count) == (False, False, 1)
    inhibitor.release("job_a")
    assert len(backend.calls) == 2


def test_a_platform_that_refuses_never_fails_the_work() -> None:
    backend = _Backend()
    backend.refuse_acquire = True
    inhibitor = _inhibitor(backend)

    with inhibitor.hold("job_a", SleepJobKind.DOWNLOAD):
        state = inhibitor.state()
        assert (state.active, state.holder_count, state.last_error) == (False, 1, "request refused")

    backend.refuse_acquire = False
    inhibitor.acquire("job_b", SleepJobKind.DOWNLOAD)
    assert inhibitor.state().active and inhibitor.state().last_error is None


def test_a_release_the_platform_refuses_is_reported_and_never_kept() -> None:
    backend = _Backend()
    backend.refuse_release = True
    inhibitor = _inhibitor(backend)

    inhibitor.acquire("job_a", SleepJobKind.GENERATION)
    inhibitor.release("job_a")

    state = inhibitor.state()
    assert (state.active, state.last_error) == (False, "release refused")


def test_the_state_counts_kinds_and_tells_the_platform_nothing_about_the_work() -> None:
    backend = _Backend()
    inhibitor = _inhibitor(backend)

    inhibitor.acquire("job_secret_name", SleepJobKind.GENERATION)
    inhibitor.acquire("job_b", SleepJobKind.GENERATION)
    inhibitor.acquire("job_c", SleepJobKind.RUNTIME_PREPARATION)

    state = inhibitor.state()
    assert state.kind_counts == {"generation": 2, "runtime_preparation": 1}
    assert state.since == NOON and state.backend == "test" and state.supported
    assert [reason for call, reason in backend.calls if call == "acquire"] == [POWER_REASON]
    assert "job" not in POWER_REASON and "generation" not in POWER_REASON


def test_an_unsupported_platform_is_reported_as_such() -> None:
    inhibitor = PowerInhibitor(UnsupportedPowerBackend())

    with inhibitor.hold("job_a", SleepJobKind.ARCHIVE):
        state = inhibitor.state()

    assert (state.supported, state.backend, state.active) == (False, "unsupported", False)
    assert state.last_error is not None


def test_work_starting_and_finishing_on_many_threads_still_asks_once() -> None:
    backend = _Backend()
    inhibitor = _inhibitor(backend)
    inhibitor.acquire("anchor", SleepJobKind.GENERATION)
    start = threading.Barrier(16)

    def work(index: int) -> None:
        start.wait()
        for round_ in range(50):
            with inhibitor.hold(f"job_{index}_{round_}", SleepJobKind.DOWNLOAD):
                pass

    threads = [threading.Thread(target=work, args=(index,)) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    inhibitor.release("anchor")

    assert backend.calls == [("acquire", POWER_REASON), ("release", 1)]


def _kernel(
    calls: list[tuple[str, object]], *, create: int = 0x50, set_ok: int = 1, clear_ok: int = 1
) -> SimpleNamespace:
    """kernel32's power request calls, recorded, each able to fail."""

    def create_request(context: object) -> int:
        reason = ctypes.cast(context, ctypes.POINTER(power_inhibition._ReasonContext)).contents
        calls.append(("create", (reason.version, reason.flags, reason.reason.simple)))
        return create

    def set_request(handle: int, kind: int) -> int:
        calls.append(("set", (handle, kind)))
        return set_ok

    def clear_request(handle: int, kind: int) -> int:
        calls.append(("clear", (handle, kind)))
        return clear_ok

    def close(handle: int) -> int:
        calls.append(("close", handle))
        return 1

    return SimpleNamespace(
        PowerCreateRequest=create_request,
        PowerSetRequest=set_request,
        PowerClearRequest=clear_request,
        CloseHandle=close,
    )


def test_windows_asks_that_the_system_stay_awake_and_lets_go_cleanly() -> None:
    calls: list[tuple[str, object]] = []
    backend = power_inhibition.WindowsPowerBackend(_kernel(calls))

    handle = backend.acquire(POWER_REASON)
    backend.release(handle)

    assert calls == [
        ("create", (0, 1, POWER_REASON)),
        ("set", (0x50, 1)),
        ("clear", (0x50, 1)),
        ("close", 0x50),
    ]


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        ({"create": 0}, ["create"]),
        ({"create": -1}, ["create"]),
        ({"set_ok": 0}, ["create", "set", "close"]),
    ],
    ids=["no-handle", "invalid-handle", "set-refused"],
)
def test_windows_refusing_the_request_raises_and_leaves_no_handle_open(
    kernel: dict[str, int], expected: list[str]
) -> None:
    calls: list[tuple[str, object]] = []
    if kernel.get("create") == -1:
        kernel = {"create": ctypes.c_void_p(-1).value or 0}
    backend = power_inhibition.WindowsPowerBackend(_kernel(calls, **kernel), lambda: 5)

    with pytest.raises(OSError) as refused:
        backend.acquire(POWER_REASON)

    assert refused.value.errno == 5
    assert [name for name, _ in calls] == expected


def test_windows_refusing_to_end_the_request_still_closes_it() -> None:
    calls: list[tuple[str, object]] = []
    backend = power_inhibition.WindowsPowerBackend(_kernel(calls, clear_ok=0), lambda: 6)
    handle = backend.acquire(POWER_REASON)

    with pytest.raises(OSError) as refused:
        backend.release(handle)

    assert refused.value.errno == 6
    assert [name for name, _ in calls] == ["create", "set", "clear", "close"]


def test_each_platform_gets_its_own_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(power_inhibition.sys, "platform", "linux")
    assert isinstance(power_inhibition.default_power_backend(), UnsupportedPowerBackend)
    monkeypatch.setattr(power_inhibition.sys, "platform", "win32")
    monkeypatch.setattr(power_inhibition, "_kernel32", lambda: _kernel([]))
    assert isinstance(
        power_inhibition.default_power_backend(), power_inhibition.WindowsPowerBackend
    )


@pytest.mark.skipif(sys.platform != "win32", reason="a real power request exists only on Windows")
def test_a_real_windows_power_request_is_made_and_ended() -> None:
    inhibitor = PowerInhibitor(power_inhibition.WindowsPowerBackend())

    with inhibitor.hold("job_a", SleepJobKind.GENERATION):
        state = inhibitor.state()

    assert (state.active, state.last_error, state.backend) == (True, None, "windows-power-request")
    assert inhibitor.state().active is False and inhibitor.state().last_error is None
