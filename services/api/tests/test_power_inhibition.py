"""The computer stays awake exactly while held work runs, and the work never depends on it."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

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
