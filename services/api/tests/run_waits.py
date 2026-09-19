"""One bounded wait for work the server is still doing, and a failure that says so.

Nine test modules each grew their own copy of the same loop: poll a run, a job
or a chat until it reaches a terminal status, giving up after five seconds. The
duplication mattered twice over.

The five seconds were never measured. Instrumenting the two copies in
`test_api.py` over 104 completed waits gave a median of 0.2s and a maximum of
1.313s on an idle machine - 26% of the budget, so a 3.8x slowdown exhausts it.
The Windows CI leg is already 1.45x slower than Linux for the same suite before
any contention from the jobs beside it, and this wait has failed there four
times under changes that could not reach a chat run.

The ceiling below is deliberately far above the measured need. It costs nothing
when the machine is healthy, because the loop returns the moment the status is
terminal and never waits out the timer. It is a bound on patience, not a
performance assertion: a timeout here still has to be diagnosed rather than read
as proof of a defect.

The other half is the message. Of the eleven timeout sites this replaces, one
reported what it had seen and none reported how long it had waited, so a red run
could not distinguish "still running when time ran out", which is load, from
"came back failed", which is a defect. Both now appear in the failure, and an
unexpected terminal status fails immediately instead of spending the remaining
patience on a state that will not change.

Deliberately not general: this is for polling a record towards a terminal
status. The suite's many narrow `Event.wait` timeouts wait on another coroutine
in the same process and have four orders of magnitude of margin; they are not
this shape and are not changed.

`wait_until` is the same wait for a state that is not one terminal status - a
plan's steps reaching an exact list, text that has started to stream, a set of
jobs that have all started. It keeps the same patience for the same reason,
and its failure says what it last saw and how long it waited.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Collection, Mapping
from typing import Any

#: How long to keep polling before giving up. See the module docstring: the
#: measured worst case is 1.3s, and the loop exits on the status rather than on
#: the clock, so this only decides how bad a machine has to be before the suite
#: stops waiting for it.
PATIENCE_SECONDS = 30.0

#: How often to look. Unchanged from the loops this replaces.
INTERVAL_SECONDS = 0.03

#: The statuses that mean the server has stopped working on it.
TERMINAL_STATUSES = frozenset({"complete", "failed", "cancelled"})


async def wait_for_terminal_status(
    read: Callable[[], Awaitable[Mapping[str, Any] | None]],
    *,
    what: str,
    terminal: Collection[str] = TERMINAL_STATUSES,
    expected: str | None = "complete",
) -> Mapping[str, Any]:
    """Poll `read` until the record it returns reaches a terminal status.

    `read` returns the record, or None when there is not one yet - a chat whose
    assistant message has not been created is waiting, not failing.

    `expected` names the terminal status the caller requires, and any other
    terminal status fails at once. Pass None when the caller wants to decide
    for itself which ending was right.
    """

    started = time.monotonic()
    deadline = started + PATIENCE_SECONDS
    last = "<nothing to read yet>"
    while time.monotonic() < deadline:
        record = await read()
        if record is not None:
            last = str(record.get("status"))
            if last in terminal:
                waited = time.monotonic() - started
                if expected is not None and last != expected:
                    raise AssertionError(
                        f"{what} ended as {last!r} rather than {expected!r} after {waited:.2f}s"
                    )
                return record
        await asyncio.sleep(INTERVAL_SECONDS)
    waited = time.monotonic() - started
    raise AssertionError(
        f"{what} was still {last!r} after {waited:.2f}s, giving up at {PATIENCE_SECONDS:g}s"
    )


#: How much of the last value seen a timeout reports; a plan or a job list can
#: be long, and the failure only needs enough to tell load from a wrong state.
SEEN_CHARACTERS = 300


async def wait_until[T](
    read: Callable[[], Awaitable[T]],
    satisfied: Callable[[T], bool],
    *,
    what: str,
    interval: float = INTERVAL_SECONDS,
) -> T:
    """Poll `read` until `satisfied` holds for what it returns, and return that.

    The caller then asserts on exactly the value that ended the wait, so a wait
    that gives up still fails on the assertion the test always made. `interval`
    lets a caller keep the pace an existing test was written against.
    """

    started = time.monotonic()
    deadline = started + PATIENCE_SECONDS
    value = await read()
    while not satisfied(value):
        if time.monotonic() >= deadline:
            waited = time.monotonic() - started
            seen = repr(value)
            if len(seen) > SEEN_CHARACTERS:
                seen = seen[:SEEN_CHARACTERS] + "..."
            raise AssertionError(
                f"{what} was still {seen} after {waited:.2f}s, giving up at {PATIENCE_SECONDS:g}s"
            )
        await asyncio.sleep(interval)
        value = await read()
    return value
