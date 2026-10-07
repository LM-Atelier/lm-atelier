"""The shared wait must say what happened, and stop early when it can.

The loops this helper replaces failed the same way for years: a bounded wait
timed out and said only "run did not complete", so a red run could not tell a
slow machine from a broken one. These cases pin the two things that make the
next failure readable - the elapsed time and the last status seen - and the one
that keeps it quick, which is refusing a terminal status the caller did not ask
for instead of waiting out the remaining patience on a state that will not
change.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import pytest
import run_waits
from run_waits import wait_for_terminal_status, wait_until


async def test_it_returns_as_soon_as_the_status_is_terminal() -> None:
    seen = 0

    async def read() -> dict[str, Any]:
        nonlocal seen
        seen += 1
        return {"status": "running" if seen < 3 else "complete", "id": "r1"}

    record = await wait_for_terminal_status(read, what="run r1")

    assert record["id"] == "r1"
    assert seen == 3, "it kept polling after the terminal status"


async def test_a_record_that_does_not_exist_yet_is_waited_for_not_failed() -> None:
    seen = 0

    async def read() -> dict[str, Any] | None:
        nonlocal seen
        seen += 1
        # A chat whose assistant message has not been created yet reads as
        # nothing at all, which is waiting rather than failing.
        return None if seen < 2 else {"status": "complete"}

    assert (await wait_for_terminal_status(read, what="the assistant run"))["status"] == "complete"


async def test_an_unexpected_ending_fails_at_once_and_names_both_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_waits, "PATIENCE_SECONDS", 30.0)

    async def read() -> dict[str, Any]:
        return {"status": "failed"}

    started = time.monotonic()
    with pytest.raises(AssertionError) as caught:
        await wait_for_terminal_status(read, what="run r2")
    waited = time.monotonic() - started

    assert "'failed'" in str(caught.value)
    assert "'complete'" in str(caught.value)
    assert "run r2" in str(caught.value)
    # The point of the early refusal: a run that has already failed is not going
    # to become complete, so spending the remaining patience on it is thirty
    # seconds of nothing per occurrence.
    assert waited < 1.0, f"it waited {waited:.2f}s for a status that was already final"


async def test_a_timeout_reports_how_long_it_waited_and_what_it_last_saw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_waits, "PATIENCE_SECONDS", 0.2)

    async def read() -> dict[str, Any]:
        return {"status": "running"}

    with pytest.raises(AssertionError) as caught:
        await wait_for_terminal_status(read, what="run r3")

    message = str(caught.value)
    assert "run r3" in message
    assert "'running'" in message, "the last status seen is the whole diagnosis"

    # Read the number back rather than looking for "0.2" in the text. The
    # budget is what the wait is willing to spend, not what it will have spent:
    # a poll that wakes late spends more, and the message then says 0.35 and
    # contains no "0.2" at all - so a substring made this control fail on a
    # loaded machine while the helper was working exactly as intended, which is
    # the very fault this file exists to remove.
    reported = re.search(r"after (\d+\.\d+)s", message)
    assert reported is not None, f"no elapsed time in the message: {message}"
    assert float(reported.group(1)) >= run_waits.PATIENCE_SECONDS, (
        "the elapsed time must be the real one; a late wakeup can only make it larger"
    )
    # And the ceiling has to read as itself. Formatted to no decimal places it
    # rendered a fifth of a second as "0s", which is the one number in the
    # sentence that is not a measurement and so has no excuse for being wrong.
    assert "giving up at 0.2s" in message


async def test_a_caller_can_accept_any_ending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_waits, "PATIENCE_SECONDS", 30.0)

    async def read() -> dict[str, Any]:
        return {"status": "cancelled"}

    record = await wait_for_terminal_status(read, what="run r4", expected=None)

    assert record["status"] == "cancelled"


async def test_a_caller_can_widen_what_counts_as_an_ending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_waits, "PATIENCE_SECONDS", 0.2)

    async def read() -> dict[str, Any]:
        return {"status": "interrupted"}

    # Not terminal by default, so this one times out ...
    with pytest.raises(AssertionError):
        await wait_for_terminal_status(read, what="the job", expected=None)

    # ... and is an ending for a caller that says so.
    record = await wait_for_terminal_status(
        read,
        what="the job",
        terminal=frozenset({"interrupted"}),
        expected=None,
    )
    assert record["status"] == "interrupted"


async def test_a_predicate_wait_returns_the_value_that_satisfied_it_at_once() -> None:
    seen = 0

    async def read() -> list[str]:
        nonlocal seen
        seen += 1
        return ["running"] * (3 - seen) + ["complete"] * seen

    value = await wait_until(read, lambda states: states == ["complete"] * 3, what="the plan")

    assert value == ["complete"] * 3
    assert seen == 3, "it kept polling after the state was reached"


async def test_a_state_already_reached_is_returned_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    async def read() -> str:
        return "streamed"

    assert await wait_until(read, bool, what="the text") == "streamed"
    assert slept == []


async def test_a_predicate_wait_polls_at_the_pace_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    seen = 0

    async def read() -> int:
        nonlocal seen
        seen += 1
        return seen

    await wait_until(read, lambda count: count == 3, what="the count", interval=0.01)
    await wait_until(read, lambda count: count == 5, what="the count")

    assert slept == [0.01, 0.01, run_waits.INTERVAL_SECONDS]


async def test_a_predicate_timeout_reports_the_last_value_and_how_long_it_waited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_waits, "PATIENCE_SECONDS", 0.2)

    async def read() -> list[str]:
        return ["failed", "blocked"]

    with pytest.raises(AssertionError) as caught:
        await wait_until(read, lambda states: states == ["complete"], what="the steps of plan p1")

    message = str(caught.value)
    assert "the steps of plan p1" in message
    assert "['failed', 'blocked']" in message, "the last value seen is the whole diagnosis"
    reported = re.search(r"after (\d+\.\d+)s", message)
    assert reported is not None, f"no elapsed time in the message: {message}"
    assert float(reported.group(1)) >= run_waits.PATIENCE_SECONDS
    assert "giving up at 0.2s" in message


async def test_a_long_last_value_is_cut_short_in_the_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_waits, "PATIENCE_SECONDS", 0.05)

    long_value = "x" * (run_waits.SEEN_CHARACTERS * 3)

    async def read() -> str:
        return long_value

    with pytest.raises(AssertionError) as caught:
        await wait_until(read, lambda _value: False, what="the listing")

    message = str(caught.value)
    shown = repr(long_value)
    assert shown[: run_waits.SEEN_CHARACTERS] + "..." in message
    assert shown[: run_waits.SEEN_CHARACTERS + 1] not in message
