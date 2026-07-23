from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from local_lm.domain import elapsed_milliseconds


def test_elapsed_milliseconds_treats_naive_database_timestamp_as_utc() -> None:
    started_at = datetime(2026, 7, 23, 1, 0, 0)
    completed_at = datetime(2026, 7, 23, 1, 0, 0, 250_000, tzinfo=UTC)

    assert elapsed_milliseconds(started_at, completed_at) == 250


def test_elapsed_milliseconds_normalizes_offsets_and_clamps_clock_skew() -> None:
    pacific = timezone(-timedelta(hours=7))
    started_at = datetime(2026, 7, 23, 1, 0, 0, tzinfo=UTC)
    completed_at = datetime(2026, 7, 22, 18, 0, 0, 400_000, tzinfo=pacific)

    assert elapsed_milliseconds(started_at, completed_at) == 400
    assert elapsed_milliseconds(completed_at, started_at) == 0
