from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from local_lm import domain
from local_lm.domain import Operation, elapsed_milliseconds
from local_lm.exports import ProjectExporter
from local_lm.orchestrator import ConversationOrchestrator


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


_EXPECTED_OPERATION_ROLES = {
    Operation.TEXT: "chat",
    Operation.TEXT_TO_IMAGE: "image",
    Operation.IMAGE_TO_IMAGE: "image",
    Operation.TEXT_TO_VIDEO: "video",
    Operation.IMAGE_TO_VIDEO: "video",
}


def test_role_examples_cover_the_operation_vocabulary() -> None:
    assert set(_EXPECTED_OPERATION_ROLES) == set(Operation)


@pytest.mark.parametrize(("operation", "expected"), _EXPECTED_OPERATION_ROLES.items())
def test_operation_has_an_explicit_model_role(operation: Operation, expected: str) -> None:
    assert domain.operation_model_role(operation) == expected


@pytest.mark.parametrize(("operation", "expected"), _EXPECTED_OPERATION_ROLES.items())
def test_export_and_execution_keep_the_same_operation_role(
    operation: Operation, expected: str
) -> None:
    assert ProjectExporter._role_for_operation(operation) == expected
    assert ConversationOrchestrator._role_for_operation(operation) == expected
