"""Recovery responses carry identities and bounded facts rather than resource content."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from local_lm.recovery_v1 import (
    DeletionQuarantineV1,
    PurgeRecoveryV1,
    RecoveryAction,
    RecoveryCommandV1,
    RecoveryCountsV1,
    RecoveryItemV1,
    RecoveryKind,
    RecoveryState,
)


def _item(**changes: Any) -> dict[str, Any]:
    return {
        "deletion_id": "recovery-garden",
        "kind": "chat",
        "subject_id": "chat-garden",
        "display_label": "Garden notes",
        "original_location": {"project_id": "project-garden", "project_label": "Garden"},
        "deleted_at": datetime(2026, 10, 2, tzinfo=UTC),
        "purge_after": datetime(2026, 11, 1, tzinfo=UTC),
        "state": "recoverable",
        "revision": "a" * 64,
        "counts": {"messages": 4, "jobs": 2, "retained_bytes": 1024},
        **changes,
    }


def test_a_recovery_item_keeps_its_identity_location_and_original_deadline() -> None:
    item = RecoveryItemV1.model_validate(_item())
    copied = RecoveryItemV1.model_validate_json(item.model_dump_json())

    assert copied == item
    assert copied.kind is RecoveryKind.CHAT
    assert copied.state is RecoveryState.RECOVERABLE
    assert copied.subject_id == "chat-garden"
    assert copied.original_location.project_id == "project-garden"
    assert copied.purge_after == datetime(2026, 11, 1, tzinfo=UTC)
    assert copied.counts.messages == 4 and copied.counts.retained_bytes == 1024
    with pytest.raises(ValidationError):
        copied.subject_id = "chat-substitute"


def test_sqlite_utc_dates_and_explicit_offsets_keep_the_same_recovery_clock() -> None:
    deleted = datetime(2026, 10, 2)
    deadline = datetime(2026, 11, 1, 3, tzinfo=timezone(timedelta(hours=3)))

    item = RecoveryItemV1.model_validate(_item(deleted_at=deleted, purge_after=deadline))

    assert item.deleted_at == datetime(2026, 10, 2, tzinfo=UTC)
    assert item.purge_after == datetime(2026, 11, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "field", ["prompt", "negative_prompt", "grammar", "media", "raw_path", "credential"]
)
def test_a_recovery_response_refuses_resource_payload_fields(field: str) -> None:
    with pytest.raises(ValidationError) as refused:
        RecoveryItemV1.model_validate(_item(**{field: "neutral fixture"}))

    assert any(error["type"] == "extra_forbidden" for error in refused.value.errors())


@pytest.mark.parametrize("value", [-1, True, "4"])
def test_recovery_counts_refuse_negative_or_coerced_measurements(value: object) -> None:
    with pytest.raises(ValidationError):
        RecoveryCountsV1.model_validate({"messages": value})


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "z" * 64])
def test_a_recovery_command_requires_an_exact_lowercase_digest(digest: str) -> None:
    with pytest.raises(ValidationError):
        RecoveryCommandV1.model_validate(
            {"expected_revision": digest, "impact_sha256": "b" * 64, "operation_key": "garden"}
        )


def test_permanent_deletion_requires_its_exact_acknowledgement() -> None:
    command = {"expected_revision": "a" * 64, "impact_sha256": "b" * 64, "operation_key": "garden"}
    with pytest.raises(ValidationError):
        PurgeRecoveryV1.model_validate(command)
    with pytest.raises(ValidationError):
        PurgeRecoveryV1.model_validate({**command, "acknowledgement": "restore"})

    accepted = PurgeRecoveryV1.model_validate({**command, "acknowledgement": "permanently-delete"})
    assert accepted.acknowledgement == "permanently-delete"


def test_the_deletion_quarantine_binds_only_the_checked_command() -> None:
    quarantine = DeletionQuarantineV1(
        kind=RecoveryKind.CHAT,
        subject_id="chat-garden",
        action=RecoveryAction.TRASH,
        expected_revision="a" * 64,
        impact_sha256="b" * 64,
        operation_key="garden-trash",
        delete_generated_media=True,
    )

    assert set(quarantine.model_dump()) == {
        "kind",
        "subject_id",
        "action",
        "expected_revision",
        "impact_sha256",
        "operation_key",
        "delete_generated_media",
    }


def test_an_expiry_that_precedes_deletion_is_not_a_valid_recovery_item() -> None:
    with pytest.raises(ValidationError):
        RecoveryItemV1.model_validate(_item(purge_after=datetime(2026, 10, 1, tzinfo=UTC)))
