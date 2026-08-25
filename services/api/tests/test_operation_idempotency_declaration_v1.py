from __future__ import annotations

import pytest

from local_lm.operation_idempotency_declaration_v1 import (
    DIGEST_LEN,
    INVALID,
    MAX_ID,
    OperationIdempotencyDeclarationV1,
    OperationIdempotencyError,
    classify_idempotency_digest,
    declare_operation_idempotency,
)


def test_records_requested_rules_without_persist() -> None:
    declaration = declare_operation_idempotency(
        chat_id="c1",
        operation_key="op1",
        request_digest="ab" * 32,
        expected_message_id="m1",
        expected_revision_id="r1",
    )
    assert declaration.requested_replay_rule == "same_digest_is_replay"
    assert declaration.requested_conflict_rule == "different_digest_is_conflict"
    assert declaration.repository_event_persisted is False
    assert declaration.execution_authorized is False
    assert not hasattr(declaration, "same_key_same_digest_replays")
    assert not hasattr(declaration, "same_key_different_digest_conflicts")


def test_digest_compare_is_not_a_repository_replay() -> None:
    first = declare_operation_idempotency(
        chat_id="chat-a",
        operation_key="delete-a",
        request_digest="aa" * 32,
        expected_message_id="m-a",
    )
    second = declare_operation_idempotency(
        chat_id="chat-b",
        operation_key="edit-b",
        request_digest="aa" * 32,
        expected_message_id="m-b",
    )
    assert first.repository_event_persisted is False
    assert second.repository_event_persisted is False
    assert (
        classify_idempotency_digest(left_digest="aa" * 32, right_digest="aa" * 32) == "same_digest"
    )
    assert (
        classify_idempotency_digest(left_digest="aa" * 32, right_digest="bb" * 32)
        == "different_digest"
    )
    assert classify_idempotency_digest(left_digest="nope", right_digest="aa" * 32) == "invalid"
    assert classify_idempotency_digest(left_digest="aa" * 32, right_digest="aa" * 32) not in {
        "replay",
        "conflict",
    }


def test_invalid() -> None:
    with pytest.raises(OperationIdempotencyError, match=INVALID):
        declare_operation_idempotency(
            chat_id="c",
            operation_key="k",
            request_digest="zz",
            expected_message_id="m",
        )
    with pytest.raises(OperationIdempotencyError, match=INVALID):
        declare_operation_idempotency(
            chat_id="c" * (MAX_ID + 1),
            operation_key="k",
            request_digest="ab" * 32,
            expected_message_id="m",
        )
    with pytest.raises(OperationIdempotencyError, match=INVALID):
        declare_operation_idempotency(
            chat_id="c1",
            operation_key="k",
            request_digest="g" * DIGEST_LEN,
            expected_message_id="m",
        )


def test_public_constructor_cannot_authorize() -> None:
    with pytest.raises(OperationIdempotencyError, match=INVALID):
        OperationIdempotencyDeclarationV1()
    with pytest.raises(TypeError):
        OperationIdempotencyDeclarationV1(
            schema="lm-atelier-operation-idempotency-declaration-v1",
            schema_version=1,
            execution_authorized=True,
        )
