"""Pure durable operation idempotency declaration (items 39/40).

Records requested replay/conflict policy for a later writer-fenced
event. This module does not persist or grant commit authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-operation-idempotency-declaration-v1"
SCHEMA_VERSION: Final = 1
INVALID: Final = "operation idempotency declaration is invalid"
MAX_ID: Final = 128
DIGEST_LEN: Final = 64
DIGEST_CHARS: Final = frozenset("0123456789abcdef")
RequestedReplayRule = Literal["same_digest_is_replay"]
RequestedConflictRule = Literal["different_digest_is_conflict"]
DigestCompareResult = Literal["same_digest", "different_digest", "invalid"]
_IDEMPOTENCY_WITNESS = object()


class OperationIdempotencyError(ValueError):
    """Fixed non-echoing refusal for invalid idempotency declaration facts."""


@dataclass(frozen=True, slots=True)
class OperationIdempotencyDeclarationV1:
    schema: Literal["lm-atelier-operation-idempotency-declaration-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    chat_id: str = field(init=False)
    operation_key: str = field(init=False)
    request_digest: str = field(init=False)
    expected_message_id: str = field(init=False)
    expected_revision_id: str | None = field(init=False)
    requested_replay_rule: RequestedReplayRule = field(init=False)
    requested_conflict_rule: RequestedConflictRule = field(init=False)
    repository_event_persisted: Literal[False] = field(init=False)
    execution_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise OperationIdempotencyError(INVALID)


def _invalid() -> NoReturn:
    raise OperationIdempotencyError(INVALID)


def _require_id(value: object) -> str:
    if type(value) is not str:
        _invalid()
    if not value or len(value) > MAX_ID:
        _invalid()
    if value.strip() != value or any(ch.isspace() for ch in value):
        _invalid()
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        _invalid()
    return value


def _require_digest(value: object) -> str:
    if type(value) is not str:
        _invalid()
    if len(value) != DIGEST_LEN:
        _invalid()
    if any(ch not in DIGEST_CHARS for ch in value):
        _invalid()
    return value


def _idempotency_from_evaluator(
    *,
    witness: object,
    chat_id: str,
    operation_key: str,
    request_digest: str,
    expected_message_id: str,
    expected_revision_id: str | None,
) -> OperationIdempotencyDeclarationV1:
    if witness is not _IDEMPOTENCY_WITNESS:
        _invalid()
    declaration = object.__new__(OperationIdempotencyDeclarationV1)
    object.__setattr__(declaration, "schema", SCHEMA_ID)
    object.__setattr__(declaration, "schema_version", SCHEMA_VERSION)
    object.__setattr__(declaration, "chat_id", chat_id)
    object.__setattr__(declaration, "operation_key", operation_key)
    object.__setattr__(declaration, "request_digest", request_digest)
    object.__setattr__(declaration, "expected_message_id", expected_message_id)
    object.__setattr__(declaration, "expected_revision_id", expected_revision_id)
    object.__setattr__(declaration, "requested_replay_rule", "same_digest_is_replay")
    object.__setattr__(declaration, "requested_conflict_rule", "different_digest_is_conflict")
    object.__setattr__(declaration, "repository_event_persisted", False)
    object.__setattr__(declaration, "execution_authorized", False)
    return declaration


def declare_operation_idempotency(
    *,
    chat_id: object,
    operation_key: object,
    request_digest: object,
    expected_message_id: object,
    expected_revision_id: object = None,
) -> OperationIdempotencyDeclarationV1:
    """Record requested idempotency rules without persisting an event."""
    revision = None if expected_revision_id is None else _require_id(expected_revision_id)
    return _idempotency_from_evaluator(
        witness=_IDEMPOTENCY_WITNESS,
        chat_id=_require_id(chat_id),
        operation_key=_require_id(operation_key),
        request_digest=_require_digest(request_digest),
        expected_message_id=_require_id(expected_message_id),
        expected_revision_id=revision,
    )


def classify_idempotency_digest(
    *,
    left_digest: object,
    right_digest: object,
) -> DigestCompareResult:
    """Compare two caller-supplied digests without touching storage."""
    try:
        left = _require_digest(left_digest)
        right = _require_digest(right_digest)
    except OperationIdempotencyError:
        return "invalid"
    if left == right:
        return "same_digest"
    return "different_digest"
