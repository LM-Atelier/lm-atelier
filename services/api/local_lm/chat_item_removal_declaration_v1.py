"""Pure chat-item removal declaration.

Records a candidate tombstone plan from caller-supplied facts. This
module does not authorize mutation or verify a repository snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-chat-item-removal-declaration-v1"
SCHEMA_VERSION: Final = 1
INVALID_REMOVAL: Final = "chat item removal declaration is invalid"
MAX_PARTS: Final = 256
MAX_ID: Final = 128
CandidateCode = Literal[
    "candidate_content_tombstone",
    "refuse_active_work",
    "refuse_already_removed",
    "refuse_identity_mismatch",
]
_REQUIRED: Final = frozenset(
    {
        "message_id",
        "expected_message_id",
        "event_revision_id",
        "expected_event_revision_id",
        "idempotency_key",
        "has_replies",
        "content_already_removed",
        "chat_has_active_work",
        "parts_count",
        "references_count",
        "revision_parts_count",
        "source_content_backs_regeneration",
        "same_chat",
    }
)
_DECLARATION_WITNESS = object()


class ChatItemRemovalDeclarationError(ValueError):
    """Fixed non-echoing refusal for invalid removal declaration facts."""


@dataclass(frozen=True, slots=True)
class ChatItemRemovalDeclarationV1:
    schema: Literal["lm-atelier-chat-item-removal-declaration-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    message_ids_match: bool = field(init=False)
    event_revisions_match: bool = field(init=False)
    same_chat: bool = field(init=False)
    has_replies: bool = field(init=False)
    content_already_removed: bool = field(init=False)
    chat_has_active_work: bool = field(init=False)
    parts_count: int = field(init=False)
    references_count: int = field(init=False)
    revision_parts_count: int = field(init=False)
    source_content_backs_regeneration: bool = field(init=False)
    candidate_code: CandidateCode = field(init=False)
    planned_preserve_message_id: bool = field(init=False)
    planned_preserve_reply_edges: bool = field(init=False)
    planned_block_revision_repopulation: bool = field(init=False)
    planned_detach_payload_parts: bool = field(init=False)
    repository_snapshot_verified: Literal[False] = field(init=False)
    message_id_bound: Literal[False] = field(init=False)
    allowed: Literal[False] = field(init=False)
    execution_authorized: Literal[False] = field(init=False)
    future_context_omission_verified: Literal[False] = field(init=False)
    forensic_erasure: Literal[False] = field(init=False)
    cascade_delete_replies: Literal[False] = field(init=False)
    physical_node_deleted: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise ChatItemRemovalDeclarationError(INVALID_REMOVAL)


def _invalid() -> NoReturn:
    raise ChatItemRemovalDeclarationError(INVALID_REMOVAL)


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


def _require_bool(value: object) -> bool:
    if type(value) is not bool:
        _invalid()
    return value


def _require_nonneg_int(value: object, *, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        _invalid()
    return value


def _declaration_from_evaluator(
    *,
    witness: object,
    message_ids_match: bool,
    event_revisions_match: bool,
    same_chat: bool,
    has_replies: bool,
    content_already_removed: bool,
    chat_has_active_work: bool,
    parts_count: int,
    references_count: int,
    revision_parts_count: int,
    source_content_backs_regeneration: bool,
    candidate_code: CandidateCode,
    planned_preserve_message_id: bool,
    planned_preserve_reply_edges: bool,
    planned_block_revision_repopulation: bool,
    planned_detach_payload_parts: bool,
) -> ChatItemRemovalDeclarationV1:
    if witness is not _DECLARATION_WITNESS:
        _invalid()
    declaration = object.__new__(ChatItemRemovalDeclarationV1)
    object.__setattr__(declaration, "schema", SCHEMA_ID)
    object.__setattr__(declaration, "schema_version", SCHEMA_VERSION)
    object.__setattr__(declaration, "message_ids_match", message_ids_match)
    object.__setattr__(declaration, "event_revisions_match", event_revisions_match)
    object.__setattr__(declaration, "same_chat", same_chat)
    object.__setattr__(declaration, "has_replies", has_replies)
    object.__setattr__(declaration, "content_already_removed", content_already_removed)
    object.__setattr__(declaration, "chat_has_active_work", chat_has_active_work)
    object.__setattr__(declaration, "parts_count", parts_count)
    object.__setattr__(declaration, "references_count", references_count)
    object.__setattr__(declaration, "revision_parts_count", revision_parts_count)
    object.__setattr__(
        declaration, "source_content_backs_regeneration", source_content_backs_regeneration
    )
    object.__setattr__(declaration, "candidate_code", candidate_code)
    object.__setattr__(declaration, "planned_preserve_message_id", planned_preserve_message_id)
    object.__setattr__(declaration, "planned_preserve_reply_edges", planned_preserve_reply_edges)
    object.__setattr__(
        declaration, "planned_block_revision_repopulation", planned_block_revision_repopulation
    )
    object.__setattr__(declaration, "planned_detach_payload_parts", planned_detach_payload_parts)
    object.__setattr__(declaration, "repository_snapshot_verified", False)
    object.__setattr__(declaration, "message_id_bound", False)
    object.__setattr__(declaration, "allowed", False)
    object.__setattr__(declaration, "execution_authorized", False)
    object.__setattr__(declaration, "future_context_omission_verified", False)
    object.__setattr__(declaration, "forensic_erasure", False)
    object.__setattr__(declaration, "cascade_delete_replies", False)
    object.__setattr__(declaration, "physical_node_deleted", False)
    return declaration


def declare_chat_item_removal(facts: object) -> ChatItemRemovalDeclarationV1:
    """Build a candidate tombstone plan without authorizing mutation."""
    if type(facts) is not dict:
        _invalid()
    keys: set[str] = set()
    for raw_key in facts:
        if type(raw_key) is not str:
            _invalid()
        key = raw_key
        keys.add(key)
    if keys != _REQUIRED:
        _invalid()
    message_key = "message_id"
    expected_message_key = "expected_message_id"
    event_key = "event_revision_id"
    expected_event_key = "expected_event_revision_id"
    idempotency_key_name = "idempotency_key"
    has_replies_key = "has_replies"
    already_removed_key = "content_already_removed"
    active_work_key = "chat_has_active_work"
    parts_key = "parts_count"
    references_key = "references_count"
    revision_parts_key = "revision_parts_count"
    regeneration_key = "source_content_backs_regeneration"
    same_chat_key = "same_chat"
    message_id = _require_id(facts[message_key])
    expected_message_id = _require_id(facts[expected_message_key])
    event_revision_id = _require_id(facts[event_key])
    expected_event_revision_id = _require_id(facts[expected_event_key])
    _require_id(facts[idempotency_key_name])
    has_replies = _require_bool(facts[has_replies_key])
    content_already_removed = _require_bool(facts[already_removed_key])
    chat_has_active_work = _require_bool(facts[active_work_key])
    parts_count = _require_nonneg_int(facts[parts_key], maximum=MAX_PARTS)
    references_count = _require_nonneg_int(facts[references_key], maximum=MAX_PARTS)
    revision_parts_count = _require_nonneg_int(facts[revision_parts_key], maximum=MAX_PARTS)
    source_content_backs_regeneration = _require_bool(facts[regeneration_key])
    same_chat = _require_bool(facts[same_chat_key])
    ids_match = message_id == expected_message_id
    revs_match = event_revision_id == expected_event_revision_id
    planned: CandidateCode
    if not (same_chat and ids_match and revs_match):
        planned = "refuse_identity_mismatch"
    elif content_already_removed:
        planned = "refuse_already_removed"
    elif chat_has_active_work:
        planned = "refuse_active_work"
    else:
        planned = "candidate_content_tombstone"
    is_candidate = planned == "candidate_content_tombstone"
    return _declaration_from_evaluator(
        witness=_DECLARATION_WITNESS,
        message_ids_match=ids_match,
        event_revisions_match=revs_match,
        same_chat=same_chat,
        has_replies=has_replies,
        content_already_removed=content_already_removed,
        chat_has_active_work=chat_has_active_work,
        parts_count=parts_count,
        references_count=references_count,
        revision_parts_count=revision_parts_count,
        source_content_backs_regeneration=source_content_backs_regeneration,
        candidate_code=planned,
        planned_preserve_message_id=is_candidate,
        planned_preserve_reply_edges=is_candidate,
        planned_block_revision_repopulation=is_candidate,
        planned_detach_payload_parts=is_candidate,
    )
