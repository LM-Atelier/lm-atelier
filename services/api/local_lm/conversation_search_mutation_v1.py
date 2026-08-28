"""Pure conversation-search mutation ledger facts.

Records upsert/remove/selected-revision identity only. Never carries raw text
or grants FTS write / index rebuild authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

from .search_text_v1 import MAX_SEARCH_TOKEN_CHARS, require_bounded_exact_str

SCHEMA_ID: Final = "lm-atelier-conversation-search-mutation-v1"
SCHEMA_VERSION: Final = 1
INVALID_MUTATION: Final = "search mutation facts are invalid"
MAX_ID: Final = MAX_SEARCH_TOKEN_CHARS
MAX_OPERATION_CHARS: Final = MAX_SEARCH_TOKEN_CHARS
MAX_SEQUENCE: Final = 2**31 - 1
MutationOp = Literal["upsert", "remove", "selected_revision_changed"]
OPS: Final = frozenset({"upsert", "remove", "selected_revision_changed"})
_MUTATION_WITNESS = object()


class SearchMutationError(ValueError):
    """Fixed non-echoing refusal for invalid mutation facts."""


@dataclass(frozen=True, slots=True)
class ConversationSearchMutationV1:
    schema: Literal["lm-atelier-conversation-search-mutation-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    sequence: int = field(init=False)
    message_id: str = field(init=False)
    chat_id: str = field(init=False)
    operation: MutationOp = field(init=False)
    content_revision: str = field(init=False)
    fts_write_authorized: Literal[False] = field(init=False)
    index_rebuild_authorized: Literal[False] = field(init=False)
    raw_text_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise SearchMutationError(INVALID_MUTATION)


def _invalid() -> NoReturn:
    raise SearchMutationError(INVALID_MUTATION)


def _require_id(value: object) -> str:
    text = require_bounded_exact_str(value, max_len=MAX_ID, refuse=_invalid)
    if text.strip() != text or any(ch.isspace() for ch in text):
        _invalid()
    return text


def _mutation_from_evaluator(
    *,
    witness: object,
    sequence: int,
    message_id: str,
    chat_id: str,
    operation: MutationOp,
    content_revision: str,
) -> ConversationSearchMutationV1:
    if witness is not _MUTATION_WITNESS:
        _invalid()
    mutation = object.__new__(ConversationSearchMutationV1)
    object.__setattr__(mutation, "schema", SCHEMA_ID)
    object.__setattr__(mutation, "schema_version", SCHEMA_VERSION)
    object.__setattr__(mutation, "sequence", sequence)
    object.__setattr__(mutation, "message_id", message_id)
    object.__setattr__(mutation, "chat_id", chat_id)
    object.__setattr__(mutation, "operation", operation)
    object.__setattr__(mutation, "content_revision", content_revision)
    object.__setattr__(mutation, "fts_write_authorized", False)
    object.__setattr__(mutation, "index_rebuild_authorized", False)
    object.__setattr__(mutation, "raw_text_authorized", False)
    return mutation


def declare_search_mutation(
    *,
    sequence: object,
    message_id: object,
    chat_id: object,
    operation: object,
    content_revision: object,
) -> ConversationSearchMutationV1:
    """Bind a mutation identity without granting index-write authority."""
    if type(sequence) is not int or sequence < 1 or sequence > MAX_SEQUENCE:
        _invalid()
    operation_text = require_bounded_exact_str(
        operation, max_len=MAX_OPERATION_CHARS, refuse=_invalid
    )
    if operation_text not in OPS:
        _invalid()
    planned: MutationOp
    if operation_text == "upsert":
        planned = "upsert"
    elif operation_text == "remove":
        planned = "remove"
    else:
        planned = "selected_revision_changed"
    return _mutation_from_evaluator(
        witness=_MUTATION_WITNESS,
        sequence=sequence,
        message_id=_require_id(message_id),
        chat_id=_require_id(chat_id),
        operation=planned,
        content_revision=_require_id(content_revision),
    )
