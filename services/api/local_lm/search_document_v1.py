"""Pure search document facts for conversation search projection.

Assembles indexable document rows from caller-supplied visible text and flags.
No FTS write, DB I/O, or rebuild authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

from .search_visibility_v1 import SearchVisibilityError, evaluate_search_visibility

SCHEMA_ID: Final = "lm-atelier-search-document-v1"
SCHEMA_VERSION: Final = 1
INVALID_DOCUMENT: Final = "search document facts are invalid"
MAX_BODY: Final = 65536
MAX_ID: Final = 40
PROJECTION_SCHEMA: Final = "conversation-fts-v1"
ROLES: Final = frozenset({"user", "assistant", "system"})


class SearchDocumentError(ValueError):
    """Fixed non-echoing refusal for invalid document facts."""


_DOCUMENT_WITNESS = object()


@dataclass(frozen=True, slots=True)
class SearchDocumentV1:
    schema: Literal["lm-atelier-search-document-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    projection_schema: Literal["conversation-fts-v1"] = field(init=False)
    message_id: str = field(init=False)
    chat_id: str = field(init=False)
    role: str = field(init=False)
    body: str = field(init=False)
    has_media: bool = field(init=False)
    selected_response_revision_id: str | None = field(init=False)
    created_at_unix: int | None = field(init=False)
    eligible: Literal[True] = field(init=False)
    fts_write_authorized: Literal[False] = field(init=False)
    index_rebuild_authorized: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise SearchDocumentError(INVALID_DOCUMENT)


def _document_from_evaluator(
    *,
    witness: object,
    message_id: str,
    chat_id: str,
    role: str,
    body: str,
    has_media: bool,
    selected_response_revision_id: str | None,
    created_at_unix: int | None,
) -> SearchDocumentV1:
    if witness is not _DOCUMENT_WITNESS:
        _invalid()
    document = object.__new__(SearchDocumentV1)
    object.__setattr__(document, "schema", SCHEMA_ID)
    object.__setattr__(document, "schema_version", SCHEMA_VERSION)
    object.__setattr__(document, "projection_schema", PROJECTION_SCHEMA)
    object.__setattr__(document, "message_id", message_id)
    object.__setattr__(document, "chat_id", chat_id)
    object.__setattr__(document, "role", role)
    object.__setattr__(document, "body", body)
    object.__setattr__(document, "has_media", has_media)
    object.__setattr__(document, "selected_response_revision_id", selected_response_revision_id)
    object.__setattr__(document, "created_at_unix", created_at_unix)
    object.__setattr__(document, "eligible", True)
    object.__setattr__(document, "fts_write_authorized", False)
    object.__setattr__(document, "index_rebuild_authorized", False)
    return document


def _invalid() -> NoReturn:
    raise SearchDocumentError(INVALID_DOCUMENT)


def _require_id(value: object) -> str:
    if type(value) is not str or not value:
        _invalid()
    if value.strip() != value or any(ch.isspace() for ch in value):
        _invalid()
    if len(value) > MAX_ID:
        _invalid()
    return value


def _require_role(value: object) -> str:
    if type(value) is not str or value not in ROLES:
        _invalid()
    return value


def build_search_document(
    *,
    message_id: object,
    chat_id: object,
    role: object,
    body: object,
    has_media: object,
    transcript_visible: object,
    content_removed: object,
    private_session: object,
    helper_session: object,
    secret_payload: object,
    selected_response_revision_id: object = None,
    created_at_unix: object = None,
) -> SearchDocumentV1 | None:
    """Return a document when visibility allows; None when ineligible.

    Refuses malformed identity/body/role. Does not write an index.
    """
    mid = _require_id(message_id)
    cid = _require_id(chat_id)
    role_s = _require_role(role)
    if type(body) is not str or len(body) > MAX_BODY:
        _invalid()
    if type(has_media) is not bool:
        _invalid()
    try:
        vis = evaluate_search_visibility(
            message_id=mid,
            transcript_visible=transcript_visible,
            content_removed=content_removed,
            private_session=private_session,
            helper_session=helper_session,
            secret_payload=secret_payload,
        )
    except SearchVisibilityError as exc:
        raise SearchDocumentError(INVALID_DOCUMENT) from exc
    if not vis.eligible:
        return None
    rev = None
    if selected_response_revision_id is not None:
        rev = _require_id(selected_response_revision_id)
    ts = None
    if created_at_unix is not None:
        if type(created_at_unix) is not int or created_at_unix < 0:
            _invalid()
        ts = created_at_unix
    return _document_from_evaluator(
        witness=_DOCUMENT_WITNESS,
        message_id=mid,
        chat_id=cid,
        role=role_s,
        body=body,
        has_media=has_media,
        selected_response_revision_id=rev,
        created_at_unix=ts,
    )
