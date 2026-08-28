"""Pure conversation-search visibility and index eligibility.

Accepts caller-supplied boolean and identity facts only. No DB, FTS, API,
filesystem, or graph mutation. Never grants index rebuild, query execution,
or content restoration authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-search-visibility-v1"
SCHEMA_VERSION: Final = 1
INVALID_VISIBILITY: Final = "search visibility facts are invalid"

VisibilityCode = Literal[
    "eligible",
    "deny_not_transcript_visible",
    "deny_content_removed",
    "deny_private_session",
    "deny_secret_payload",
    "deny_helper_session",
]


class SearchVisibilityError(ValueError):
    """Fixed non-echoing refusal for invalid visibility facts."""


_EVALUATOR_WITNESS = object()


@dataclass(frozen=True, slots=True)
class SearchVisibilityV1:
    schema: Literal["lm-atelier-search-visibility-v1"]
    schema_version: Literal[1]
    message_id_bound: bool = field(init=False)
    eligible: bool = field(init=False)
    code: VisibilityCode = field(init=False)
    may_index_body: bool = field(init=False)
    may_emit_snippet: bool = field(init=False)
    may_rank: bool = field(init=False)
    index_rebuild_authorized: Literal[False] = field(default=False, init=False)
    fts_write_authorized: Literal[False] = field(default=False, init=False)
    query_execution_authorized: Literal[False] = field(default=False, init=False)
    content_restore_authorized: Literal[False] = field(default=False, init=False)

    def __post_init__(self) -> None:
        raise SearchVisibilityError(INVALID_VISIBILITY)


def _invalid() -> NoReturn:
    raise SearchVisibilityError(INVALID_VISIBILITY)


def _visibility_from_evaluator(*, witness: object, code: VisibilityCode) -> SearchVisibilityV1:
    if witness is not _EVALUATOR_WITNESS:
        _invalid()
    eligible = code == "eligible"
    decision = object.__new__(SearchVisibilityV1)
    object.__setattr__(decision, "schema", SCHEMA_ID)
    object.__setattr__(decision, "schema_version", SCHEMA_VERSION)
    object.__setattr__(decision, "message_id_bound", True)
    object.__setattr__(decision, "eligible", eligible)
    object.__setattr__(decision, "code", code)
    object.__setattr__(decision, "may_index_body", eligible)
    object.__setattr__(decision, "may_emit_snippet", eligible)
    object.__setattr__(decision, "may_rank", eligible)
    object.__setattr__(decision, "index_rebuild_authorized", False)
    object.__setattr__(decision, "fts_write_authorized", False)
    object.__setattr__(decision, "query_execution_authorized", False)
    object.__setattr__(decision, "content_restore_authorized", False)
    return decision


def _require_bool(value: object) -> bool:
    if type(value) is not bool:
        _invalid()
    return value


def _require_message_id(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any(ch.isspace() for ch in value)
    ):
        _invalid()
    if len(value) > 128:
        _invalid()
    return value


def evaluate_search_visibility(
    *,
    message_id: object,
    transcript_visible: object,
    content_removed: object,
    private_session: object,
    helper_session: object,
    secret_payload: object,
) -> SearchVisibilityV1:
    """Decide whether a message body may appear in search projection facts.

    Callers supply already-resolved domain flags, such as whether the
    message body carries a content tombstone. This function only
    classifies eligibility; it never reads storage.
    """
    mid = _require_message_id(message_id)
    tv = _require_bool(transcript_visible)
    removed = _require_bool(content_removed)
    private = _require_bool(private_session)
    helper = _require_bool(helper_session)
    secret = _require_bool(secret_payload)
    _ = mid  # identity must be bound; value not echoed in code

    if not tv:
        code: VisibilityCode = "deny_not_transcript_visible"
    elif removed:
        code = "deny_content_removed"
    elif private:
        code = "deny_private_session"
    elif helper:
        code = "deny_helper_session"
    elif secret:
        code = "deny_secret_payload"
    else:
        code = "eligible"

    return _visibility_from_evaluator(witness=_EVALUATOR_WITNESS, code=code)


def filter_indexable_bodies(
    rows: object,
) -> tuple[tuple[str, str], ...]:
    """Return only (message_id, body) rows that pass visibility.

    Each row is a mapping with keys:
      message_id, body, transcript_visible, content_removed,
      private_session, helper_session, secret_payload
    Non-mapping rows or missing keys refuse the whole batch.
    """
    if type(rows) is not list and type(rows) is not tuple:
        _invalid()
    required = frozenset(
        {
            "message_id",
            "body",
            "transcript_visible",
            "content_removed",
            "private_session",
            "helper_session",
            "secret_payload",
        }
    )
    out: list[tuple[str, str]] = []
    for row in rows:
        if type(row) is not dict or len(row) > 16:
            _invalid()
        owned: dict[str, object] = {}
        for key, value in row.items():
            if type(key) is not str:
                _invalid()
            owned[key] = value
        if not required.issubset(owned):
            _invalid()
        body = owned["body"]
        if type(body) is not str:
            _invalid()
        decision = evaluate_search_visibility(
            message_id=owned["message_id"],
            transcript_visible=owned["transcript_visible"],
            content_removed=owned["content_removed"],
            private_session=owned["private_session"],
            helper_session=owned["helper_session"],
            secret_payload=owned["secret_payload"],
        )
        if decision.eligible:
            mid = owned["message_id"]
            if type(mid) is not str:
                _invalid()
            out.append((mid, body))
    return tuple(out)
