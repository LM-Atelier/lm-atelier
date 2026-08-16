"""Pure prior-turn edit declaration (item 40).

Records caller-supplied edit intent. This module does not accept, persist,
or queue work.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-prior-turn-edit-declaration-v1"
SCHEMA_VERSION: Final = 1
INVALID_DECLARATION: Final = "prior turn edit declaration is invalid"
MAX_ID: Final = 128
MAX_ATTACHMENT_COUNT: Final = 32
MAX_ATTACHMENT_BYTES: Final = 1024
DIGEST_LEN: Final = 64
DIGEST_CHARS: Final = frozenset("0123456789abcdef")
AttachmentMode = Literal["omit_inherit", "empty_clear", "replace"]
_REQUIRED: Final = frozenset(
    {
        "chat_id",
        "source_message_id",
        "parent_message_id",
        "idempotency_key",
        "replacement_content_digest",
        "activate_branch",
    }
)
_OPTIONAL: Final = frozenset(
    {
        "source_revision_id",
        "geometry_digest",
        "workflow_digest",
        "attachment_mode",
        "attachment_ids",
    }
)
_EDIT_WITNESS = object()


class PriorTurnEditDeclarationError(ValueError):
    """Fixed non-echoing refusal for invalid edit declaration facts."""


@dataclass(frozen=True, slots=True)
class PriorTurnEditDeclarationV1:
    schema: Literal["lm-atelier-prior-turn-edit-declaration-v1"] = field(init=False)
    schema_version: Literal[1] = field(init=False)
    chat_id: str = field(init=False)
    source_message_id: str = field(init=False)
    parent_message_id: str = field(init=False)
    source_revision_id: str | None = field(init=False)
    idempotency_key: str = field(init=False)
    replacement_content_digest: str = field(init=False)
    geometry_digest: str | None = field(init=False)
    workflow_digest: str | None = field(init=False)
    attachment_mode: AttachmentMode = field(init=False)
    attachment_ids: tuple[str, ...] = field(init=False)
    activate_branch: Literal[False] = field(init=False)
    accepted: Literal[False] = field(init=False)
    persisted: Literal[False] = field(init=False)
    repository_snapshot_verified: Literal[False] = field(init=False)
    execution_authorized: Literal[False] = field(init=False)
    queue_safe: Literal[False] = field(init=False)

    def __post_init__(self) -> None:
        raise PriorTurnEditDeclarationError(INVALID_DECLARATION)


def _invalid() -> NoReturn:
    raise PriorTurnEditDeclarationError(INVALID_DECLARATION)


def _require_id(value: object) -> str:
    if type(value) is not str:
        _invalid()
    if not value or len(value) > MAX_ID:
        _invalid()
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        _invalid()
    for ch in value:
        category = unicodedata.category(ch)
        if category == "Cc" or category == "Cf":
            _invalid()
    if value.strip() != value or any(ch.isspace() for ch in value):
        _invalid()
    if "/" in value or "\\" in value:
        _invalid()
    if value in {".", ".."}:
        _invalid()
    return value


def _utf8_size(value: str) -> int:
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        _invalid()
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        _invalid()


def _require_digest(value: object) -> str:
    if type(value) is not str:
        _invalid()
    if len(value) != DIGEST_LEN:
        _invalid()
    if any(ch not in DIGEST_CHARS for ch in value):
        _invalid()
    return value


def _require_attachment_ids(raw_ids: object) -> tuple[str, ...]:
    if type(raw_ids) is not list and type(raw_ids) is not tuple:
        _invalid()
    if len(raw_ids) > MAX_ATTACHMENT_COUNT:
        _invalid()
    total = 0
    for item in raw_ids:
        if type(item) is not str:
            _invalid()
        total += _utf8_size(item)
        if total > MAX_ATTACHMENT_BYTES:
            _invalid()
    collected: list[str] = []
    for item in raw_ids:
        collected.append(_require_id(item))
    ids = tuple(collected)
    if len(ids) != len(set(ids)):
        _invalid()
    return ids


def _require_mode(value: object) -> AttachmentMode:
    if type(value) is not str:
        _invalid()
    if not value or len(value) > MAX_ID:
        _invalid()
    planned: AttachmentMode
    if value == "omit_inherit":
        planned = "omit_inherit"
    elif value == "empty_clear":
        planned = "empty_clear"
    elif value == "replace":
        planned = "replace"
    else:
        _invalid()
    return planned


def _edit_from_evaluator(
    *,
    witness: object,
    chat_id: str,
    source_message_id: str,
    parent_message_id: str,
    source_revision_id: str | None,
    idempotency_key: str,
    replacement_content_digest: str,
    geometry_digest: str | None,
    workflow_digest: str | None,
    attachment_mode: AttachmentMode,
    attachment_ids: tuple[str, ...],
) -> PriorTurnEditDeclarationV1:
    if witness is not _EDIT_WITNESS:
        _invalid()
    declaration = object.__new__(PriorTurnEditDeclarationV1)
    object.__setattr__(declaration, "schema", SCHEMA_ID)
    object.__setattr__(declaration, "schema_version", SCHEMA_VERSION)
    object.__setattr__(declaration, "chat_id", chat_id)
    object.__setattr__(declaration, "source_message_id", source_message_id)
    object.__setattr__(declaration, "parent_message_id", parent_message_id)
    object.__setattr__(declaration, "source_revision_id", source_revision_id)
    object.__setattr__(declaration, "idempotency_key", idempotency_key)
    object.__setattr__(declaration, "replacement_content_digest", replacement_content_digest)
    object.__setattr__(declaration, "geometry_digest", geometry_digest)
    object.__setattr__(declaration, "workflow_digest", workflow_digest)
    object.__setattr__(declaration, "attachment_mode", attachment_mode)
    object.__setattr__(declaration, "attachment_ids", attachment_ids)
    object.__setattr__(declaration, "activate_branch", False)
    object.__setattr__(declaration, "accepted", False)
    object.__setattr__(declaration, "persisted", False)
    object.__setattr__(declaration, "repository_snapshot_verified", False)
    object.__setattr__(declaration, "execution_authorized", False)
    object.__setattr__(declaration, "queue_safe", False)
    return declaration


def declare_prior_turn_edit(facts: object) -> PriorTurnEditDeclarationV1:
    """Parse edit intent into a declaration without accepting or queuing."""
    if type(facts) is not dict:
        _invalid()
    keys: set[str] = set()
    for raw_key in facts:
        if type(raw_key) is not str:
            _invalid()
        key = raw_key
        keys.add(key)
    if not _REQUIRED.issubset(keys):
        _invalid()
    if not keys.issubset(_REQUIRED | _OPTIONAL):
        _invalid()
    activate_key = "activate_branch"
    if facts[activate_key] is not False:
        _invalid()
    mode_key = "attachment_mode"
    mode = _require_mode(facts[mode_key]) if mode_key in keys else "omit_inherit"
    ids_key = "attachment_ids"
    ids = _require_attachment_ids(facts[ids_key]) if ids_key in keys else ()
    if mode == "replace" and not ids:
        _invalid()
    if mode == "empty_clear" and ids:
        _invalid()
    if mode == "omit_inherit" and ids:
        _invalid()
    geo_key = "geometry_digest"
    if geo_key in keys and facts[geo_key] is not None:
        geometry = _require_digest(facts[geo_key])
    else:
        geometry = None
    workflow_key = "workflow_digest"
    if workflow_key in keys and facts[workflow_key] is not None:
        workflow = _require_digest(facts[workflow_key])
    else:
        workflow = None
    revision_key = "source_revision_id"
    if revision_key in keys and facts[revision_key] is not None:
        revision = _require_id(facts[revision_key])
    else:
        revision = None
    chat_key = "chat_id"
    source_key = "source_message_id"
    parent_key = "parent_message_id"
    idempotency_key_name = "idempotency_key"
    digest_key = "replacement_content_digest"
    return _edit_from_evaluator(
        witness=_EDIT_WITNESS,
        chat_id=_require_id(facts[chat_key]),
        source_message_id=_require_id(facts[source_key]),
        parent_message_id=_require_id(facts[parent_key]),
        source_revision_id=revision,
        idempotency_key=_require_id(facts[idempotency_key_name]),
        replacement_content_digest=_require_digest(facts[digest_key]),
        geometry_digest=geometry,
        workflow_digest=workflow,
        attachment_mode=mode,
        attachment_ids=ids,
    )
