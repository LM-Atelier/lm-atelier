from __future__ import annotations

import inspect
from typing import Any

import pytest

from local_lm.search_document_v1 import (
    INVALID_DOCUMENT,
    MAX_BODY,
    MAX_ID,
    SearchDocumentError,
    SearchDocumentV1,
    build_search_document,
)


def test_eligible_document() -> None:
    doc = build_search_document(
        message_id="m1",
        chat_id="c1",
        role="user",
        body="hello world",
        has_media=False,
        transcript_visible=True,
        content_removed=False,
        private_session=False,
        helper_session=False,
        secret_payload=False,
        created_at_unix=100,
    )
    assert doc is not None
    assert doc.eligible is True
    assert doc.projection_schema == "conversation-fts-v1"
    assert doc.fts_write_authorized is False
    assert doc.index_rebuild_authorized is False
    assert doc.body == "hello world"


def test_ineligible_returns_none() -> None:
    assert (
        build_search_document(
            message_id="m1",
            chat_id="c1",
            role="assistant",
            body="x",
            has_media=True,
            transcript_visible=True,
            content_removed=True,
            private_session=False,
            helper_session=False,
            secret_payload=False,
        )
        is None
    )
    assert (
        build_search_document(
            message_id="m1",
            chat_id="c1",
            role="user",
            body="x",
            has_media=False,
            transcript_visible=False,
            content_removed=False,
            helper_session=False,
            secret_payload=False,
            private_session=True,
        )
        is None
    )


def test_invalid_and_hostile() -> None:
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(
            message_id="bad id",
            chat_id="c",
            role="user",
            body="x",
            has_media=False,
            transcript_visible=True,
            content_removed=False,
            private_session=False,
            helper_session=False,
            secret_payload=False,
        )
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(
            message_id="m",
            chat_id="c",
            role="bot",
            body="x",
            has_media=False,
            transcript_visible=True,
            content_removed=False,
            private_session=False,
            helper_session=False,
            secret_payload=False,
        )
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(
            message_id="m",
            chat_id="c",
            role="user",
            body="x",
            has_media="no",
            transcript_visible=True,
            content_removed=False,
            private_session=False,
            helper_session=False,
            secret_payload=False,
        )
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(
            message_id="m",
            chat_id="c",
            role="user",
            body="x" * (MAX_BODY + 1),
            has_media=False,
            transcript_visible=True,
            content_removed=False,
            private_session=False,
            helper_session=False,
            secret_payload=False,
        )
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(
            message_id="m",
            chat_id="c",
            role="user",
            body="x",
            has_media=False,
            transcript_visible=True,
            content_removed=False,
            private_session=False,
            helper_session=False,
            secret_payload=False,
            created_at_unix=True,
        )

    class HostileStr(str):
        def strip(self, *args, **kwargs):
            raise RuntimeError("private attacker detail")

    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT) as caught:
        build_search_document(
            message_id=HostileStr("m1"),
            chat_id="c1",
            role="user",
            body="hello",
            has_media=False,
            transcript_visible=True,
            content_removed=False,
            private_session=False,
            helper_session=False,
            secret_payload=False,
        )
    assert "private attacker detail" not in str(caught.value)


def test_refuses_empty_non_string_and_oversize_ids() -> None:
    common = {
        "role": "user",
        "body": "x",
        "has_media": False,
        "transcript_visible": True,
        "content_removed": False,
        "private_session": False,
        "helper_session": False,
        "secret_payload": False,
    }
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(message_id="", chat_id="c1", **common)
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(message_id="m1", chat_id="", **common)
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(message_id=1, chat_id="c1", **common)
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(message_id="x" * (MAX_ID + 1), chat_id="c1", **common)


def test_public_constructor_cannot_mint_eligible_document() -> None:
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        SearchDocumentV1()
    with pytest.raises(TypeError):
        SearchDocumentV1(
            schema="wrong-schema",
            schema_version=1,
            projection_schema="conversation-fts-v1",
            message_id="",
            chat_id="",
            role="attacker",
            body="secret",
            has_media=False,
            selected_response_revision_id=None,
            created_at_unix=None,
        )


def test_every_deny_flag_must_be_stated_by_the_caller() -> None:
    """Restoring a permissive default has to break something, and this is it.

    Every other call in this file supplies all three flags, so the whole suite
    would keep passing if the defaults came back - which makes those tests
    blind to the property this module exists to hold. This one observes the
    signature and the refusal directly.

    The flags are the exclusion decision: a private session, a helper session,
    or a secret payload must never reach the index. The module this one
    delegates to refuses a whole batch when a row omits any of them, so
    accepting an omission here would defeat that at the only place it is used.
    """

    signature = inspect.signature(build_search_document)
    for name in ("private_session", "helper_session", "secret_payload"):
        assert signature.parameters[name].default is inspect.Parameter.empty

    # Through an untyped callable, so omission is a runtime question rather
    # than one the type checker answers before the test can observe it.
    call: Any = build_search_document
    with pytest.raises(TypeError, match="missing 3 required keyword-only arguments"):
        call(
            message_id="m1",
            chat_id="c1",
            role="user",
            body="hello world",
            has_media=False,
            transcript_visible=True,
            content_removed=False,
        )


def test_a_document_cannot_be_minted_without_the_evaluator_witness() -> None:
    """The authority guard: only the evaluator may declare a document eligible.

    SearchDocumentV1 carries eligible=True plus two authorization flags, and it
    is built through object.__new__ precisely because __post_init__ always
    raises - the dataclass cannot be constructed normally at all. So the
    module-private sentinel is the only thing between "the evaluator decided
    this message is indexable" and "a caller asserted it was".

    It was unbound. Deleting the check left the COMPLETE API suite green,
    because every legitimate path goes through build_search_document and passes
    the right sentinel; only an attempted forgery separates the two states.
    """
    from local_lm import search_document_v1 as module

    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        module._document_from_evaluator(
            witness=object(),
            message_id="m" * 8,
            chat_id="c" * 8,
            role="user",
            body="hello",
            has_media=False,
            selected_response_revision_id=None,
            created_at_unix=None,
        )

    # The real evaluator still builds one, so the guard refuses forgery rather
    # than refusing everything.
    document = build_search_document(
        message_id="m" * 8,
        chat_id="c" * 8,
        role="user",
        body="hello",
        has_media=False,
        transcript_visible=True,
        content_removed=False,
        private_session=False,
        helper_session=False,
        secret_payload=False,
    )
    assert document is not None
    assert document.eligible is True
    assert document.fts_write_authorized is False


def test_refuses_a_non_string_body_and_a_malformed_revision_id() -> None:
    common = {
        "message_id": "m1",
        "chat_id": "c1",
        "role": "user",
        "has_media": False,
        "transcript_visible": True,
        "content_removed": False,
        "private_session": False,
        "helper_session": False,
        "secret_payload": False,
    }
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(body=1, **common)
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(body="x", selected_response_revision_id="", **common)
    with pytest.raises(SearchDocumentError, match=INVALID_DOCUMENT):
        build_search_document(body="x", selected_response_revision_id=1, **common)
