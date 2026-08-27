from __future__ import annotations

import pytest

from local_lm.conversation_search_mutation_v1 import (
    INVALID_MUTATION,
    MAX_ID,
    MAX_OPERATION_CHARS,
    MAX_SEQUENCE,
    ConversationSearchMutationV1,
    SearchMutationError,
    declare_search_mutation,
)


def test_declares_identity_without_raw_text() -> None:
    mutation = declare_search_mutation(
        sequence=3,
        message_id="m1",
        chat_id="c1",
        operation="upsert",
        content_revision="rev-9",
    )
    assert mutation.sequence == 3
    assert mutation.operation == "upsert"
    assert mutation.fts_write_authorized is False
    assert mutation.index_rebuild_authorized is False
    assert mutation.raw_text_authorized is False
    assert "hello" not in repr(mutation)


def test_refuses_invalid_and_unbounded_facts() -> None:
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        declare_search_mutation(
            sequence=0,
            message_id="m1",
            chat_id="c1",
            operation="upsert",
            content_revision="rev-9",
        )
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        declare_search_mutation(
            sequence=MAX_SEQUENCE + 1,
            message_id="m1",
            chat_id="c1",
            operation="upsert",
            content_revision="rev-9",
        )
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        declare_search_mutation(
            sequence=1,
            message_id="bad id",
            chat_id="c1",
            operation="remove",
            content_revision="rev-9",
        )
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        declare_search_mutation(
            sequence=True,
            message_id="m1",
            chat_id="c1",
            operation="upsert",
            content_revision="rev-9",
        )
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        declare_search_mutation(
            sequence=1,
            message_id="m1",
            chat_id="c1",
            operation="rewrite",
            content_revision="rev-9",
        )
    accepted = declare_search_mutation(
        sequence=1,
        message_id="m" * MAX_ID,
        chat_id="c" * MAX_ID,
        operation="upsert",
        content_revision="r" * MAX_ID,
    )
    assert accepted.message_id == "m" * MAX_ID
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        declare_search_mutation(
            sequence=1,
            message_id="m" * (MAX_ID + 1),
            chat_id="c1",
            operation="upsert",
            content_revision="rev-9",
        )
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        declare_search_mutation(
            sequence=1,
            message_id="m" * 10_000,
            chat_id="c1",
            operation="upsert",
            content_revision="rev-9",
        )
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        declare_search_mutation(
            sequence=1,
            message_id=" " * 10_000,
            chat_id="c1",
            operation="upsert",
            content_revision="rev-9",
        )
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        declare_search_mutation(
            sequence=1,
            message_id="m1",
            chat_id="c1",
            operation="x" * MAX_OPERATION_CHARS,
            content_revision="rev-9",
        )
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        declare_search_mutation(
            sequence=1,
            message_id="m1",
            chat_id="c1",
            operation="x" * (MAX_OPERATION_CHARS + 1),
            content_revision="rev-9",
        )
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        declare_search_mutation(
            sequence=1,
            message_id="m1",
            chat_id="c1",
            operation="x" * 10_000,
            content_revision="rev-9",
        )


def test_public_constructor_cannot_mint_write_authority() -> None:
    with pytest.raises(SearchMutationError, match=INVALID_MUTATION):
        ConversationSearchMutationV1()
    with pytest.raises(TypeError):
        ConversationSearchMutationV1(
            sequence=1,
            message_id="m1",
            chat_id="c1",
            operation="upsert",
            content_revision="rev-9",
            fts_write_authorized=True,
        )
