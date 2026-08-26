from __future__ import annotations

import pytest

from local_lm.search_cursor_v1 import (
    INVALID_CURSOR,
    MAX_DIGEST_CHARS,
    MAX_GENERATION,
    MAX_OFFSET,
    MAX_TOKEN_CHARS,
    SearchCursorError,
    SearchCursorV1,
    bind_search_cursor,
    decode_search_cursor,
    encode_search_cursor,
)

DIGEST = "a" * MAX_DIGEST_CHARS


def test_bind_encode_decode_round_trip() -> None:
    cursor = bind_search_cursor(
        index_generation=3,
        query_digest=DIGEST,
        offset=20,
        expires_at_unix=2_000,
        now_unix=1_000,
    )
    assert cursor.contains_query_text is False
    assert cursor.query_execution_authorized is False
    token = encode_search_cursor(cursor)
    assert "hello" not in token
    assert len(token) <= MAX_TOKEN_CHARS
    restored = decode_search_cursor(
        token,
        now_unix=1_500,
        index_generation=3,
        query_digest=DIGEST,
    )
    assert restored.offset == 20
    assert restored.index_generation == 3
    assert restored.query_digest == DIGEST


def test_refuses_unbounded_digest_token_and_stale_facts() -> None:
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        bind_search_cursor(
            index_generation=1,
            query_digest="b" * MAX_DIGEST_CHARS,
            offset=0,
            expires_at_unix=10,
            now_unix=11,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        bind_search_cursor(
            index_generation=1,
            query_digest="g" * MAX_DIGEST_CHARS,
            offset=0,
            expires_at_unix=10,
            now_unix=1,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        bind_search_cursor(
            index_generation=1,
            query_digest="a" * (MAX_DIGEST_CHARS + 1),
            offset=0,
            expires_at_unix=10,
            now_unix=1,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        bind_search_cursor(
            index_generation=1,
            query_digest="a" * 10_000,
            offset=0,
            expires_at_unix=10,
            now_unix=1,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        bind_search_cursor(
            index_generation=MAX_GENERATION + 1,
            query_digest=DIGEST,
            offset=0,
            expires_at_unix=10,
            now_unix=1,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        bind_search_cursor(
            index_generation=1,
            query_digest=DIGEST,
            offset=MAX_OFFSET + 1,
            expires_at_unix=10,
            now_unix=1,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        bind_search_cursor(
            index_generation=True,
            query_digest=DIGEST,
            offset=0,
            expires_at_unix=10,
            now_unix=1,
        )
    cursor = bind_search_cursor(
        index_generation=1,
        query_digest=DIGEST,
        offset=0,
        expires_at_unix=10,
        now_unix=1,
    )
    token = encode_search_cursor(cursor)
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        decode_search_cursor(
            token,
            now_unix=1,
            index_generation=2,
            query_digest=DIGEST,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        decode_search_cursor(
            token,
            now_unix=1,
            index_generation=1,
            query_digest="b" * MAX_DIGEST_CHARS,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        decode_search_cursor(
            "x" * (MAX_TOKEN_CHARS + 1), now_unix=1, index_generation=1, query_digest=DIGEST
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        decode_search_cursor("x" * 10_000, now_unix=1, index_generation=1, query_digest=DIGEST)
    zero = bind_search_cursor(
        index_generation=0,
        query_digest=DIGEST,
        offset=0,
        expires_at_unix=10,
        now_unix=1,
    )
    assert encode_search_cursor(zero).startswith("v1.0.0.")
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        decode_search_cursor(
            f"v1.007.0.10.{DIGEST}",
            now_unix=1,
            index_generation=7,
            query_digest=DIGEST,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        decode_search_cursor(
            f"v1.7.000.10.{DIGEST}",
            now_unix=1,
            index_generation=7,
            query_digest=DIGEST,
        )


def test_public_constructor_cannot_mint_query_text() -> None:
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        SearchCursorV1()
    with pytest.raises(TypeError):
        SearchCursorV1(
            schema="lm-atelier-search-cursor-v1",
            schema_version=1,
            index_generation=1,
            query_digest=DIGEST,
            offset=0,
            expires_at_unix=10,
            contains_query_text=True,
        )


def test_a_cursor_is_expired_at_the_instant_it_names() -> None:
    """The expiry second is already too late, through binding and decoding.

    `expires_at_unix` names when the cursor stops being valid, so accepting it
    at exactly that value leaves a one-second window where a stale cursor still
    pages. Editor sessions in this repository treat `expires_at <= now` as
    expired; a pagination cursor is the same kind of fence.
    """
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        bind_search_cursor(
            index_generation=1,
            query_digest=DIGEST,
            offset=0,
            expires_at_unix=1_000,
            now_unix=1_000,
        )

    live = bind_search_cursor(
        index_generation=1,
        query_digest=DIGEST,
        offset=0,
        expires_at_unix=1_000,
        now_unix=999,
    )
    token = encode_search_cursor(live)
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        decode_search_cursor(
            token,
            now_unix=1_000,
            index_generation=1,
            query_digest=DIGEST,
        )
