from __future__ import annotations

import pytest

from local_lm.search_cursor_v1 import (
    INVALID_CURSOR,
    MAX_DIGEST_CHARS,
    MAX_EXPIRES,
    MAX_GENERATION,
    MAX_INT_DIGITS,
    MAX_OFFSET,
    MAX_TOKEN_CHARS,
    SearchCursorError,
    SearchCursorV1,
    bind_search_cursor,
    decode_search_cursor,
    encode_search_cursor,
    require_int_text,
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


def test_require_int_text_refuses_empty_non_string_and_non_digits() -> None:
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        require_int_text("", maximum=MAX_OFFSET)
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        require_int_text(1, maximum=MAX_OFFSET)
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        require_int_text("1a", maximum=MAX_OFFSET)
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        require_int_text("1" * (MAX_INT_DIGITS + 1), maximum=MAX_OFFSET)


def test_require_int_text_refuses_a_value_above_its_ceiling() -> None:
    """The digit string can be legal and still name a place past the bound.

    Empty, mixed, and over-long inputs are already pinned. A six-digit offset
    of 100001 and a ten-digit generation of 2147483648 pass those checks and
    only the value ceiling refuses them. Without this, dropping that comparison
    leaves require_int_text returning the parsed integer, and decode still
    looks green because bind_search_cursor's integer bound catches the same
    numbers later.
    """
    assert require_int_text(str(MAX_OFFSET), maximum=MAX_OFFSET) == MAX_OFFSET
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        require_int_text(str(MAX_OFFSET + 1), maximum=MAX_OFFSET)
    assert require_int_text(str(MAX_GENERATION), maximum=MAX_GENERATION) == MAX_GENERATION
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        require_int_text(str(MAX_GENERATION + 1), maximum=MAX_GENERATION)


def test_bind_refuses_a_short_hex_digest() -> None:
    """A query digest is exactly 64 lowercase hex characters, not at most 64.

    Oversize and non-hex inputs are already pinned. A 63-character hex string
    is inside the length ceiling that require_bounded_exact_str enforces, and
    every character is in the digest alphabet, so only the exact-length check
    refuses it. Mutating that check left a short digest accepted.
    """
    short = "a" * (MAX_DIGEST_CHARS - 1)
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        bind_search_cursor(
            index_generation=1,
            query_digest=short,
            offset=0,
            expires_at_unix=10,
            now_unix=1,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        decode_search_cursor(
            f"v1.1.0.10.{short}",
            now_unix=1,
            index_generation=1,
            query_digest=DIGEST,
        )


def test_decode_refuses_wrong_prefix_and_wrong_part_count() -> None:
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        decode_search_cursor(
            f"v0.1.0.10.{DIGEST}",
            now_unix=1,
            index_generation=1,
            query_digest=DIGEST,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        decode_search_cursor(
            f"v1.1.0.{DIGEST}",
            now_unix=1,
            index_generation=1,
            query_digest=DIGEST,
        )
    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        encode_search_cursor(object())


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


def test_a_cursor_cannot_be_minted_without_the_binder_witness() -> None:
    """The authority guard: only bind_search_cursor may mint a cursor.

    SearchCursorV1 records index_generation, contains_query_text and
    query_execution_authorized, and it is built through object.__new__ because
    the dataclass refuses ordinary construction. So the module-private sentinel
    is the whole boundary between "the binder issued this cursor" and "a caller
    asserted it" - including its index generation, which is what makes a stale
    cursor detectable.

    It was unbound. Deleting the check left the COMPLETE API suite green,
    because every legitimate path goes through bind_search_cursor and passes the
    right sentinel; only an attempted forgery separates the two states.
    """
    from local_lm import search_cursor_v1 as module

    with pytest.raises(SearchCursorError, match=INVALID_CURSOR):
        module._cursor_from_evaluator(
            witness=object(),
            index_generation=1,
            query_digest=DIGEST,
            offset=0,
            expires_at_unix=2,
        )

    # The real binder still works, so the guard refuses forgery rather than
    # refusing everything.
    cursor = bind_search_cursor(
        index_generation=1,
        query_digest=DIGEST,
        offset=0,
        expires_at_unix=2,
        now_unix=1,
    )
    assert cursor.query_execution_authorized is False
    assert cursor.contains_query_text is False


def test_a_max_field_cursor_encodes_inside_the_token_ceiling() -> None:
    """Field ceilings keep the token spelling inside MAX_TOKEN_CHARS.

    encode_search_cursor used to refuse a token longer than 128 characters, but
    a legal cursor's longest spelling is v1 plus 10-digit generation, 6-digit
    offset, 10-digit expiry, 64-character digest and four dots: 96 characters.
    That length check could not fire. This pins the implication instead.
    """
    cursor = bind_search_cursor(
        index_generation=MAX_GENERATION,
        query_digest=DIGEST,
        offset=MAX_OFFSET,
        expires_at_unix=MAX_EXPIRES,
        now_unix=MAX_EXPIRES - 1,
    )
    token = encode_search_cursor(cursor)
    assert len(token) <= MAX_TOKEN_CHARS
    assert len(token) == len(f"v1.{MAX_GENERATION}.{MAX_OFFSET}.{MAX_EXPIRES}.{DIGEST}")
