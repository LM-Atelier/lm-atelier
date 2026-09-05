from __future__ import annotations

import pytest

from local_lm.conversation_search_compose_v1 import (
    COMPOSE_ROW_KEYS,
    INVALID_COMPOSE,
    MAX_COMPOSE_BODY_CHARS,
    MAX_COMPOSE_KEYS,
    MAX_COMPOSE_ROWS,
    MAX_COMPOSE_TEXT_CHARS,
    MAX_COMPOSE_WORK,
    SearchComposeError,
    SearchComposeResultV1,
    _bound_id,
    _owned_compose_row,
    _RowTooWide,
    compose_conversation_search,
)
from local_lm.conversation_search_query_v1 import MAX_ID_CHARS
from local_lm.search_privacy_v1 import MAX_PRIVACY_QUERY_CHARS


def _row(mid: str, body: str, chat_id: str = "chat-1", **over):
    base = {
        "message_id": mid,
        "chat_id": chat_id,
        "body": body,
        "transcript_visible": True,
        "content_removed": False,
        "private_session": False,
        "helper_session": False,
        "secret_payload": False,
    }
    base.update(over)
    return base


def _refuse(rows, query="hello", **kwargs) -> None:
    with pytest.raises(SearchComposeError, match=INVALID_COMPOSE) as caught:
        compose_conversation_search(rows, query, **kwargs)
    assert str(caught.value) == INVALID_COMPOSE
    assert (
        not isinstance(caught.value.__cause__, SearchComposeError) or caught.value.__cause__ is None
    )


def test_compose_binds_row_chat_id_and_drops_tombstone() -> None:
    rows = [
        _row("keep", "hello world", chat_id="chat-a"),
        _row("gone", "hello world removed", chat_id="chat-a", content_removed=True),
        _row("other", "hello there", chat_id="chat-b"),
    ]
    result = compose_conversation_search(rows, "hello world", limit=10)
    assert result.eligible == 2
    ids = [h.location.message_id for h in result.page.hits]
    assert "keep" in ids
    assert "gone" not in ids
    keep = next(h for h in result.page.hits if h.location.message_id == "keep")
    assert keep.location.chat_id == "chat-a"
    other = next(h for h in result.page.hits if h.location.message_id == "other")
    assert other.location.chat_id == "chat-b"
    assert result.query_execution_authorized is False
    assert result.fts_write_authorized is False
    assert result.index_rebuild_authorized is False
    assert result.privacy.may_log_raw_query is False
    assert "hello world" not in str(result.privacy)
    assert result.considered == 3


def test_compose_requires_chat_id() -> None:
    row = _row("m1", "hello")
    del row["chat_id"]
    _refuse([row])
    _refuse([_row("m1", "hello", chat_id="bad id")])


def test_bound_id_refuses_empty_and_non_string_values() -> None:
    with pytest.raises(SearchComposeError, match=INVALID_COMPOSE) as empty:
        _bound_id("")
    assert str(empty.value) == INVALID_COMPOSE
    with pytest.raises(SearchComposeError, match=INVALID_COMPOSE) as numbered:
        _bound_id(1)
    assert str(numbered.value) == INVALID_COMPOSE


def test_compose_refuses_empty_ids() -> None:
    _refuse([_row("m1", "hello", chat_id="")])
    _refuse([_row("", "hello")])


def test_compose_refuses_non_string_ids() -> None:
    numbered_chat = _row("m1", "hello")
    numbered_chat["chat_id"] = 1
    _refuse([numbered_chat])
    numbered_message = _row("m1", "hello")
    numbered_message["message_id"] = 1
    _refuse([numbered_message])


def test_compose_refuses_duplicate_message_ids_across_chats() -> None:
    rows = [
        _row("shared", "hello from first", chat_id="chat-a"),
        _row("shared", "hello from second", chat_id="chat-b"),
    ]
    _refuse(rows)


def test_compose_refuses_duplicate_message_ids_in_same_chat() -> None:
    rows = [
        _row("shared", "hello first copy", chat_id="chat-a"),
        _row("shared", "hello second copy", chat_id="chat-a"),
    ]
    _refuse(rows)


def test_compose_refuses_over_row_cap() -> None:
    rows = [_row(f"m{i:02d}", "hello world") for i in range(MAX_COMPOSE_ROWS + 1)]
    _refuse(rows)


def test_compose_refuses_over_body_cap() -> None:
    _refuse([_row("m1", "x" * (MAX_COMPOSE_BODY_CHARS + 1))])


def test_compose_refuses_over_aggregate_text_cap() -> None:
    chunk = "h" * MAX_COMPOSE_BODY_CHARS
    needed = (MAX_COMPOSE_TEXT_CHARS // MAX_COMPOSE_BODY_CHARS) + 1
    rows = [_row(f"m{i:02d}", chunk) for i in range(needed)]
    _refuse(rows, query="h")


def test_compose_refuses_over_work_cap() -> None:
    term_count = 16
    query = " ".join(f"t{i:02d}" for i in range(term_count))
    rows_needed = (MAX_COMPOSE_WORK // term_count) + 1
    rows = [_row(f"m{i:02d}", query) for i in range(rows_needed)]
    assert rows_needed <= MAX_COMPOSE_ROWS
    _refuse(rows, query=query)


def test_compose_wraps_invalid_query_and_visibility() -> None:
    _refuse([_row("m1", "hello")], query=None)
    _refuse([_row("m1", "hello")], query=123)
    _refuse([_row("m1", "hello")], query="")
    _refuse([_row("m1", "hello")], query="   ")
    _refuse([_row("m1", "hello", transcript_visible="yes")])
    _refuse([_row("m1", "hello")], limit=0)
    _refuse([_row("m1", "hello")], limit=True)
    _refuse([_row("m1", "hello")], limit=99)
    _refuse("not-a-sequence")  # type: ignore[arg-type]
    _refuse([_row("m1", "hello", chat_id="x" * (MAX_ID_CHARS + 1))])
    _refuse([{"not": "a complete row"}])


def test_compose_pages_after_bounded_rank() -> None:
    rows = [_row(f"m{i:02d}", "hello world") for i in range(12)]
    result = compose_conversation_search(rows, "hello", limit=5)
    assert result.eligible == 12
    assert len(result.page.hits) == 5
    assert result.page.next_cursor == "5"
    assert result.page.total_reported == 12


def test_compose_refuses_hostile_str_keys_without_equality() -> None:
    class HostileKey(str):
        def __eq__(self, other):
            raise RuntimeError("private attacker detail")

        def __hash__(self):
            return str.__hash__(self)

    row = {HostileKey(key): value for key, value in _row("m1", "hello").items()}
    with pytest.raises(SearchComposeError, match=INVALID_COMPOSE) as caught:
        compose_conversation_search([row], "hello")
    assert str(caught.value) == INVALID_COMPOSE
    assert "private attacker detail" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_compose_refuses_hostile_mapping_subclass() -> None:
    class HostileDict(dict):
        def get(self, key, default=None):
            raise RuntimeError("private attacker detail")

    with pytest.raises(SearchComposeError, match=INVALID_COMPOSE) as caught:
        compose_conversation_search([HostileDict(_row("m1", "hello"))], "hello")
    assert str(caught.value) == INVALID_COMPOSE
    assert "private attacker detail" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_compose_refuses_hostile_sequence_and_string_subclasses() -> None:
    class HostileList(list):
        def __iter__(self):
            raise RuntimeError("private attacker detail")

    class HostileStr(str):
        def split(self, *args, **kwargs):
            raise RuntimeError("private attacker detail")

    _refuse(HostileList([_row("m1", "hello")]))
    _refuse([_row("m1", "hello")], query=HostileStr("hello"))


def test_compose_refuses_oversized_whitespace_query() -> None:
    _refuse([_row("m1", "hello")], query=" " * (MAX_PRIVACY_QUERY_CHARS + 1))


def test_compose_authority_flags_are_not_constructor_settable() -> None:
    with pytest.raises(TypeError):
        SearchComposeResultV1(
            schema="lm-atelier-search-compose-v1",
            schema_version=1,
            page=compose_conversation_search([_row("m1", "hello")], "hello").page,
            privacy=compose_conversation_search([_row("m1", "hello")], "hello").privacy,
            bounds=compose_conversation_search([_row("m1", "hello")], "hello").bounds,
            considered=1,
            eligible=1,
            query_execution_authorized=True,
        )


def test_a_row_wider_than_the_ceiling_is_refused_without_being_read() -> None:
    """The key ceiling refuses BEFORE the copy, not after it.

    Without this the bound is unobservable and therefore unproven: an oversized
    row is refused either way, and the only difference is how much work was
    done first. Asserting the named refusal is what distinguishes bounded
    acquisition from acquisition that happens to end in a refusal, and it does
    it without timing anything.
    """
    wide = {f"key-{index}": index for index in range(MAX_COMPOSE_KEYS + 1)}
    with pytest.raises(_RowTooWide):
        _owned_compose_row(wide)


def test_a_legal_sized_row_with_the_wrong_keys_is_refused_generically() -> None:
    """The other half of the distinction, without which the first test is weak.

    A test that only ever sees _RowTooWide cannot tell a real ceiling from a
    function that raises it for everything. This row is exactly the permitted
    width, so nothing but the key-set check can refuse it.
    """
    wrong = {f"key-{index}": index for index in range(len(COMPOSE_ROW_KEYS))}
    assert len(wrong) <= MAX_COMPOSE_KEYS
    with pytest.raises(SearchComposeError) as caught:
        _owned_compose_row(wrong)
    assert type(caught.value) is SearchComposeError


def test_the_ceiling_refusal_stays_invisible_to_a_caller() -> None:
    """The private distinction must not become a public one.

    Naming the refusal internally is only safe while the outside still sees the
    single fixed error with the same text; otherwise the refusal has started
    describing its input, which is what the normalization exists to prevent.
    """
    wide = dict.fromkeys((f"key-{index}" for index in range(MAX_COMPOSE_KEYS + 1)), 0)
    with pytest.raises(SearchComposeError) as caught:
        compose_conversation_search([wide], "hello")

    # EXACT type, not isinstance. The previous version of this test used the
    # shared _refuse helper, which checks isinstance, message and cause - all
    # of which a leaked subclass satisfies. It passed while the private type
    # was reaching callers, so it asserted less than it appeared to.
    assert type(caught.value) is SearchComposeError
    assert not isinstance(caught.value, _RowTooWide)
    assert str(caught.value) == INVALID_COMPOSE

    # BOTH links, not just the cause. `raise ... from None` clears __cause__
    # and suppresses the traceback display, but leaves __context__ pointing at
    # the private exception - and __context__ is a public attribute, so the
    # distinction was still reachable by any caller who looked. Asserting only
    # the cause is what let that through.
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_compose_refuses_a_non_string_body() -> None:
    numbered = _row("m1", "hello")
    numbered["body"] = 1
    _refuse([numbered])
