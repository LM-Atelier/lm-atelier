"""Hostile tests for pure search result pages.

Written against the ways a cursor guard can be wrong rather than against
paging. The interesting failures all have the same shape: a character class
that says "this is a digit" and an `int()` that disagrees, so the module raises
something other than its own fixed refusal. Because `SearchPageError`
subclasses `ValueError`, a caller catching broadly cannot tell the difference,
which is exactly why these are asserted on the specific type.
"""

from __future__ import annotations

import pytest

from local_lm import search_page_v1
from local_lm.search_cursor_v1 import MAX_INT_DIGITS, MAX_OFFSET, SearchCursorError
from local_lm.search_hit_v1 import SearchHitV1, build_search_hit
from local_lm.search_page_v1 import (
    DEFAULT_PAGE_SIZE,
    INVALID_PAGE,
    MAX_PAGE_SIZE,
    SearchPageError,
    SearchPageV1,
    _offset,
    build_search_page,
)

BODY = "the alpha and the beta appear here"


def _hit(message_id: str) -> SearchHitV1:
    hit = build_search_hit(message_id=message_id, chat_id="c1", body=BODY, query="alpha")
    assert hit is not None
    return hit


def _hits(count: int) -> list[SearchHitV1]:
    return [_hit(f"m{index:02d}") for index in range(count)]


def test_a_page_slices_in_order_and_says_where_the_next_one_starts() -> None:
    page = build_search_page(_hits(7), limit=3)
    assert [hit.location.message_id for hit in page.hits] == ["m00", "m01", "m02"]
    assert page.next_cursor == "3"
    assert page.total_reported == 7

    later = build_search_page(_hits(7), limit=3, cursor=page.next_cursor)
    assert [hit.location.message_id for hit in later.hits] == ["m03", "m04", "m05"]
    assert later.next_cursor == "6"

    last = build_search_page(_hits(7), limit=3, cursor="6")
    assert [hit.location.message_id for hit in last.hits] == ["m06"]
    assert last.next_cursor is None


def test_a_cursor_past_the_end_is_an_empty_page_rather_than_a_refusal() -> None:
    """A stale cursor is not a hostile one.

    Rows can be removed between two requests, so an offset that was valid when
    it was issued can address nothing by the time it comes back. Returning an
    empty final page keeps that idempotent; refusing would turn ordinary
    staleness into an error the caller has to special-case.
    """
    page = build_search_page(_hits(3), limit=5, cursor="99")
    assert page.hits == ()
    assert page.next_cursor is None
    assert page.total_reported == 3


def test_a_superscript_digit_is_refused_as_a_page_error_not_an_int_crash() -> None:
    """U+00B2 is a digit to `str` and a ValueError to `int`.

    `"\u00b2".isdigit()` is True, and `int("\u00b2")` raises. A guard written
    with `isdigit` therefore passes this cursor and the conversion on the next
    line raises `ValueError("invalid literal for int()")` - not the fixed
    refusal this module promises. The assertion is on `SearchPageError`
    specifically: a bare `ValueError` does not satisfy it, while an
    `except ValueError` in a caller would have hidden the whole problem.
    """
    for cursor in ("\u00b2", "\u2075", "1\u00b2"):
        with pytest.raises(SearchPageError, match=INVALID_PAGE):
            build_search_page(_hits(3), limit=2, cursor=cursor)


def test_a_non_ascii_decimal_digit_is_refused_even_though_int_accepts_it() -> None:
    """U+0663 parses to 3, and still is not a cursor this module ever wrote.

    `isdecimal` is true and `int()` succeeds, so nothing crashes - which is why
    the ASCII check is separate rather than folded in. The only cursors that
    exist are the ones `build_search_page` emits, and those are ASCII.
    """
    for cursor in ("\u0663", "\U0001d7f6"):
        with pytest.raises(SearchPageError, match=INVALID_PAGE):
            build_search_page(_hits(3), limit=2, cursor=cursor)


def test_an_over_long_all_digit_cursor_is_refused_before_int_sees_it() -> None:
    """`int()` refuses a decimal string longer than 4300 digits.

    Every character is an ASCII digit, so a character check alone passes and the
    conversion raises `ValueError("Exceeds the limit (4300 digits)")` instead of
    the fixed refusal. The length bound is what stops it, well before that.
    """
    with pytest.raises(SearchPageError, match=INVALID_PAGE):
        build_search_page(_hits(3), limit=2, cursor="1" * 4301)
    with pytest.raises(SearchPageError, match=INVALID_PAGE):
        build_search_page(_hits(3), limit=2, cursor="1" * (MAX_INT_DIGITS + 1))


def test_a_padded_cursor_is_refused_so_one_page_has_one_spelling() -> None:
    """ "2" and "002" must not both address page two.

    A leading zero makes an offset ambiguous: two callers holding different
    strings believe they hold different positions when they hold one. The
    sibling cursor module refuses it and this must agree, or a cursor minted by
    one and read by the other means different things.
    """
    for padded in ("002", "0000000000", "01"):
        with pytest.raises(SearchPageError, match=INVALID_PAGE):
            build_search_page(_hits(5), limit=2, cursor=padded)
    assert _offset("0") == 0
    assert _offset("2") == 2


def test_an_offset_beyond_the_ceiling_is_refused() -> None:
    """An offset is a position in a result set, not an arbitrary integer."""
    with pytest.raises(SearchPageError, match=INVALID_PAGE):
        build_search_page(_hits(3), limit=2, cursor=str(MAX_OFFSET + 1))
    assert _offset(str(MAX_OFFSET)) == MAX_OFFSET


def test_more_rows_than_a_cursor_can_address_are_refused() -> None:
    """Whatever it hands out, it must accept back - made structural.

    `_offset` refuses an offset past MAX_OFFSET, so with more rows than that the
    next offset could be one this module would reject, stranding a caller on a
    page they cannot leave. Bounding the input makes that impossible rather than
    conditional: if the row count cannot exceed the ceiling, neither can an
    emitted cursor.

    The bound is checked before the rows are validated, so this costs a list and
    no hit construction.
    """
    # Every element is a VALID hit, one object referenced many times, so the
    # refusal can only come from the row bound. Filling the list with invalid
    # items would be refused by item validation instead and would pass whether
    # or not the bound exists - which is exactly how the first version of this
    # test proved nothing.
    one = _hit("m00")
    with pytest.raises(SearchPageError, match=INVALID_PAGE):
        build_search_page([one] * (MAX_OFFSET + 1), limit=2)
    assert build_search_page([one] * MAX_OFFSET, limit=2).total_reported == MAX_OFFSET
    every = [
        build_search_page(_hits(9), limit=2, cursor=c).next_cursor for c in ("0", "2", "4", "6")
    ]
    for emitted in every:
        if emitted is not None:
            assert _offset(emitted) == int(emitted)


def test_the_page_delegates_its_cursor_to_the_one_canonical_parser() -> None:
    """Production must CALL the sibling parser, not agree with it.

    This replaces an equivalence table over eighteen inputs. That table proved
    the two implementations agreed on the inputs it happened to contain, which
    is a sample and not an invariant: a fifth rule added to the canonical parser
    drifts silently past any finite table that does not contain an input it
    changes. With one implementation there is nothing to drift, and the only
    thing left worth asserting is that the page really does delegate.

    The parser is replaced with a spy that records its arguments and refuses.
    If anyone reimplements the rules locally the spy is never called and this
    fails, which is exactly the regression the single-source rule exists to
    prevent.
    """
    calls: list[tuple[object, int]] = []

    def spy(value: object, *, maximum: int) -> int:
        calls.append((value, maximum))
        raise SearchCursorError("refused by the canonical parser")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(search_page_v1, "require_int_text", spy)
    try:
        with pytest.raises(SearchPageError, match=INVALID_PAGE):
            build_search_page(_hits(3), limit=2, cursor="5")
    finally:
        monkeypatch.undo()

    assert calls == [("5", MAX_OFFSET)], calls


def test_the_parsers_answer_becomes_the_offset() -> None:
    """Delegation is not enough on its own; the answer has to be used.

    A page that called the parser and then ignored what it returned would pass
    the spy test above while paging from the wrong place, so the returned value
    is followed through to the window it selects.
    """
    used: list[int] = []

    def spy(value: object, *, maximum: int) -> int:
        used.append(4)
        return 4

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(search_page_v1, "require_int_text", spy)
    try:
        page = build_search_page(_hits(9), limit=2, cursor="anything-the-spy-accepts")
    finally:
        monkeypatch.undo()

    assert used == [4]
    assert [hit.location.message_id for hit in page.hits] == ["m04", "m05"]


def test_a_cursor_the_canonical_parser_refuses_is_a_page_error() -> None:
    """The rules still hold end to end, now through the one parser.

    These are the same hostile cursors as before - a superscript digit that
    str.isdigit accepts and int() rejects, a non-ASCII decimal, a padded offset
    that would alias one page to two spellings, an over-long all-digit string,
    and a value past the ceiling. They are asserted here to prove the
    translation is wired up for every rule, not to restate the parser's own
    tests.
    """
    for cursor in (
        "\u00b2",
        "\u2075",
        "\u0663",
        "002",
        "01",
        "1" * (MAX_INT_DIGITS + 1),
        "1" * 4301,
        str(MAX_OFFSET + 1),
        "-1",
        "1.5",
        "abc",
        "",
        " 1",
    ):
        with pytest.raises(SearchPageError, match=INVALID_PAGE):
            build_search_page(_hits(5), limit=2, cursor=cursor)
    assert build_search_page(_hits(9), limit=2, cursor="4").hits[0].location.message_id == "m04"


def test_a_page_refuses_hostile_shapes() -> None:
    hits = _hits(1)
    for bad_hits in ("nope", 7, None, {"a": 1}, iter(hits)):
        with pytest.raises(SearchPageError, match=INVALID_PAGE):
            build_search_page(bad_hits, limit=1)
    for bad_limit in (0, -1, MAX_PAGE_SIZE + 1, True, 1.0, "3", None):
        with pytest.raises(SearchPageError, match=INVALID_PAGE):
            build_search_page(hits, limit=bad_limit)
    for bad_cursor in ("-1", "1.5", "abc", "", " 1", "1 ", 3, 3.0, b"3"):
        with pytest.raises(SearchPageError, match=INVALID_PAGE):
            build_search_page(hits, limit=1, cursor=bad_cursor)
    with pytest.raises(SearchPageError, match=INVALID_PAGE):
        build_search_page(["not-a-hit"], limit=1)
    with pytest.raises(SearchPageError, match=INVALID_PAGE):
        build_search_page([*hits, "not-a-hit"], limit=1)


def test_a_page_reports_a_count_on_every_path_and_claims_no_chat_load() -> None:
    """`total_reported` is an int always, and the record cannot claim a load."""
    empty = build_search_page([], limit=DEFAULT_PAGE_SIZE)
    assert empty.total_reported == 0
    assert empty.hits == ()
    assert empty.next_cursor is None
    assert isinstance(empty, SearchPageV1)
    assert empty.loads_entire_chat is False
    assert empty.schema == "lm-atelier-search-page-v1"
    assert empty.schema_version == 1
    with pytest.raises(TypeError):
        SearchPageV1(  # type: ignore[call-arg]
            schema="lm-atelier-search-page-v1",
            schema_version=1,
            hits=(),
            next_cursor=None,
            total_reported=0,
            loads_entire_chat=True,
        )
