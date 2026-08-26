from __future__ import annotations

import pytest

from local_lm.search_snippet_v1 import (
    INVALID_SNIPPET,
    MAX_BODY_CHARS,
    MAX_QUERY_CHARS,
    MAX_TERMS,
    SearchSnippetError,
    SearchSnippetV1,
    build_search_snippet,
)


def test_segments_mark_first_match() -> None:
    snippet = build_search_snippet("prefix hello world suffix", "hello")
    assert snippet.html_authorized is False
    assert snippet.loads_entire_chat is False
    matched = [part for part in snippet.segments if part.matched]
    assert len(matched) == 1
    assert matched[0].text.casefold() == "hello"
    assert "<" not in "".join(part.text for part in snippet.segments)


def test_no_match_is_single_unmatched_segment() -> None:
    snippet = build_search_snippet("nothing relevant here", "hello")
    assert snippet.segments == snippet.segments
    assert all(part.matched is False for part in snippet.segments)
    assert "nothing" in snippet.segments[0].text


def test_refuses_hostile_and_unbounded_inputs() -> None:
    with pytest.raises(SearchSnippetError, match=INVALID_SNIPPET):
        build_search_snippet("", "hello")
    with pytest.raises(SearchSnippetError, match=INVALID_SNIPPET):
        build_search_snippet("hello", "")
    with pytest.raises(SearchSnippetError, match=INVALID_SNIPPET):
        build_search_snippet("x" * (MAX_BODY_CHARS + 1), "x")
    with pytest.raises(SearchSnippetError, match=INVALID_SNIPPET):
        build_search_snippet("hello", "hello", radius=0)
    with pytest.raises(SearchSnippetError, match=INVALID_SNIPPET):
        build_search_snippet("hello", "h" * (MAX_QUERY_CHARS + 1))
    with pytest.raises(SearchSnippetError, match=INVALID_SNIPPET):
        build_search_snippet("hello", " ".join(["term"] * (MAX_TERMS + 1)))

    class HostileStr(str):
        def casefold(self):
            raise RuntimeError("private attacker detail")

    with pytest.raises(SearchSnippetError, match=INVALID_SNIPPET) as caught:
        build_search_snippet(HostileStr("hello world"), "hello")
    assert "private attacker detail" not in str(caught.value)
    with pytest.raises(TypeError):
        SearchSnippetV1(
            schema="lm-atelier-search-snippet-v1",
            schema_version=1,
            segments=(),
            html_authorized=True,
        )


def _reassembled(snippet: SearchSnippetV1) -> str:
    return "".join(part.text for part in snippet.segments)


def test_expansions_before_the_match_do_not_shift_the_highlight() -> None:
    """The casefold of U+00DF (sharp s) is "ss": every folded index left of
    the match used to shift the slice into unrelated text. An ordinary
    German sentence reproduces it."""
    body = "gro\u00dfe Stra\u00dfe hello world"
    snippet = build_search_snippet(body, "hello")
    matched = [part for part in snippet.segments if part.matched]
    assert len(matched) == 1
    assert matched[0].text == "hello"
    assert _reassembled(snippet) == body


def test_a_ligature_before_the_match_keeps_the_boundary() -> None:
    body = "\ufb01le hello"
    snippet = build_search_snippet(body, "hello")
    matched = [part for part in snippet.segments if part.matched]
    assert len(matched) == 1
    assert matched[0].text == "hello"
    assert _reassembled(snippet) == body


def test_many_expansions_cannot_silently_lose_the_highlight() -> None:
    """Enough expansions used to push the folded index past the original
    body, where clamping slicing erased the highlight without an error."""
    body = "\u00df" * 10 + " hello tail"
    snippet = build_search_snippet(body, "hello")
    matched = [part for part in snippet.segments if part.matched]
    assert len(matched) == 1
    assert matched[0].text == "hello"
    assert _reassembled(snippet) == body


def test_an_expansion_inside_the_match_stays_on_codepoint_boundaries() -> None:
    """Full casefolding is deliberate recall: a query spelled "strasse"
    finds the U+00DF spelling, and the matched slice covers whole original
    code points rather than splitting the expansion."""
    body = "Stra\u00dfe hello"
    snippet = build_search_snippet(body, "strasse")
    matched = [part for part in snippet.segments if part.matched]
    assert len(matched) == 1
    assert matched[0].text == "Stra\u00dfe"
    assert matched[0].text.casefold() == "strasse"
    assert _reassembled(snippet) == body


def test_a_match_starting_inside_an_expansion_loses_no_adjacent_text() -> None:
    """A term that begins mid-expansion snaps to the containing original
    code point: the matched casefold contains the term and reassembly is
    lossless, instead of dropping the character the fold split."""
    body = "\u00dfhello rest"
    snippet = build_search_snippet(body, "shello")
    matched = [part for part in snippet.segments if part.matched]
    assert len(matched) == 1
    assert matched[0].text == "\u00dfhello"
    assert "shello" in matched[0].text.casefold()
    assert _reassembled(snippet) == body


def test_the_first_listed_term_with_a_hit_wins() -> None:
    snippet = build_search_snippet("alpha beta gamma", "missing beta")
    matched = [part for part in snippet.segments if part.matched]
    assert len(matched) == 1
    assert matched[0].text == "beta"


def test_a_maximum_length_query_match_is_never_truncated() -> None:
    """The ceiling was spent prefix-first, so a maximum-legal 200-character
    term came back as a 189-character segment still marked matched - the
    same invariant reopened at the legal boundary. The match is reserved
    in full before any context is allocated."""
    term = "x" * MAX_QUERY_CHARS
    body = "p" * 60 + term + "tail"
    snippet = build_search_snippet(body, term)
    matched = [part for part in snippet.segments if part.matched]
    assert len(matched) == 1
    assert term in matched[0].text.casefold()
    assert sum(len(part.text) for part in snippet.segments) <= 240


def test_a_folded_term_larger_than_the_ceiling_still_matches_whole() -> None:
    """A legal term can FOLD past the snippet ceiling (100 sharp-s fold to
    200 characters); the matched original slice is provably shorter than
    the folded term and must arrive complete."""
    term = "s" * 200
    body = "\u00df" * 100 + " tail"
    snippet = build_search_snippet(body, term)
    matched = [part for part in snippet.segments if part.matched]
    assert len(matched) == 1
    assert matched[0].text == "\u00df" * 100
    assert term in matched[0].text.casefold()
    assert sum(len(part.text) for part in snippet.segments) <= 240


def test_a_term_whose_fold_exceeds_the_query_cap_refuses() -> None:
    """Casefold expansion can TRIPLE a term: 100 raw U+FB03 are legal
    pre-fold but fold to 300 characters, and the whole-code-point match
    then overran the snippet ceiling with a 300-character matched
    segment. The cap binds the normalized term, so this query refuses
    instead of producing an over-ceiling snippet."""
    with pytest.raises(SearchSnippetError, match=INVALID_SNIPPET):
        build_search_snippet("\ufb03" * 100 + " body", "\ufb03" * 100)


def test_a_normalized_term_at_the_cap_still_matches_within_the_ceiling() -> None:
    """The boundary control: a term whose FOLD lands exactly on the cap is
    accepted, matches whole original code points, and stays inside the
    snippet ceiling."""
    query = "\ufb03" * 66 + "xx"  # folds to 66*3 + 2 = 200 characters
    body = "ffi" * 66 + "xx tail"
    snippet = build_search_snippet(body, query)
    matched = [part for part in snippet.segments if part.matched]
    assert len(matched) == 1
    assert matched[0].text == "ffi" * 66 + "xx"
    assert sum(len(part.text) for part in snippet.segments) <= 240
