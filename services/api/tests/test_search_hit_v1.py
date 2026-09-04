"""Hostile tests for pure search hit assembly.

Written against the ways the previous shape could be wrong rather than against
the happy path: a hit that renders its own text, a record that can be told it
loaded a chat, work done for a row that is then discarded, and an identity that
does not survive assembly.
"""

from __future__ import annotations

import dataclasses

import pytest

from local_lm.conversation_search_query_v1 import ConversationSearchError
from local_lm.search_hit_v1 import SearchHitV1, build_search_hit
from local_lm.search_snippet_v1 import SearchSnippetError, SearchSnippetV1

BODY = "the alpha and the beta appear here"


def _hit(**overrides: object) -> SearchHitV1 | None:
    kwargs: dict[str, object] = {
        "message_id": "m1",
        "chat_id": "c1",
        "body": BODY,
        "query": "alpha",
    }
    kwargs.update(overrides)
    return build_search_hit(**kwargs)


def test_a_hit_carries_structured_segments_not_rendered_text() -> None:
    """The snippet is a record, not a string the caller has to render.

    The earlier shape carried `snippet: str` built by a helper that has since
    been deleted. A string cannot say it was never HTML; the record can, and
    the guarantee stays attached to the thing it describes.
    """
    hit = _hit()
    assert hit is not None
    assert isinstance(hit.snippet, SearchSnippetV1)
    assert hit.snippet.html_authorized is False
    assert hit.snippet.segments
    assert any(segment.matched for segment in hit.snippet.segments)


def test_a_body_covering_no_term_is_not_a_hit() -> None:
    assert _hit(query="unrelated") is None


def test_nothing_is_built_for_a_row_that_is_discarded() -> None:
    """A miss costs no location and no snippet.

    Measured rather than assumed: an unusable message_id would refuse if a
    location were built, so a non-matching query returning None proves the
    score is checked before anything is constructed.
    """
    assert _hit(query="unrelated", message_id="has space") is None


def test_an_oversized_body_refuses_at_scoring_not_at_the_snippet() -> None:
    """Which module refuses is a fact about order, and it is worth pinning.

    Both the query module and the snippet module cap a body at 8192, and
    scoring runs first, so an oversized body never reaches the snippet module.
    A caller handling only SearchSnippetError would miss this refusal - which
    is exactly why these are not wrapped in a single error type.
    """
    with pytest.raises(ConversationSearchError):
        _hit(body="x" * 9000, query="x")


def test_the_fixed_facts_cannot_be_set_or_mutated() -> None:
    hit = _hit()
    assert hit is not None
    assert hit.loads_entire_chat is False
    assert hit.activates_branch is False
    with pytest.raises(TypeError):
        SearchHitV1(  # type: ignore[call-arg]
            schema="lm-atelier-search-hit-v1",
            schema_version=1,
            location=hit.location,
            snippet=hit.snippet,
            score=1,
            loads_entire_chat=True,
        )
    with pytest.raises(dataclasses.FrozenInstanceError):
        hit.__setattr__("activates_branch", True)


def test_identity_survives_assembly() -> None:
    hit = _hit(message_id="shared", chat_id="chat-two", project_id="p1")
    assert hit is not None
    assert hit.location.message_id == "shared"
    assert hit.location.chat_id == "chat-two"
    assert hit.location.project_id == "p1"


@pytest.mark.parametrize("bad", [None, 1, "", "has space", b"m1"])
def test_an_unusable_identifier_refuses_through_its_own_module(bad: object) -> None:
    """Sibling refusals propagate rather than being wrapped.

    Which module refused is the useful part of the answer: a caller that cannot
    build a location has a different problem from one whose body is too large.
    """
    with pytest.raises(ConversationSearchError):
        _hit(message_id=bad)


def test_a_snippet_refusal_reaches_the_caller_as_a_snippet_error() -> None:
    """A refusal raised inside assembly escapes assembly unchanged.

    The point is the assembler boundary, so this goes through
    `build_search_hit` rather than calling the snippet module directly. A test
    that called the sibling itself would prove only the sibling's own
    validation, and would stay green if a catch or a wrapper appeared inside
    the assembler - which is the regression worth binding.

    The path is naturally reachable and needs no patching. A hundred U+FB03
    ligatures are a hundred characters to the query module, which is inside its
    raw ceiling, but each one casefolds to three, so the normalized term is
    three hundred characters and the snippet module refuses it. Scoring runs
    first and has to pass, so the body contains the same run.
    """
    term = "ﬃ" * 100
    with pytest.raises(SearchSnippetError):
        build_search_hit(
            message_id="m1",
            chat_id="c1",
            body=f"the {term} appears here",
            query=term,
        )


def test_the_score_is_the_distinct_term_coverage() -> None:
    assert _hit(query="alpha beta").score == 2  # type: ignore[union-attr]
    assert _hit(query="alpha missing").score == 1  # type: ignore[union-attr]
