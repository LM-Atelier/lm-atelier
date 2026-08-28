"""Hostile tests for the two search declaration modules, plus a drift guard.

These modules tell a caller what the system will accept. The tests below are
written against the ways a declaration can be wrong without being internally
inconsistent: permitting more than the implementation accepts, reporting a count
that is true about the input and false about the system, or letting a caller
construct a record claiming a permission the module has no authority to give.
"""

from __future__ import annotations

import dataclasses

import pytest

from local_lm.conversation_search_query_v1 import (
    ConversationSearchError,
    normalize_search_query,
    query_terms,
)
from local_lm.search_privacy_v1 import (
    MAX_PRIVACY_QUERY_CHARS,
    MAX_PRIVACY_TERMS,
    SearchPrivacyError,
    SearchPrivacyV1,
    classify_search_privacy,
)
from local_lm.search_resource_bounds_v1 import (
    MAX_QUERY_CHARS,
    MAX_TERMS,
    SearchResourceBoundsError,
    SearchResourceBoundsV1,
    declare_search_resource_bounds,
)
from local_lm.search_snippet_v1 import SearchSnippetError, build_search_snippet

AT_CAP_QUERY = "q" * 200
OVER_CAP_QUERY = "q" * 201
AT_CAP_TERMS = " ".join(f"t{index}" for index in range(16))
OVER_CAP_TERMS = " ".join(f"t{index}" for index in range(17))


def _snippet_accepts(query: str) -> bool:
    try:
        build_search_snippet("a body containing q and t0 for matching", query)
    except SearchSnippetError:
        return False
    return True


def _query_accepts(query: str) -> bool:
    try:
        query_terms(normalize_search_query(query))
    except ConversationSearchError:
        return False
    return True


def _privacy_accepts(query: str) -> bool:
    try:
        classify_search_privacy(query)
    except SearchPrivacyError:
        return False
    return True


def _bounds_accepts(*, chars: int, terms: int) -> bool:
    try:
        declare_search_resource_bounds(query_chars=chars, term_count=terms)
    except SearchResourceBoundsError:
        return False
    return True


def test_every_module_agrees_on_the_same_boundary_by_behaviour() -> None:
    """One cross-module contract, asserted as behaviour rather than constants.

    A declaration that permits more than the implementation accepts turns a
    caller's correct behaviour into a refusal: a 256-character query reads as
    acceptable while the modules doing the work refuse anything over 200. Each
    module is internally consistent, which is why nothing catches it.

    Comparing constants would not be enough. A module can hold the right number
    and fail to enforce it, and a module can enforce the right boundary without
    exposing a constant at all. What the system owes a caller is that every
    participating module answers the same way at the same input, so that is what
    is asserted.

    conversation_search_query_v1 is included. It enforces the same boundary and
    is on trunk, so leaving it out let its cap drift without any test noticing:
    raising its limit to 201 left every assertion here passing, which is the
    exact defect this contract exists to catch.
    """
    assert _privacy_accepts(AT_CAP_QUERY) is True
    assert _privacy_accepts(OVER_CAP_QUERY) is False

    assert _snippet_accepts(AT_CAP_QUERY) is True
    assert _snippet_accepts(OVER_CAP_QUERY) is False

    assert _query_accepts(AT_CAP_QUERY) is True
    assert _query_accepts(OVER_CAP_QUERY) is False

    assert _bounds_accepts(chars=200, terms=1) is True
    assert _bounds_accepts(chars=201, terms=1) is False

    assert _privacy_accepts(AT_CAP_TERMS) is True
    assert _privacy_accepts(OVER_CAP_TERMS) is False

    assert _snippet_accepts(AT_CAP_TERMS) is True
    assert _snippet_accepts(OVER_CAP_TERMS) is False

    assert _query_accepts(AT_CAP_TERMS) is True
    assert _query_accepts(OVER_CAP_TERMS) is False

    assert _bounds_accepts(chars=10, terms=16) is True
    assert _bounds_accepts(chars=10, terms=17) is False


def test_a_query_the_search_would_refuse_is_not_declared_acceptable() -> None:
    assert classify_search_privacy(AT_CAP_QUERY).query_char_count == 200
    with pytest.raises(SearchPrivacyError):
        classify_search_privacy("q" * (MAX_PRIVACY_QUERY_CHARS + 1))


def test_a_term_count_the_search_would_refuse_is_not_reported() -> None:
    """A count true about the string and false about the system is not a fact."""
    assert classify_search_privacy(AT_CAP_TERMS).term_count == MAX_PRIVACY_TERMS
    with pytest.raises(SearchPrivacyError):
        classify_search_privacy(OVER_CAP_TERMS)


@pytest.mark.parametrize("query", [None, 1, b"needle", ["needle"], object()])
def test_privacy_refuses_anything_that_is_not_text(query: object) -> None:
    with pytest.raises(SearchPrivacyError):
        classify_search_privacy(query)


def test_an_empty_query_measures_zero_rather_than_refusing() -> None:
    facts = classify_search_privacy("")
    assert facts.query_char_count == 0
    assert facts.term_count == 0


def test_the_privacy_record_carries_no_query_text() -> None:
    """The whole point of the record: facts about a query, never the query."""
    facts = classify_search_privacy("a distinctive secret phrase")
    values = [getattr(facts, item.name) for item in dataclasses.fields(facts)]
    assert not any(isinstance(value, str) and "secret" in value for value in values)


def test_no_caller_can_claim_permission_to_log_or_echo_the_query() -> None:
    facts = classify_search_privacy("needle")
    assert facts.may_log_raw_query is False
    assert facts.may_put_raw_query_in_url is False
    assert facts.may_echo_raw_query is False

    with pytest.raises(TypeError):
        SearchPrivacyV1(  # type: ignore[call-arg]
            schema="lm-atelier-search-privacy-v1",
            schema_version=1,
            term_count=1,
            query_char_count=6,
            may_log_raw_query=True,
        )
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.__setattr__("may_echo_raw_query", True)


def test_bounds_refuse_counts_the_search_would_refuse() -> None:
    assert (
        declare_search_resource_bounds(query_chars=MAX_QUERY_CHARS, term_count=1).query_chars == 200
    )
    with pytest.raises(SearchResourceBoundsError):
        declare_search_resource_bounds(query_chars=MAX_QUERY_CHARS + 1, term_count=1)
    with pytest.raises(SearchResourceBoundsError):
        declare_search_resource_bounds(query_chars=10, term_count=MAX_TERMS + 1)


def test_measured_counts_accept_zero_and_configured_ceilings_do_not() -> None:
    """The asymmetry is deliberate, so it is pinned rather than left to drift.

    query_chars and term_count are what a request MEASURED - an empty query is
    genuinely zero of both, and the compose layer supplies them straight from
    the privacy facts. limit, window and snippet_chars are what a search may
    USE, and zero of those describes a search that cannot return anything.
    """
    measured = declare_search_resource_bounds(query_chars=0, term_count=0)
    assert measured.query_chars == 0
    assert measured.term_count == 0

    for field_name in ("limit", "window", "snippet_chars"):
        with pytest.raises(SearchResourceBoundsError):
            declare_search_resource_bounds(**{field_name: 0}, query_chars=1, term_count=1)


@pytest.mark.parametrize("value", [True, False])
def test_a_bool_is_not_an_acceptable_count(value: bool) -> None:
    """bool is an int in Python, so it passes a naive type check."""
    with pytest.raises(SearchResourceBoundsError):
        declare_search_resource_bounds(query_chars=value, term_count=1)
    with pytest.raises(SearchResourceBoundsError):
        declare_search_resource_bounds(limit=value, query_chars=1, term_count=1)


@pytest.mark.parametrize("value", [None, "20", 1.5, [20]])
def test_bounds_refuse_anything_that_is_not_a_whole_number(value: object) -> None:
    with pytest.raises(SearchResourceBoundsError):
        declare_search_resource_bounds(query_chars=value, term_count=1)


def test_no_caller_can_authorize_query_execution_or_an_index_write() -> None:
    bounds = declare_search_resource_bounds(query_chars=6, term_count=1)
    assert bounds.query_execution_authorized is False
    assert bounds.fts_write_authorized is False

    with pytest.raises(TypeError):
        SearchResourceBoundsV1(  # type: ignore[call-arg]
            schema="lm-atelier-search-resource-bounds-v1",
            schema_version=1,
            limit=20,
            window=32,
            snippet_chars=160,
            query_chars=6,
            term_count=1,
            fts_write_authorized=True,
        )
    with pytest.raises(dataclasses.FrozenInstanceError):
        bounds.__setattr__("query_execution_authorized", True)


def test_neither_module_publishes_a_private_roadmap_coordinate() -> None:
    """The hygiene ratchet on trunk refuses a diff that adds one.

    Both docstrings carried one, so copying either module forward unchanged
    would have failed the gate on a line nobody was reading.
    """
    import re

    import local_lm.search_privacy_v1 as privacy
    import local_lm.search_resource_bounds_v1 as bounds

    coordinate = re.compile(r"\((?:roadmap )?item \d+\)")
    for module in (privacy, bounds):
        assert module.__doc__ is not None
        assert coordinate.search(module.__doc__) is None
