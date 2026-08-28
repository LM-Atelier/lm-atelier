"""Hostile tests for the pure conversation-search query facts.

Written against the rejection of the earlier shape rather than against the happy
path: every test below corresponds to a way the previous module could be made to
produce a wrong answer or unbounded work, so each one fails if that shape comes
back.
"""

from __future__ import annotations

import dataclasses

import pytest

from local_lm.conversation_search_query_v1 import (
    MAX_BODY_CHARS,
    MAX_ID_CHARS,
    MAX_QUERY_CHARS,
    MAX_RANKED_ROWS,
    MAX_TERMS,
    MAX_TOTAL_BODY_CHARS,
    ConversationSearchError,
    SearchHitLocation,
    hit_location,
    normalize_search_query,
    query_terms,
    rank_candidate_bodies,
    rank_identity_bodies,
    term_coverage_score,
)


def test_the_module_exposes_no_snippet_builder() -> None:
    """Snippets belong to search_snippet_v1, which slices correctly.

    The earlier build_snippet found a match index in casefolded text and sliced
    the original, so the window drifted one character per expanding character
    before the match. Re-adding a snippet builder here would reintroduce a
    second implementation of something already solved elsewhere.
    """
    import local_lm.conversation_search_query_v1 as module

    assert not hasattr(module, "build_snippet")


@pytest.mark.parametrize(
    "raw",
    [None, 1, b"needle", ["needle"], "", "   ", "\t\n"],
)
def test_a_query_that_is_not_usable_text_is_refused(raw: object) -> None:
    with pytest.raises(ConversationSearchError):
        normalize_search_query(raw)


def test_a_query_over_the_cap_is_refused_before_and_after_collapsing() -> None:
    assert normalize_search_query("a" * MAX_QUERY_CHARS) == "a" * MAX_QUERY_CHARS
    with pytest.raises(ConversationSearchError):
        normalize_search_query("a" * (MAX_QUERY_CHARS + 1))


def test_whitespace_collapses_rather_than_creating_empty_terms() -> None:
    assert normalize_search_query("  two   words \n") == "two words"
    assert query_terms("  Two   WORDS ") == ("two", "words")


def test_more_terms_than_the_cap_are_refused() -> None:
    assert len(query_terms(" ".join(f"t{index}" for index in range(MAX_TERMS)))) == MAX_TERMS
    with pytest.raises(ConversationSearchError):
        query_terms(" ".join(f"t{index}" for index in range(MAX_TERMS + 1)))


def test_a_body_over_the_cap_is_refused_rather_than_scanned() -> None:
    assert term_coverage_score("x" * MAX_BODY_CHARS, "x") == 1
    with pytest.raises(ConversationSearchError):
        term_coverage_score("x" * (MAX_BODY_CHARS + 1), "x")


def test_scoring_counts_distinct_terms_and_folds_case() -> None:
    assert term_coverage_score("Alpha and BETA", "alpha beta") == 2
    assert term_coverage_score("Alpha only", "alpha beta") == 1
    assert term_coverage_score("nothing here", "alpha beta") == 0


def test_fixed_location_facts_cannot_be_set_by_a_caller() -> None:
    """activates_branch and loads_entire_chat are facts, not defaults.

    They are Literal[False] with init=False, so a caller cannot pass True and
    cannot mutate one afterwards. A fixed fact a constructor can be talked out
    of is not a fixed fact.
    """
    location = hit_location(message_id="m1", chat_id="c1")
    assert location.activates_branch is False
    assert location.loads_entire_chat is False

    with pytest.raises(TypeError):
        SearchHitLocation(  # type: ignore[call-arg]
            message_id="m1",
            chat_id="c1",
            project_id=None,
            activates_branch=True,
        )
    with pytest.raises(dataclasses.FrozenInstanceError):
        location.__setattr__("activates_branch", True)


@pytest.mark.parametrize(
    "bad",
    [None, 1, "", "has space", "tab\there", "x" * (MAX_ID_CHARS + 1)],
)
def test_identifiers_that_could_not_address_a_message_are_refused(bad: object) -> None:
    with pytest.raises(ConversationSearchError):
        hit_location(message_id=bad, chat_id="c1")
    with pytest.raises(ConversationSearchError):
        hit_location(message_id="m1", chat_id=bad)


def test_more_rows_than_the_cap_are_refused_before_any_scoring() -> None:
    rows = [(f"m{index}", "body with needle") for index in range(MAX_RANKED_ROWS)]
    assert len(rank_candidate_bodies(rows, "needle")) == MAX_RANKED_ROWS
    with pytest.raises(ConversationSearchError):
        rank_candidate_bodies(rows + [("overflow", "body with needle")], "needle")


def test_the_aggregate_text_cap_refuses_a_corpus_under_the_row_cap() -> None:
    """Rows alone are not a bound: few rows can still carry unbounded text."""
    chunk = "n" * MAX_BODY_CHARS
    count = (MAX_TOTAL_BODY_CHARS // MAX_BODY_CHARS) + 1
    assert count <= MAX_RANKED_ROWS
    rows = [(f"m{index}", chunk) for index in range(count)]
    with pytest.raises(ConversationSearchError):
        rank_candidate_bodies(rows, "n")


@pytest.mark.parametrize("candidates", [None, "rows", 5, {"m1": "body"}])
def test_a_corpus_that_is_not_a_sequence_of_rows_is_refused(candidates: object) -> None:
    with pytest.raises(ConversationSearchError):
        rank_candidate_bodies(candidates, "needle")
    with pytest.raises(ConversationSearchError):
        rank_identity_bodies(candidates, "needle")


def test_rows_of_the_wrong_width_are_refused() -> None:
    with pytest.raises(ConversationSearchError):
        rank_candidate_bodies([("m1", "body", "extra")], "body")
    with pytest.raises(ConversationSearchError):
        rank_identity_bodies([("m1", "body")], "body")


def test_duplicate_message_ids_keep_their_own_chat_identity() -> None:
    """The defect this shape exists to prevent.

    Keying results by message id alone let two rows sharing an id collapse onto
    one mapping, so both hits resolved to whichever chat was seen last. Identity
    travels with the row, so a duplicate id stays attached to its own chat.
    """
    ranked = rank_identity_bodies(
        [
            ("shared", "chat-one", "needle in the first chat"),
            ("shared", "chat-two", "needle in the second chat"),
        ],
        "needle",
    )
    assert len(ranked) == 2
    assert {(row[0], row[1]) for row in ranked} == {
        ("shared", "chat-one"),
        ("shared", "chat-two"),
    }


def test_ranking_is_deterministic_and_drops_rows_that_match_nothing() -> None:
    ranked = rank_candidate_bodies(
        [
            ("m2", "alpha beta"),
            ("m1", "alpha beta"),
            ("m3", "alpha only"),
            ("m4", "unrelated text"),
        ],
        "alpha beta",
    )
    assert [row[0] for row in ranked] == ["m1", "m2", "m3"]
    assert [row[2] for row in ranked] == [2, 2, 1]
