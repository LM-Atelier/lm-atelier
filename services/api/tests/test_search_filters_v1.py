from __future__ import annotations

import pytest

from local_lm.search_filters_v1 import (
    INVALID_FILTER,
    MAX_FILTER_KEYS,
    MAX_ID_CHARS,
    MAX_TIME_CHARS,
    SearchFilterError,
    SearchFiltersV1,
    validate_search_filters,
)


def test_valid_and_unknown_key() -> None:
    filters = validate_search_filters(
        {
            "project_id": "proj-1",
            "role": "user",
            "has_media": True,
            "since": "2026-08-01T00:00:00Z",
        }
    )
    assert filters.project_id == "proj-1"
    assert filters.role == "user"
    assert filters.has_media is True
    assert filters.query_execution_authorized is False
    assert filters.fts_write_authorized is False
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"extra": 1})
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"role": "narrator"})
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"has_media": 1})
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"since": "August"})


def test_empty_and_chat_only() -> None:
    filters = validate_search_filters({})
    assert filters.project_id is None and filters.chat_id is None
    second = validate_search_filters({"chat_id": "chat-9", "until": "2026-08-14"})
    assert second.chat_id == "chat-9"
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"chat_id": "bad id"})


def test_refuses_empty_non_string_and_oversize_ids() -> None:
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"chat_id": ""})
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"project_id": ""})
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"chat_id": 1})
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"chat_id": "x" * (MAX_ID_CHARS + 1)})


def test_refuses_empty_non_string_and_oversize_times() -> None:
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"since": ""})
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"until": ""})
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"since": 1})
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"since": "2" * (MAX_TIME_CHARS + 1)})


def test_refuses_times_that_omit_the_date_dashes() -> None:
    """A filter time is YYYY-MM-DD at the start, not merely ten characters.

    Empty, non-string, and oversize times are already pinned. "August" is
    shorter than ten characters, so it is refused by the length half of the
    same check. A ten-character string with a dash in the month slot but not
    the year slot, or the year slot but not the month slot, is inside that
    length floor and is not empty, so only the matching dash check refuses
    it. Mutating either dash check left that spelling accepted.
    """
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"since": "2026x08-01"})
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"until": "2026-08x01"})


def test_refuses_hostile_keys_and_constructor_authority() -> None:
    class HostileKey(str):
        def __eq__(self, other):
            raise RuntimeError("private attacker detail")

        def __hash__(self):
            return str.__hash__(self)

    with pytest.raises(SearchFilterError, match=INVALID_FILTER) as caught:
        validate_search_filters({HostileKey("chat_id"): "chat-1"})
    assert "private attacker detail" not in str(caught.value)
    with pytest.raises(TypeError):
        SearchFiltersV1(
            project_id=None,
            chat_id=None,
            role=None,
            has_media=None,
            since=None,
            until=None,
            query_execution_authorized=True,
        )


def test_refuses_a_non_dict_and_an_oversized_filter_map() -> None:
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters([])
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters("chat-1")
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({f"k{i}": i for i in range(MAX_FILTER_KEYS + 1)})


def test_refuses_a_non_string_role() -> None:
    with pytest.raises(SearchFilterError, match=INVALID_FILTER):
        validate_search_filters({"role": 1})
