from __future__ import annotations

import pytest

from local_lm.search_index_status_v1 import (
    INVALID_STATUS,
    MAX_GENERATION,
    SearchIndexStatusError,
    SearchIndexStatusV1,
    declare_search_index_status,
)


def test_ready_building_and_degraded() -> None:
    ready = declare_search_index_status(
        state="ready",
        generation=3,
        indexed_through=12,
        detail_code="ok",
    )
    assert ready.state == "ready"
    assert ready.query_execution_authorized is False
    assert ready.fts_write_authorized is False
    assert ready.index_rebuild_authorized is False
    building = declare_search_index_status(
        state="building",
        generation=4,
        indexed_through=12,
        detail_code="building",
    )
    assert building.state == "building"
    degraded = declare_search_index_status(
        state="degraded",
        generation=4,
        indexed_through=12,
        detail_code="missing_projection",
    )
    assert degraded.detail_code == "missing_projection"


def test_refuses_inconsistent_and_unbounded_facts() -> None:
    with pytest.raises(SearchIndexStatusError, match=INVALID_STATUS):
        declare_search_index_status(
            state="ready",
            generation=1,
            indexed_through=1,
            detail_code="building",
        )
    with pytest.raises(SearchIndexStatusError, match=INVALID_STATUS):
        declare_search_index_status(
            state="degraded",
            generation=1,
            indexed_through=1,
            detail_code="ok",
        )
    with pytest.raises(SearchIndexStatusError, match=INVALID_STATUS):
        declare_search_index_status(
            state="ready",
            generation=MAX_GENERATION + 1,
            indexed_through=1,
            detail_code="ok",
        )
    with pytest.raises(SearchIndexStatusError, match=INVALID_STATUS):
        declare_search_index_status(
            state="ready",
            generation=True,
            indexed_through=1,
            detail_code="ok",
        )
    with pytest.raises(SearchIndexStatusError, match=INVALID_STATUS):
        declare_search_index_status(
            state="ready",
            generation=1,
            indexed_through=1,
            detail_code="secret payload leaked",
        )


def test_public_constructor_cannot_mint_ready_status() -> None:
    with pytest.raises(SearchIndexStatusError, match=INVALID_STATUS):
        SearchIndexStatusV1()
    with pytest.raises(TypeError):
        SearchIndexStatusV1(
            state="ready",
            generation=1,
            indexed_through=1,
            detail_code="ok",
            query_execution_authorized=True,
        )
