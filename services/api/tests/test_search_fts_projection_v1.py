from __future__ import annotations

import pytest

import local_lm.search_fts_projection_v1 as projection_mod
from local_lm.search_fts_projection_v1 import (
    INVALID_PROJECTION,
    MAX_DOCUMENTS,
    MAX_GENERATION,
    SearchFtsProjectionError,
    SearchFtsProjectionV1,
    declare_search_fts_projection,
)
from local_lm.search_text_v1 import MAX_SEARCH_TOKEN_CHARS


def test_declares_local_cache_without_export() -> None:
    projection = declare_search_fts_projection(
        projection_name="conversation-fts-v1",
        generation=3,
        document_count=12,
    )
    assert projection.projection_name == "conversation-fts-v1"
    assert projection.export_authorized is False
    assert projection.fts_write_authorized is False
    assert projection.query_execution_authorized is False
    assert not hasattr(projection_mod, "MAX_NAME_CHARS")


def test_refuses_unknown_and_unbounded_facts() -> None:
    with pytest.raises(SearchFtsProjectionError, match=INVALID_PROJECTION):
        declare_search_fts_projection(
            projection_name="conversation-fts-v2",
            generation=1,
            document_count=1,
        )
    with pytest.raises(SearchFtsProjectionError, match=INVALID_PROJECTION):
        declare_search_fts_projection(
            projection_name="x" * MAX_SEARCH_TOKEN_CHARS,
            generation=1,
            document_count=1,
        )
    with pytest.raises(SearchFtsProjectionError, match=INVALID_PROJECTION):
        declare_search_fts_projection(
            projection_name="x" * (MAX_SEARCH_TOKEN_CHARS + 1),
            generation=1,
            document_count=1,
        )
    with pytest.raises(SearchFtsProjectionError, match=INVALID_PROJECTION):
        declare_search_fts_projection(
            projection_name="x" * 10_000,
            generation=1,
            document_count=1,
        )
    with pytest.raises(SearchFtsProjectionError, match=INVALID_PROJECTION):
        declare_search_fts_projection(
            projection_name="conversation-fts-v1",
            generation=MAX_GENERATION + 1,
            document_count=1,
        )
    with pytest.raises(SearchFtsProjectionError, match=INVALID_PROJECTION):
        declare_search_fts_projection(
            projection_name="conversation-fts-v1",
            generation=1,
            document_count=MAX_DOCUMENTS + 1,
        )
    with pytest.raises(SearchFtsProjectionError, match=INVALID_PROJECTION):
        declare_search_fts_projection(
            projection_name="conversation-fts-v1",
            generation=True,
            document_count=1,
        )


def test_public_constructor_cannot_authorize_export() -> None:
    with pytest.raises(SearchFtsProjectionError, match=INVALID_PROJECTION):
        SearchFtsProjectionV1()
    with pytest.raises(TypeError):
        SearchFtsProjectionV1(
            schema="lm-atelier-search-fts-projection-v1",
            schema_version=1,
            projection_name="conversation-fts-v1",
            generation=1,
            document_count=1,
            export_authorized=True,
        )
