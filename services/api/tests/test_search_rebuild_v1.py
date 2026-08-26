from __future__ import annotations

import pytest

import local_lm.search_rebuild_v1 as rebuild_mod
from local_lm.search_rebuild_v1 import (
    INVALID_REBUILD,
    MAX_GENERATION,
    MAX_SEQUENCE,
    SearchRebuildError,
    SearchRebuildV1,
    declare_search_rebuild,
)
from local_lm.search_text_v1 import MAX_SEARCH_TOKEN_CHARS


def test_declares_watermark_without_write_authority() -> None:
    rebuild = declare_search_rebuild(
        phase="shadow",
        start_sequence=10,
        end_sequence=10,
        from_generation=3,
        to_generation=4,
    )
    assert rebuild.phase == "shadow"
    assert rebuild.query_execution_authorized is False
    assert rebuild.fts_write_authorized is False
    assert rebuild.generation_swap_authorized is False
    catching = declare_search_rebuild(
        phase="catch_up",
        start_sequence=10,
        end_sequence=40,
        from_generation=3,
        to_generation=4,
    )
    assert catching.end_sequence == 40
    swap = declare_search_rebuild(
        phase="swap",
        start_sequence=10,
        end_sequence=40,
        from_generation=3,
        to_generation=4,
    )
    assert swap.phase == "swap"
    assert not hasattr(rebuild_mod, "MAX_PHASE_CHARS")


def test_refuses_inconsistent_and_unbounded_facts() -> None:
    with pytest.raises(SearchRebuildError, match=INVALID_REBUILD):
        declare_search_rebuild(
            phase="swap",
            start_sequence=20,
            end_sequence=10,
            from_generation=1,
            to_generation=2,
        )
    with pytest.raises(SearchRebuildError, match=INVALID_REBUILD):
        declare_search_rebuild(
            phase="shadow",
            start_sequence=1,
            end_sequence=1,
            from_generation=4,
            to_generation=4,
        )
    with pytest.raises(SearchRebuildError, match=INVALID_REBUILD):
        declare_search_rebuild(
            phase="x" * MAX_SEARCH_TOKEN_CHARS,
            start_sequence=1,
            end_sequence=1,
            from_generation=1,
            to_generation=2,
        )
    with pytest.raises(SearchRebuildError, match=INVALID_REBUILD):
        declare_search_rebuild(
            phase="x" * (MAX_SEARCH_TOKEN_CHARS + 1),
            start_sequence=1,
            end_sequence=1,
            from_generation=1,
            to_generation=2,
        )
    with pytest.raises(SearchRebuildError, match=INVALID_REBUILD):
        declare_search_rebuild(
            phase="x" * 10_000,
            start_sequence=1,
            end_sequence=1,
            from_generation=1,
            to_generation=2,
        )
    with pytest.raises(SearchRebuildError, match=INVALID_REBUILD):
        declare_search_rebuild(
            phase="shadow",
            start_sequence=MAX_SEQUENCE + 1,
            end_sequence=MAX_SEQUENCE + 1,
            from_generation=1,
            to_generation=2,
        )
    with pytest.raises(SearchRebuildError, match=INVALID_REBUILD):
        declare_search_rebuild(
            phase="shadow",
            start_sequence=1,
            end_sequence=1,
            from_generation=MAX_GENERATION,
            to_generation=MAX_GENERATION + 1,
        )
    with pytest.raises(SearchRebuildError, match=INVALID_REBUILD):
        declare_search_rebuild(
            phase="shadow",
            start_sequence=True,
            end_sequence=1,
            from_generation=1,
            to_generation=2,
        )


def test_public_constructor_cannot_mint_swap_authority() -> None:
    with pytest.raises(SearchRebuildError, match=INVALID_REBUILD):
        SearchRebuildV1()
    with pytest.raises(TypeError):
        SearchRebuildV1(
            schema="lm-atelier-search-rebuild-v1",
            schema_version=1,
            phase="swap",
            start_sequence=1,
            end_sequence=2,
            from_generation=1,
            to_generation=2,
            generation_swap_authorized=True,
        )
