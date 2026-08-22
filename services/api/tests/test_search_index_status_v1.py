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


@pytest.mark.parametrize(
    "state",
    [
        "frobnicate",
        "READY",
        "ready\x00",
        "\u202eready",
        "",
    ],
)
def test_unknown_or_hostile_states_are_refused(state: str) -> None:
    """An unknown state must be refused, never coerced into a known one.

    The consistency guards below the state check are written as
    ``narrowed == X and ...``, which is vacuously true for any state they do
    not name - so this refusal is the only thing standing between an unknown
    state and a minted status (claude/R1177 review finding). The detail code
    here is one that is VALID for degraded: pairing with "ok" would let a
    coerce-to-degraded defect hide behind the degraded/ok consistency guard,
    which refuses "ok" anyway. With "version_mismatch", coercion mints a
    status and the test catches it.
    """
    with pytest.raises(SearchIndexStatusError):
        declare_search_index_status(
            state=state,
            generation=1,
            indexed_through=1,
            detail_code="version_mismatch",
        )


def test_detail_code_cap_and_oversized_are_refused() -> None:
    """Pin the refusals at the cap, one past it, and far past it.

    The code orders the length ceiling before DETAIL_CODES membership so an
    attacker-sized string is never hashed (codex/R984). That ordering is not
    observable from the outcome - membership would refuse these too - so this
    test pins the refusals and the ordering is asserted by reading the code,
    not by this name.
    """
    for detail_code in ("x" * 40, "x" * 41, "x" * 100_000):
        with pytest.raises(SearchIndexStatusError):
            declare_search_index_status(
                state="ready",
                generation=1,
                indexed_through=1,
                detail_code=detail_code,
            )


def test_an_attacker_sized_state_is_refused() -> None:
    """Oversized exact text refuses via the equality chain, which compares
    against short literals and never hashes the input (codex/R984 class)."""
    with pytest.raises(SearchIndexStatusError):
        declare_search_index_status(
            state="ready" + "x" * 100_000,
            generation=1,
            indexed_through=1,
            detail_code="ok",
        )
