from __future__ import annotations

import pytest

from local_lm.prior_turn_edit_source_preserve_v1 import (
    INVALID_SOURCE_PRESERVE,
    MAX_ID,
    PriorTurnEditSourcePreserveError,
    PriorTurnEditSourcePreserveV1,
    declare_prior_turn_edit_source_preserve,
)


def test_records_requested_policy_without_mutating_source() -> None:
    preserve = declare_prior_turn_edit_source_preserve(source_message_id="m1")
    assert preserve.requested_policy == "no_source_mutation"
    assert preserve.source_mutation_authorized is False
    assert preserve.cancel_source_authorized is False


def test_refuses_unbounded_ids() -> None:
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id="bad id")
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id="m" * (MAX_ID + 1))
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id="m" * 10_000)
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id=True)
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id="m\ud800")


def test_public_constructor_cannot_authorize_mutation() -> None:
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        PriorTurnEditSourcePreserveV1()
    with pytest.raises(TypeError):
        PriorTurnEditSourcePreserveV1(
            schema="lm-atelier-prior-turn-edit-source-preserve-v1",
            schema_version=1,
            source_message_id="m1",
            source_mutation_authorized=True,
        )


def test_refuses_slash_dot_segments_and_controls() -> None:
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id="m1/extra")
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id="m1\\x")
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id=".")
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id="..")
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id="m1\x00x")
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id="m1\x7f")
    with pytest.raises(PriorTurnEditSourcePreserveError, match=INVALID_SOURCE_PRESERVE):
        declare_prior_turn_edit_source_preserve(source_message_id="m1\u202e")
