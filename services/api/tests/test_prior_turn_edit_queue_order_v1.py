from __future__ import annotations

import pytest

from local_lm.prior_turn_edit_queue_order_v1 import (
    INVALID_QUEUE_ORDER,
    MAX_ID,
    PriorTurnEditQueueOrderError,
    PriorTurnEditQueueOrderV1,
    declare_prior_turn_edit_queue_order,
)


def test_records_requested_policy_without_reordering() -> None:
    order = declare_prior_turn_edit_queue_order(chat_id="c1")
    assert order.requested_policy == "append_new_ordinal"
    assert order.reorder_queue_authorized is False
    assert order.activate_branch is False


def test_refuses_unbounded_ids() -> None:
    with pytest.raises(PriorTurnEditQueueOrderError, match=INVALID_QUEUE_ORDER):
        declare_prior_turn_edit_queue_order(chat_id="bad id")
    with pytest.raises(PriorTurnEditQueueOrderError, match=INVALID_QUEUE_ORDER):
        declare_prior_turn_edit_queue_order(chat_id="c" * (MAX_ID + 1))
    with pytest.raises(PriorTurnEditQueueOrderError, match=INVALID_QUEUE_ORDER):
        declare_prior_turn_edit_queue_order(chat_id="c" * 10_000)
    with pytest.raises(PriorTurnEditQueueOrderError, match=INVALID_QUEUE_ORDER):
        declare_prior_turn_edit_queue_order(chat_id=True)
    with pytest.raises(PriorTurnEditQueueOrderError, match=INVALID_QUEUE_ORDER):
        declare_prior_turn_edit_queue_order(chat_id="c\ud800")


def test_public_constructor_cannot_authorize_reorder() -> None:
    with pytest.raises(PriorTurnEditQueueOrderError, match=INVALID_QUEUE_ORDER):
        PriorTurnEditQueueOrderV1()
    with pytest.raises(TypeError):
        PriorTurnEditQueueOrderV1(
            schema="lm-atelier-prior-turn-edit-queue-order-v1",
            schema_version=1,
            chat_id="c1",
            reorder_queue_authorized=True,
        )
