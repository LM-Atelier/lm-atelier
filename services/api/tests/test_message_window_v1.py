from __future__ import annotations

import pytest

from local_lm.message_window_v1 import (
    DEFAULT_WINDOW,
    INVALID_WINDOW,
    MAX_ID_CHARS,
    MAX_INPUT_CHARS,
    MAX_INPUT_IDS,
    MAX_WINDOW,
    MessageWindowError,
    MessageWindowPlan,
    plan_message_window,
)

IDS = tuple(f"msg-{i:02d}" for i in range(1, 21))


def _refuse(*args, **kwargs) -> None:
    with pytest.raises(MessageWindowError, match=INVALID_WINDOW) as caught:
        plan_message_window(*args, **kwargs)
    assert str(caught.value) == INVALID_WINDOW


def test_latest_and_bounds() -> None:
    plan = plan_message_window(IDS, mode="latest", limit=5)
    assert plan.message_ids == IDS[-5:]
    assert plan.anchor_id is None
    assert plan.loads_entire_chat is False
    assert plan.branch_activation_authorized is False
    _refuse(IDS, mode="latest", limit=0)
    _refuse(IDS, mode="latest", limit=MAX_WINDOW + 1)
    _refuse(IDS, mode="latest", limit=True)
    _refuse(IDS, mode="latest", limit=5, anchor_id="msg-10")


def test_older_newer_around() -> None:
    older = plan_message_window(IDS, mode="older", anchor_id="msg-10", limit=3)
    assert older.message_ids == ("msg-07", "msg-08", "msg-09")
    assert older.anchor_id == "msg-10"
    newer = plan_message_window(IDS, mode="newer", anchor_id="msg-10", limit=3)
    assert newer.message_ids == ("msg-11", "msg-12", "msg-13")
    around = plan_message_window(IDS, mode="around", anchor_id="msg-10", limit=5)
    assert "msg-10" in around.message_ids
    assert len(around.message_ids) == 5
    assert around.loads_entire_chat is False


def test_default_limit_and_bad_anchor() -> None:
    plan = plan_message_window(IDS, mode="latest")
    assert len(plan.message_ids) == min(DEFAULT_WINDOW, len(IDS))
    _refuse(IDS, mode="older", anchor_id="missing")
    _refuse(IDS, mode="older", anchor_id="bad id")
    _refuse(IDS, mode="older")
    _refuse(IDS, mode="nope")
    _refuse([], mode="latest")
    _refuse("msg-01", mode="latest")


def test_refuses_over_input_count_before_item_work() -> None:
    over = tuple(f"i{i:03d}" for i in range(MAX_INPUT_IDS + 1))
    _refuse(over, mode="latest")
    at_cap = tuple(f"i{i:03d}" for i in range(MAX_INPUT_IDS))
    plan = plan_message_window(at_cap, mode="around", anchor_id=at_cap[-1], limit=5)
    assert at_cap[-1] in plan.message_ids
    assert len(plan.message_ids) == 5


def test_refuses_over_aggregate_input_chars() -> None:
    needed = (MAX_INPUT_CHARS // MAX_ID_CHARS) + 1
    assert needed <= MAX_INPUT_IDS
    ids = tuple(f"{i:03d}" + ("x" * (MAX_ID_CHARS - 3)) for i in range(needed))
    _refuse(ids, mode="latest")


def test_refuses_duplicate_and_hostile_identity() -> None:
    _refuse(("msg-01", "msg-01"), mode="latest")
    _refuse([f"x{'y' * MAX_ID_CHARS}"], mode="latest")


def test_refuses_hostile_subclasses() -> None:
    class HostileList(list):
        def __iter__(self):
            raise RuntimeError("private attacker detail")

    class HostileStr(str):
        def __contains__(self, item):
            raise RuntimeError("private attacker detail")

    _refuse(HostileList(list(IDS)), mode="latest")
    _refuse([HostileStr("msg-01"), "msg-02"], mode="latest")
    _refuse(IDS, mode=HostileStr("latest"))


def test_authority_flags_are_not_constructor_settable() -> None:
    with pytest.raises(TypeError):
        MessageWindowPlan(
            schema="lm-atelier-message-window-v1",
            schema_version=1,
            mode="latest",
            message_ids=("msg-01",),
            anchor_id=None,
            loads_entire_chat=True,
        )
    with pytest.raises(TypeError):
        MessageWindowPlan(
            schema="lm-atelier-message-window-v1",
            schema_version=1,
            mode="latest",
            message_ids=("msg-01",),
            anchor_id=None,
            branch_activation_authorized=True,
        )
