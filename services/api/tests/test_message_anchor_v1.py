from __future__ import annotations

import pytest

from local_lm.message_anchor_v1 import (
    INVALID_ANCHOR,
    MAX_FRAGMENT_CHARS,
    MAX_ID_CHARS,
    AnchorError,
    MessageAnchorV1,
    encode_message_anchor,
    parse_message_anchor,
)


def test_round_trip() -> None:
    frag = encode_message_anchor("abc123")
    assert frag == "msg=abc123"
    anchor = parse_message_anchor("#" + frag)
    assert anchor.message_id == "abc123"
    assert anchor.changes_active_branch is False
    assert anchor.history_write_authorized is False
    assert parse_message_anchor(frag).message_id == "abc123"
    with pytest.raises(AnchorError, match=INVALID_ANCHOR):
        encode_message_anchor("bad id")
    with pytest.raises(AnchorError, match=INVALID_ANCHOR):
        parse_message_anchor("nope")


def test_refuses_hostile_and_oversize() -> None:
    with pytest.raises(AnchorError, match=INVALID_ANCHOR):
        encode_message_anchor("x" * (MAX_ID_CHARS + 1))
    with pytest.raises(AnchorError, match=INVALID_ANCHOR):
        parse_message_anchor("#" + ("x" * (MAX_FRAGMENT_CHARS + 1)))
    with pytest.raises(AnchorError, match=INVALID_ANCHOR):
        parse_message_anchor("msg=abc/def")

    class HostileStr(str):
        def startswith(self, *args, **kwargs):
            raise RuntimeError("private attacker detail")

    with pytest.raises(AnchorError, match=INVALID_ANCHOR) as caught:
        parse_message_anchor(HostileStr("msg=abc123"))
    assert "private attacker detail" not in str(caught.value)
    with pytest.raises(TypeError):
        MessageAnchorV1(message_id="abc123", changes_active_branch=True)
