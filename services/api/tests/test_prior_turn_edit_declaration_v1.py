from __future__ import annotations

import pytest

from local_lm.prior_turn_edit_declaration_v1 import (
    INVALID_DECLARATION,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENT_COUNT,
    MAX_ID,
    PriorTurnEditDeclarationError,
    PriorTurnEditDeclarationV1,
    declare_prior_turn_edit,
)


def _facts(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "chat_id": "c1",
        "source_message_id": "m1",
        "parent_message_id": "p1",
        "idempotency_key": "k1",
        "replacement_content_digest": "a" * 64,
        "activate_branch": False,
    }
    base.update(over)
    return base


def test_declaration_never_accepts() -> None:
    declaration = declare_prior_turn_edit(_facts())
    assert declaration.accepted is False
    assert declaration.persisted is False
    assert declaration.repository_snapshot_verified is False
    assert declaration.execution_authorized is False
    assert declaration.queue_safe is False
    assert declaration.activate_branch is False
    assert declaration.attachment_mode == "omit_inherit"


def test_attachment_modes() -> None:
    assert (
        declare_prior_turn_edit(_facts(attachment_mode="empty_clear")).attachment_mode
        == "empty_clear"
    )
    declaration = declare_prior_turn_edit(
        _facts(attachment_mode="replace", attachment_ids=["a1", "a2"])
    )
    assert declaration.attachment_ids == ("a1", "a2")
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(attachment_mode="replace", attachment_ids=[]))
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(attachment_mode="omit_inherit", attachment_ids=["x"]))


def test_activate_must_be_false() -> None:
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(activate_branch=True))
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(replacement_content_digest="Z" * 64))
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(chat_id="c" * (MAX_ID + 1)))


def test_attachment_list_bounds() -> None:
    exact_count = [f"a{index:02d}" for index in range(MAX_ATTACHMENT_COUNT)]
    built = declare_prior_turn_edit(_facts(attachment_mode="replace", attachment_ids=exact_count))
    assert built.attachment_ids == tuple(exact_count)
    over_count = [f"a{index:02d}" for index in range(MAX_ATTACHMENT_COUNT + 1)]
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(attachment_mode="replace", attachment_ids=over_count))
    exact_bytes = [("x" * 127) + format(index, "x") for index in range(8)]
    assert sum(len(item.encode("utf-8")) for item in exact_bytes) == MAX_ATTACHMENT_BYTES
    assert declare_prior_turn_edit(
        _facts(attachment_mode="replace", attachment_ids=exact_bytes)
    ).attachment_ids == tuple(exact_bytes)
    over_bytes = [*exact_bytes, "z"]
    assert sum(len(item.encode("utf-8")) for item in over_bytes) == MAX_ATTACHMENT_BYTES + 1
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(attachment_mode="replace", attachment_ids=over_bytes))
    hostile = [f"h{index:04d}" for index in range(10_000)]
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(attachment_mode="replace", attachment_ids=hostile))


def test_malformed_utf8_refuses() -> None:
    bad = "bad\ud800"
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(attachment_mode="replace", attachment_ids=[bad]))
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(chat_id="c\ud800"))
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(source_message_id="m\ud800"))
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(parent_message_id="p\ud800"))
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(idempotency_key="k\ud800"))
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        declare_prior_turn_edit(_facts(source_revision_id="r\ud800"))


def test_public_constructor_cannot_accept() -> None:
    with pytest.raises(PriorTurnEditDeclarationError, match=INVALID_DECLARATION):
        PriorTurnEditDeclarationV1()
    with pytest.raises(TypeError):
        PriorTurnEditDeclarationV1(
            schema="lm-atelier-prior-turn-edit-declaration-v1",
            schema_version=1,
            accepted=True,
        )
