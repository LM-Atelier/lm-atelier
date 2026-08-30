from __future__ import annotations

from dataclasses import fields

import pytest

import local_lm.prior_turn_edit_snapshot_v1 as snapshot_module
from local_lm.prior_turn_edit_snapshot_v1 import (
    INVALID_SNAPSHOT,
    MAX_CONTEXT_BYTES,
    MAX_CONTEXT_COUNT,
    MAX_ID,
    PriorTurnEditSnapshotError,
    PriorTurnEditSnapshotV1,
    declare_prior_turn_edit_snapshot,
)

DIGEST = "a" * 64


class _ListSubclass(list[object]):
    pass


class _StringSubclass(str):
    pass


class _EqualWitness:
    def __eq__(self, other: object) -> bool:
        return True


def _declare(
    *,
    source_message_id: object = "m1",
    snapshot_digest: object = DIGEST,
    context_message_ids: object = ("c1", "c2"),
) -> PriorTurnEditSnapshotV1:
    return declare_prior_turn_edit_snapshot(
        source_message_id=source_message_id,
        snapshot_digest=snapshot_digest,
        context_message_ids=context_message_ids,
    )


def test_records_bounded_snapshot_facts_without_authority() -> None:
    snapshot = _declare(context_message_ids=["c2", "m1", "c1"])

    assert snapshot.schema == "lm-atelier-prior-turn-edit-snapshot-v1"
    assert snapshot.schema_version == 1
    assert snapshot.source_message_id == "m1"
    assert snapshot.snapshot_digest == DIGEST
    assert snapshot.context_message_ids == ("c2", "m1", "c1")
    assert snapshot.accepted is False
    assert snapshot.execution_authorized is False
    assert snapshot.late_bind_authorized is False


def test_authority_fields_are_closed_to_ordinary_construction() -> None:
    authority = {"accepted", "execution_authorized", "late_bind_authorized"}
    declared_fields = {item.name: item for item in fields(PriorTurnEditSnapshotV1)}
    assert all(declared_fields[name].init is False for name in authority)

    with pytest.raises(PriorTurnEditSnapshotError) as raised:
        PriorTurnEditSnapshotV1()
    assert raised.value.args == (INVALID_SNAPSHOT,)
    with pytest.raises(TypeError):
        PriorTurnEditSnapshotV1(accepted=True)  # type: ignore[call-arg]


def test_private_constructor_refuses_nonidentical_witness() -> None:
    with pytest.raises(PriorTurnEditSnapshotError) as raised:
        snapshot_module._snapshot_from_evaluator(
            witness=_EqualWitness(),
            source_message_id="m1",
            snapshot_digest=DIGEST,
            context_message_ids=("c1",),
        )

    assert raised.value.args == (INVALID_SNAPSHOT,)


@pytest.mark.parametrize(
    "candidate",
    [None, True, 1, _StringSubclass(DIGEST), "A" * 64, "a" * 63, "g" * 64],
)
def test_refuses_non_exact_lowercase_sha256(candidate: object) -> None:
    with pytest.raises(PriorTurnEditSnapshotError) as raised:
        _declare(snapshot_digest=candidate)
    assert raised.value.args == (INVALID_SNAPSHOT,)


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        "x" * (MAX_ID + 1),
        " leading",
        "trailing ",
        "embedded space",
        "m/child",
        "m\\child",
        ".",
        "..",
        "m\x00x",
        "m\x7f",
        "m\u202e",
        "m\ud800",
    ],
)
def test_refuses_ambiguous_source_identity(candidate: str) -> None:
    with pytest.raises(PriorTurnEditSnapshotError) as raised:
        _declare(source_message_id=candidate)
    assert raised.value.args == (INVALID_SNAPSHOT,)


@pytest.mark.parametrize(
    "candidate",
    ["c/child", "c\\child", ".", "..", "c\u202e", "c\ud800"],
)
def test_context_id_uses_same_captured_identity_grammar(candidate: str) -> None:
    with pytest.raises(PriorTurnEditSnapshotError) as raised:
        _declare(context_message_ids=[candidate])
    assert raised.value.args == (INVALID_SNAPSHOT,)


def test_exact_identity_and_collection_bounds_are_accepted() -> None:
    exact_ids = [f"c{index:02d}" for index in range(MAX_CONTEXT_COUNT)]
    snapshot = _declare(
        source_message_id="s" * MAX_ID,
        context_message_ids=exact_ids,
    )
    assert snapshot.source_message_id == "s" * MAX_ID
    assert snapshot.context_message_ids == tuple(exact_ids)

    exact_bytes = [
        chr(0x100 + index) * MAX_ID for index in range(MAX_CONTEXT_BYTES // (MAX_ID * 2))
    ]
    assert sum(len(item.encode("utf-8")) for item in exact_bytes) == MAX_CONTEXT_BYTES
    assert _declare(context_message_ids=exact_bytes).context_message_ids == tuple(exact_bytes)


def test_refuses_context_count_and_aggregate_byte_overflow() -> None:
    over_count = [f"c{index:02d}" for index in range(MAX_CONTEXT_COUNT + 1)]
    with pytest.raises(PriorTurnEditSnapshotError, match=INVALID_SNAPSHOT):
        _declare(context_message_ids=over_count)

    exact_bytes = [
        chr(0x100 + index) * MAX_ID for index in range(MAX_CONTEXT_BYTES // (MAX_ID * 2))
    ]
    with pytest.raises(PriorTurnEditSnapshotError, match=INVALID_SNAPSHOT):
        _declare(context_message_ids=[*exact_bytes, "x"])


@pytest.mark.parametrize(
    "candidate",
    [None, {"c1"}, "c1", _ListSubclass(["c1"]), ["c1", True]],
)
def test_refuses_non_exact_context_collections_and_items(candidate: object) -> None:
    with pytest.raises(PriorTurnEditSnapshotError, match=INVALID_SNAPSHOT):
        _declare(context_message_ids=candidate)


def test_context_collection_is_owned_ordered_and_unique() -> None:
    context = ["c2", "c1"]
    snapshot = _declare(context_message_ids=context)
    context.append("late")

    assert snapshot.context_message_ids == ("c2", "c1")
    with pytest.raises(PriorTurnEditSnapshotError, match=INVALID_SNAPSHOT):
        _declare(context_message_ids=["c1", "c1"])


def test_context_bounds_cover_items_added_during_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ["c1"]
    original_utf8_size = snapshot_module._utf8_size
    mutated = False

    def mutate_during_first_item(value: str) -> int:
        nonlocal mutated
        if not mutated:
            mutated = True
            context.extend(f"late-{index}" for index in range(MAX_CONTEXT_COUNT))
        return original_utf8_size(value)

    monkeypatch.setattr(snapshot_module, "_utf8_size", mutate_during_first_item)
    with pytest.raises(PriorTurnEditSnapshotError, match=INVALID_SNAPSHOT):
        _declare(context_message_ids=context)


def test_refusal_never_echoes_hostile_input() -> None:
    secret = "private-value/child"
    with pytest.raises(PriorTurnEditSnapshotError) as raised:
        _declare(source_message_id=secret)
    assert raised.value.args == (INVALID_SNAPSHOT,)
    assert secret not in str(raised.value)
