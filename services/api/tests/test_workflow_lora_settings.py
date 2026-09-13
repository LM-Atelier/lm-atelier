from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import pytest

from local_lm.settings_registry import IMAGE_SETTINGS, validate_settings
from local_lm.workflow_lora_overrides import (
    WorkflowLoraOverrideError,
    parse_workflow_lora_override_resolution,
    parse_workflow_lora_overrides,
    workflow_lora_override_resolution_payload,
    workflow_lora_override_resolution_sha256,
    workflow_lora_overrides_payload,
    workflow_lora_overrides_sha256,
)
from local_lm.workflow_lora_settings import (
    WORKFLOW_LORA_OVERRIDES_SETTING_KEY,
    WorkflowLoraSettingsError,
    overlay_workflow_lora_overrides,
    split_workflow_lora_overrides_setting,
    workflow_lora_override_resolution_as_overrides,
    workflow_lora_overrides_setting_value,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _slot_id(index: int) -> str:
    return f"wflora_{_digest(index)}"


def _slot(
    index: int,
    *,
    changes: dict[str, object] | None = None,
    authority: str | None = None,
) -> dict[str, object]:
    return {
        "slot_id": _slot_id(index),
        "loader_contract": "comfy-core-lora-loader-v1",
        "loader_authority_sha256": authority or _digest(100 + index),
        "changes": changes if changes is not None else {"model_strength": 0.5},
    }


def _target(
    index: int,
    *,
    definition: str | None = None,
    revision: str | None = None,
    overrides: list[object] | None = None,
) -> dict[str, object]:
    return {
        "workflow_family_id": f"wffamily-{index}",
        "workflow_definition_id": definition or f"workflow-{index}",
        "workflow_variant_key": "text_to_image",
        "workflow_revision_id": revision or f"wfrev-{index}",
        "slot_contract_version": 1,
        "revision_scope_sha256": _digest(index * 10 + 1),
        "api_graph_sha256": _digest(index * 10 + 2),
        "dependency_contract_sha256": _digest(index * 10 + 3),
        "activation_binding_sha256": _digest(index * 10 + 4),
        "activation_witness_sha256": _digest(index * 10 + 5),
        "overrides": overrides if overrides is not None else [_slot(index)],
    }


def _envelope(*targets: dict[str, object]) -> dict[str, object]:
    return {"version": 1, "targets": list(targets)}


def _resolution(
    *,
    target: dict[str, object] | None = None,
    overrides: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    raw_target = dict(target or _target(1))
    raw_target.pop("overrides")
    return {
        "version": 1,
        "target": raw_target,
        "overrides": overrides
        if overrides is not None
        else [
            {
                "slot_id": _slot_id(1),
                "loader_contract": "comfy-core-lora-loader-v1",
                "loader_authority_sha256": _digest(101),
                "changes": {
                    "clip_strength": {"value": 0.75, "origin": "chat"},
                    "model_strength": {"value": 1, "origin": "turn"},
                },
            }
        ],
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


_HOSTILE_UNKNOWN_KEYS = (
    "private\rkey",
    "private\nkey",
    "private\x00key",
    r"C:\private\model",
    r"\\server\share\model",
    "../private/model",
    r"..\private\model",
)
_UNKNOWN_KEY_NESTINGS = (
    "envelope",
    "envelope_target",
    "envelope_slot",
    "envelope_changes",
    "resolution",
    "resolution_target",
    "resolution_slot",
    "resolution_changes",
    "resolution_field",
)
_INVALID_PAYLOAD_FIELDS_MESSAGE = "Workflow LoRA override payload has unsupported or missing fields"


def _payload_with_unknown_key(
    nesting: str,
    key: str,
) -> tuple[dict[str, object], Callable[[object], object]]:
    if nesting.startswith("envelope"):
        payload = _envelope(_target(1))
        target = cast(dict[str, Any], cast(list[object], payload["targets"])[0])
        slot = cast(dict[str, Any], cast(list[object], target["overrides"])[0])
        changes = cast(dict[str, Any], slot["changes"])
        containers = {
            "envelope": payload,
            "envelope_target": target,
            "envelope_slot": slot,
            "envelope_changes": changes,
        }
        containers[nesting][key] = "private"
        return payload, parse_workflow_lora_overrides

    payload = _resolution()
    target = cast(dict[str, Any], payload["target"])
    slot = cast(dict[str, Any], cast(list[object], payload["overrides"])[0])
    changes = cast(dict[str, Any], slot["changes"])
    field = cast(dict[str, Any], changes["model_strength"])
    containers = {
        "resolution": payload,
        "resolution_target": target,
        "resolution_slot": slot,
        "resolution_changes": changes,
        "resolution_field": field,
    }
    containers[nesting][key] = "private"
    return payload, parse_workflow_lora_override_resolution


class _HashEqualityTrap:
    def __hash__(self) -> int:
        raise AssertionError("hostile slot id was hashed")

    def __eq__(self, other: object) -> bool:
        del other
        raise AssertionError("hostile slot id was compared")


class _HostileString(str):
    def __hash__(self) -> int:
        raise AssertionError("hostile string slot id was hashed")

    def __eq__(self, other: object) -> bool:
        del other
        raise AssertionError("hostile string slot id was compared")


def test_reserved_setting_is_split_and_canonicalized_before_generic_validation() -> None:
    raw_envelope = _envelope(
        _target(2, overrides=[_slot(3), _slot(2)]),
        _target(
            1,
            overrides=[
                _slot(
                    1,
                    changes={"clip_strength": 0.25, "model_strength": 1},
                )
            ],
        ),
    )
    settings = {
        "steps": 24,
        WORKFLOW_LORA_OVERRIDES_SETTING_KEY: raw_envelope,
    }

    parsed, ordinary = split_workflow_lora_overrides_setting(settings, role="image")

    assert ordinary == {"steps": 24}
    assert validate_settings(ordinary, IMAGE_SETTINGS)["steps"] == 24
    assert parsed is not None
    payload = workflow_lora_overrides_setting_value(parsed)
    assert [target["workflow_definition_id"] for target in payload["targets"]] == [
        "workflow-1",
        "workflow-2",
    ]
    assert [
        override["slot_id"]
        for override in cast(list[dict[str, Any]], payload["targets"])[1]["overrides"]
    ] == [_slot_id(2), _slot_id(3)]
    raw_envelope["targets"] = []
    assert len(parsed.targets) == 2


def test_absent_and_empty_reserved_values_distinguish_inheritance_from_reset() -> None:
    absent, ordinary = split_workflow_lora_overrides_setting({"steps": 10}, role="chat")
    empty, empty_ordinary = split_workflow_lora_overrides_setting(
        {WORKFLOW_LORA_OVERRIDES_SETTING_KEY: _envelope()},
        role="video",
    )

    assert absent is None
    assert ordinary == {"steps": 10}
    assert empty is not None and empty.targets == ()
    assert empty_ordinary == {}


def test_reserved_value_is_limited_to_media_request_roles() -> None:
    for role in ("chat", "load", "", cast(Any, type("Role", (str,), {})("image"))):
        with pytest.raises(WorkflowLoraSettingsError) as caught:
            split_workflow_lora_overrides_setting(
                {WORKFLOW_LORA_OVERRIDES_SETTING_KEY: _envelope()},
                role=role,
            )
        assert caught.value.code == "unsupported_workflow_lora_setting_role"


def test_settings_transport_rejects_outer_and_nested_builtin_subclasses() -> None:
    hostile_dict = type("HostileDict", (dict,), {})
    hostile_list = type("HostileList", (list,), {})

    with pytest.raises(WorkflowLoraSettingsError):
        split_workflow_lora_overrides_setting(
            cast(Any, hostile_dict()),
            role="image",
        )

    raw = _envelope(_target(1))
    raw["targets"] = hostile_list(cast(list[object], raw["targets"]))
    with pytest.raises(WorkflowLoraOverrideError):
        split_workflow_lora_overrides_setting(
            {WORKFLOW_LORA_OVERRIDES_SETTING_KEY: raw},
            role="image",
        )

    raw = _envelope(_target(1))
    target = cast(dict[str, object], cast(list[object], raw["targets"])[0])
    target["overrides"] = [hostile_dict(cast(dict[str, object], _slot(1)))]
    with pytest.raises(WorkflowLoraOverrideError):
        split_workflow_lora_overrides_setting(
            {WORKFLOW_LORA_OVERRIDES_SETTING_KEY: raw},
            role="image",
        )


@pytest.mark.parametrize("key", _HOSTILE_UNKNOWN_KEYS)
@pytest.mark.parametrize("nesting", _UNKNOWN_KEY_NESTINGS)
def test_unsupported_keys_at_every_nesting_are_fixed_and_never_echoed(
    nesting: str,
    key: str,
) -> None:
    payload, parser = _payload_with_unknown_key(nesting, key)

    with pytest.raises(WorkflowLoraOverrideError) as caught:
        parser(payload)

    assert str(caught.value) == _INVALID_PAYLOAD_FIELDS_MESSAGE
    assert caught.value.code == (
        "unsupported_workflow_lora_override_field"
        if nesting in {"envelope_changes", "resolution_changes"}
        else "invalid_workflow_lora_overrides"
    )
    assert key not in str(caught.value)


@pytest.mark.parametrize(
    "slot_id_factory",
    [
        pytest.param(list, id="list"),
        pytest.param(dict, id="dict"),
        pytest.param(_HashEqualityTrap, id="hash-equality-trap"),
        pytest.param(
            lambda: _HostileString(_slot_id(1)),
            id="string-subclass-hash-equality-trap",
        ),
        pytest.param(lambda: "../private", id="traversal-string"),
        pytest.param(lambda: "wflora_" + "a" * 63, id="short-slot-token"),
    ],
)
def test_resolution_slot_id_is_refused_before_hash_or_equality_in_every_consumer(
    slot_id_factory: Callable[[], object],
) -> None:
    slot_id = slot_id_factory()
    resolution = parse_workflow_lora_override_resolution(_resolution())
    forged_override = replace(
        resolution.overrides[0],
        slot_id=cast(Any, slot_id),
    )
    forged = replace(resolution, overrides=(forged_override,))
    consumers: tuple[Callable[[], object], ...] = (
        lambda: workflow_lora_override_resolution_payload(forged),
        lambda: workflow_lora_override_resolution_sha256(forged),
        lambda: workflow_lora_override_resolution_as_overrides(forged),
    )

    for consume in consumers:
        with pytest.raises(WorkflowLoraOverrideError) as caught:
            consume()
        assert caught.value.code == "invalid_workflow_lora_override_resolution"
        assert str(caught.value) == "Workflow LoRA override resolution has invalid slots"

    raw = _resolution()
    cast(list[dict[str, Any]], raw["overrides"])[0]["slot_id"] = slot_id
    with pytest.raises(WorkflowLoraOverrideError) as parsed:
        parse_workflow_lora_override_resolution(raw)
    assert parsed.value.code == "invalid_workflow_lora_override_resolution"
    assert str(parsed.value) == "Workflow LoRA override resolution has invalid slots"


@pytest.mark.parametrize(
    "slot_id_factory",
    [
        pytest.param(list, id="list"),
        pytest.param(dict, id="dict"),
        pytest.param(_HashEqualityTrap, id="hash-equality-trap"),
        pytest.param(
            lambda: _HostileString(_slot_id(1)),
            id="string-subclass-hash-equality-trap",
        ),
    ],
)
def test_durable_envelope_consumers_refuse_hostile_slot_ids_without_touching_them(
    slot_id_factory: Callable[[], object],
) -> None:
    envelope = parse_workflow_lora_overrides(_envelope(_target(1)))
    target = envelope.targets[0]
    forged_slot = replace(
        target.overrides[0],
        slot_id=cast(Any, slot_id_factory()),
    )
    forged = replace(
        envelope,
        targets=(replace(target, overrides=(forged_slot,)),),
    )
    consumers: tuple[Callable[[], object], ...] = (
        lambda: workflow_lora_overrides_payload(forged),
        lambda: workflow_lora_overrides_setting_value(forged),
        lambda: overlay_workflow_lora_overrides(forged),
    )

    for consume in consumers:
        with pytest.raises(WorkflowLoraOverrideError) as caught:
            consume()
        assert caught.value.code == "invalid_workflow_lora_override_slot"
        assert str(caught.value) == "Workflow LoRA override slot id is invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workflow_family_id", "../private"),
        ("workflow_definition_id", "folder/private"),
        ("workflow_variant_key", r"folder\private"),
        ("workflow_revision_id", r"C:\private"),
        ("workflow_revision_id", "C:private"),
    ],
)
def test_every_durable_target_uses_safe_tokens_without_echoing_paths(
    field: str,
    value: str,
) -> None:
    active = _target(1)
    inactive = _target(2)
    inactive[field] = value

    with pytest.raises(WorkflowLoraOverrideError) as caught:
        split_workflow_lora_overrides_setting(
            {WORKFLOW_LORA_OVERRIDES_SETTING_KEY: _envelope(active, inactive)},
            role="image",
        )

    assert caught.value.code == "invalid_workflow_lora_override_identity"
    assert value not in str(caught.value)


def test_field_wise_overlay_preserves_sibling_fields_slots_and_inactive_targets() -> None:
    lower = parse_workflow_lora_overrides(
        _envelope(
            _target(
                1,
                overrides=[
                    _slot(
                        1,
                        changes={"model_strength": 0.25, "clip_strength": 0.4},
                    ),
                    _slot(3, changes={"enabled": False}),
                ],
            ),
            _target(2, overrides=[_slot(2, changes={"model_strength": 0.2})]),
        )
    )
    higher = parse_workflow_lora_overrides(
        _envelope(
            _target(
                1,
                overrides=[
                    _slot(1, changes={"model_strength": 0.9}),
                    _slot(4, changes={"clip_strength": 0.8}),
                ],
            )
        )
    )

    result = overlay_workflow_lora_overrides(lower, None, higher)
    payload = workflow_lora_overrides_payload(result)

    assert len(result.targets) == 2
    by_definition = {
        cast(str, target["workflow_definition_id"]): cast(list[dict[str, Any]], target["overrides"])
        for target in cast(list[dict[str, Any]], payload["targets"])
    }
    current = {item["slot_id"]: item for item in by_definition["workflow-1"]}
    assert current[_slot_id(1)]["changes"] == {
        "model_strength": 0.9,
        "clip_strength": 0.4,
    }
    assert current[_slot_id(3)]["changes"] == {"enabled": False}
    assert current[_slot_id(4)]["changes"] == {"clip_strength": 0.8}
    assert by_definition["workflow-2"][0]["changes"] == {"model_strength": 0.2}


def test_overlay_reset_and_conflicting_slot_evidence_are_explicit() -> None:
    lower = parse_workflow_lora_overrides(_envelope(_target(1)))
    reset = parse_workflow_lora_overrides(_envelope())
    higher = parse_workflow_lora_overrides(
        _envelope(_target(2, overrides=[_slot(2, changes={"clip_strength": 0.6})]))
    )

    assert overlay_workflow_lora_overrides(lower, reset).targets == ()
    assert overlay_workflow_lora_overrides(lower, reset, None, higher) == higher

    conflicting = parse_workflow_lora_overrides(
        _envelope(
            _target(
                1,
                overrides=[_slot(1, changes={"clip_strength": 0.5}, authority=_digest(999))],
            )
        )
    )
    with pytest.raises(WorkflowLoraSettingsError) as caught:
        overlay_workflow_lora_overrides(lower, conflicting)
    assert caught.value.code == "conflicting_workflow_lora_override_slot_evidence"

    hostile_type = type("HostileEnvelope", (type(lower),), {})
    hostile = hostile_type(lower.version, lower.targets)
    with pytest.raises(WorkflowLoraOverrideError):
        overlay_workflow_lora_overrides(cast(Any, hostile))


def test_resolution_parser_is_exact_canonical_and_digest_stable() -> None:
    raw = _resolution(
        overrides=[
            {
                "slot_id": _slot_id(2),
                "loader_contract": "comfy-core-lora-loader-v1",
                "loader_authority_sha256": _digest(102),
                "changes": {"enabled": {"value": False, "origin": "project"}},
            },
            cast(dict[str, object], _resolution()["overrides"][0]),
        ]
    )

    parsed = parse_workflow_lora_override_resolution(raw)
    payload = workflow_lora_override_resolution_payload(parsed)

    assert [item["slot_id"] for item in cast(list[dict[str, Any]], payload["overrides"])] == [
        _slot_id(1),
        _slot_id(2),
    ]
    first_changes = cast(list[dict[str, Any]], payload["overrides"])[0]["changes"]
    assert list(first_changes) == ["model_strength", "clip_strength"]
    assert parsed.overrides[0].changes[0].value == 1.0
    assert parsed.inactive_targets == ()
    assert (
        workflow_lora_override_resolution_sha256(parsed)
        == hashlib.sha256(_canonical_json(payload)).hexdigest()
    )
    assert parse_workflow_lora_override_resolution(payload) == parsed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_strength", True),
        ("clip_strength", False),
        ("enabled", 1),
        ("enabled", 0.0),
        ("model_strength", float("nan")),
        ("clip_strength", float("inf")),
    ],
)
def test_resolution_parser_rejects_boolean_numeric_confusion_and_nonfinite_values(
    field: str,
    value: object,
) -> None:
    raw = _resolution()
    override = cast(list[dict[str, Any]], raw["overrides"])[0]
    override["changes"] = {field: {"value": value, "origin": "turn"}}

    with pytest.raises(WorkflowLoraOverrideError):
        parse_workflow_lora_override_resolution(raw)


def test_resolution_parser_rejects_duplicates_private_fields_and_subclasses() -> None:
    baseline = _resolution()
    malformed = []

    duplicate = deepcopy(baseline)
    duplicate["overrides"] = [
        deepcopy(cast(list[object], baseline["overrides"])[0]),
        deepcopy(cast(list[object], baseline["overrides"])[0]),
    ]
    malformed.append(duplicate)

    private = deepcopy(baseline)
    private["node_id"] = "private-node"
    malformed.append(private)

    inactive = deepcopy(baseline)
    inactive["inactive_targets"] = []
    malformed.append(inactive)

    private_change = deepcopy(baseline)
    change = cast(
        dict[str, Any],
        cast(list[dict[str, Any]], private_change["overrides"])[0]["changes"],
    )["model_strength"]
    cast(dict[str, object], change)["path"] = "private/model.safetensors"
    malformed.append(private_change)

    hostile_dict = type("HostileDict", (dict,), {})
    hostile_list = type("HostileList", (list,), {})
    hostile_string = type("HostileString", (str,), {})
    nested_change = deepcopy(baseline)
    cast(
        dict[str, Any],
        cast(list[dict[str, Any]], nested_change["overrides"])[0]["changes"],
    )["model_strength"] = hostile_dict({"value": 1.0, "origin": "turn"})
    hostile_origin = deepcopy(baseline)
    cast(
        dict[str, Any],
        cast(list[dict[str, Any]], hostile_origin["overrides"])[0]["changes"],
    )["model_strength"] = {
        "value": 1.0,
        "origin": hostile_string("turn"),
    }
    malformed.extend(
        [
            hostile_dict(baseline),
            {**baseline, "overrides": hostile_list(cast(list[object], baseline["overrides"]))},
            nested_change,
            hostile_origin,
        ]
    )

    for value in malformed:
        with pytest.raises(WorkflowLoraOverrideError):
            parse_workflow_lora_override_resolution(value)


def test_resolution_conversion_has_zero_or_one_current_target_and_strips_origins() -> None:
    parsed = parse_workflow_lora_override_resolution(_resolution())
    effective = workflow_lora_override_resolution_as_overrides(parsed)
    payload = workflow_lora_overrides_payload(effective)

    assert len(effective.targets) == 1
    assert cast(list[dict[str, Any]], payload["targets"])[0]["overrides"][0]["changes"] == {
        "model_strength": 1.0,
        "clip_strength": 0.75,
    }
    assert "origin" not in json.dumps(payload)

    empty = parse_workflow_lora_override_resolution(_resolution(overrides=[]))
    assert workflow_lora_override_resolution_as_overrides(empty).targets == ()


def test_public_setting_payload_has_no_private_graph_fields_or_machine_paths() -> None:
    value = parse_workflow_lora_overrides(_envelope(_target(1)))
    serialized = json.dumps(workflow_lora_overrides_setting_value(value), sort_keys=True)

    for private_name in (
        "node_id",
        "locator",
        "entry_locator",
        "api_graph_json",
        "ui_graph_json",
        "runtime_reference",
        "safetensors",
    ):
        assert private_name not in serialized
    assert (
        workflow_lora_overrides_sha256(value)
        == hashlib.sha256(_canonical_json(workflow_lora_overrides_setting_value(value))).hexdigest()
    )
