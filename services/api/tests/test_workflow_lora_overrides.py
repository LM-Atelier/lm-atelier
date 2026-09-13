from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from dataclasses import fields as dataclass_fields
from typing import Any, cast

import pytest

import local_lm.workflow_lora_overrides as overrides_module
from local_lm.auxiliary_assets import MAX_LORA_STRENGTH
from local_lm.workflow_lora_overrides import (
    MAX_WORKFLOW_LORA_OVERRIDE_TARGETS,
    MAX_WORKFLOW_LORA_OVERRIDES_PER_TARGET,
    MAX_WORKFLOW_LORA_OVERRIDES_TOTAL,
    WORKFLOW_LORA_OVERRIDE_ORIGINS,
    WorkflowLoraOverrideCatalog,
    WorkflowLoraOverrideError,
    WorkflowLoraOverrideLayer,
    WorkflowLoraOverrides,
    parse_workflow_lora_overrides,
    resolve_workflow_lora_override_layers,
    workflow_lora_override_resolution_payload,
    workflow_lora_override_resolution_sha256,
    workflow_lora_overrides_payload,
    workflow_lora_overrides_sha256,
)
from local_lm.workflow_lora_slots import WorkflowLoraAssetBinding, WorkflowLoraSlot


def _digest(index: int) -> str:
    return f"{index:064x}"


def _slot_id(index: int) -> str:
    return f"wflora_{_digest(index)}"


def _slot_override(
    index: int = 1,
    *,
    changes: dict[str, object] | None = None,
    loader_contract: str = "comfy-core-lora-loader-v1",
    authority: str | None = None,
) -> dict[str, object]:
    return {
        "slot_id": _slot_id(index),
        "loader_contract": loader_contract,
        "loader_authority_sha256": authority or _digest(90 + index),
        "changes": changes if changes is not None else {"model_strength": 0.75},
    }


def _target(
    *,
    revision: str = "wfrev-one",
    definition: str = "workflow-one",
    family: str | None = "wffamily-one",
    variant: str | None = "image",
    overrides: Sequence[object] | None = None,
    digest_offset: int = 0,
) -> dict[str, object]:
    return {
        "workflow_family_id": family,
        "workflow_definition_id": definition,
        "workflow_variant_key": variant,
        "workflow_revision_id": revision,
        "slot_contract_version": 1,
        "revision_scope_sha256": _digest(1 + digest_offset),
        "api_graph_sha256": _digest(2 + digest_offset),
        "dependency_contract_sha256": _digest(3 + digest_offset),
        "activation_binding_sha256": _digest(4 + digest_offset),
        "activation_witness_sha256": _digest(5 + digest_offset),
        "overrides": list(overrides) if overrides is not None else [_slot_override()],
    }


def _envelope(*targets: dict[str, object]) -> dict[str, object]:
    return {"version": 1, "targets": list(targets)}


def _parsed_target(**updates: object) -> Any:
    raw = _target()
    raw.update(updates)
    return parse_workflow_lora_overrides(_envelope(raw)).targets[0].witness


def _asset_binding(index: int = 1) -> WorkflowLoraAssetBinding:
    return WorkflowLoraAssetBinding(
        dependency_slot=f"lora-{index}",
        requirement_key="default",
        resource_identity_sha256=_digest(120 + index),
        runtime_reference=f"styles/{index}.safetensors",
        sha256=_digest(140 + index),
    )


def _catalog_slot(
    index: int = 1,
    *,
    loader_contract: str = "comfy-core-lora-loader-v1",
    authority: str | None = None,
    editable_fields: tuple[str, ...] = ("model_strength", "clip_strength"),
    editability: str = "editable",
    dependency_required: bool | None = False,
    default_enabled: bool | None = True,
    model_strength: float | None = 1.0,
    clip_strength: float | None = 1.0,
    asset: WorkflowLoraAssetBinding | None = None,
    has_asset: bool = True,
) -> WorkflowLoraSlot:
    return WorkflowLoraSlot(
        slot_id=_slot_id(index),
        position=index - 1,
        loader_type="LoraLoader",
        loader_contract=loader_contract,
        loader_authority_sha256=authority or _digest(90 + index),
        editability=cast(Any, editability),
        read_only_reason=None if editability == "editable" else "locked",
        dependency_required=dependency_required,
        observed_runtime_reference=f"styles/{index}.safetensors",
        asset_binding=(asset if asset is not None else _asset_binding(index))
        if has_asset
        else None,
        default_enabled=default_enabled,
        default_model_strength=model_strength,
        default_clip_strength=clip_strength,
        strength_mode="separate",
        editable_fields=cast(Any, editable_fields),
    )


def _layer(origin: str, *targets: dict[str, object]) -> WorkflowLoraOverrideLayer:
    return WorkflowLoraOverrideLayer(
        cast(Any, origin),
        parse_workflow_lora_overrides(_envelope(*targets)),
    )


def _subclass_copy(value: Any) -> Any:
    hostile_type = type(f"Hostile{type(value).__name__}", (type(value),), {})
    return hostile_type(*(getattr(value, field.name) for field in dataclass_fields(value)))


def test_empty_top_level_targets_are_the_canonical_reset() -> None:
    parsed = parse_workflow_lora_overrides({"targets": [], "version": 1})

    assert parsed == WorkflowLoraOverrides(version=1, targets=())
    assert workflow_lora_overrides_payload(parsed) == {"version": 1, "targets": []}
    assert (
        workflow_lora_overrides_sha256(parsed)
        == hashlib.sha256(b'{"targets":[],"version":1}').hexdigest()
    )


def test_target_slot_and_field_order_are_canonical_and_numbers_are_normalized() -> None:
    first = _target(
        revision="wfrev-z",
        digest_offset=20,
        overrides=[
            _slot_override(2, changes={"clip_strength": -0.0, "model_strength": 1}),
            _slot_override(1, changes={"model_strength": 0.5}),
        ],
    )
    second = _target(
        revision="wfrev-a",
        digest_offset=10,
        overrides=[_slot_override(3, changes={"enabled": False, "model_strength": 1.0})],
    )
    reversed_input = parse_workflow_lora_overrides(_envelope(first, second))
    canonical_input = parse_workflow_lora_overrides(
        _envelope(
            deepcopy(second),
            {
                **deepcopy(first),
                "overrides": list(reversed(cast(list[object], first["overrides"]))),
            },
        )
    )

    assert reversed_input == canonical_input
    assert workflow_lora_overrides_sha256(reversed_input) == workflow_lora_overrides_sha256(
        canonical_input
    )
    payload = workflow_lora_overrides_payload(reversed_input)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert workflow_lora_overrides_sha256(reversed_input) == hashlib.sha256(encoded).hexdigest()
    first_payload = cast(list[dict[str, Any]], payload["targets"])[1]
    assert [item["slot_id"] for item in first_payload["overrides"]] == [
        _slot_id(1),
        _slot_id(2),
    ]
    assert first_payload["overrides"][1]["changes"] == {
        "model_strength": 1.0,
        "clip_strength": 0.0,
    }


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda value: value.update(extra=True), "invalid_workflow_lora_overrides"),
        (
            lambda value: cast(dict[str, Any], value["targets"][0]).update(extra=True),
            "invalid_workflow_lora_overrides",
        ),
        (
            lambda value: cast(dict[str, Any], value["targets"][0]["overrides"][0]).update(
                node_id="private"
            ),
            "invalid_workflow_lora_overrides",
        ),
        (
            lambda value: cast(
                dict[str, Any], value["targets"][0]["overrides"][0]["changes"]
            ).update(path=["inputs", "strength_model"]),
            "unsupported_workflow_lora_override_field",
        ),
    ],
)
def test_every_envelope_level_forbids_extra_fields(mutate: Any, code: str) -> None:
    value = _envelope(_target())
    mutate(value)

    with pytest.raises(WorkflowLoraOverrideError) as caught:
        parse_workflow_lora_overrides(value)

    assert caught.value.code == code


@pytest.mark.parametrize(
    "value",
    [
        None,
        (),
        {"version": 1, "targets": ()},
        {"version": True, "targets": []},
        {"version": 2, "targets": []},
    ],
)
def test_parser_requires_exact_json_builtins_and_versions(value: object) -> None:
    with pytest.raises(WorkflowLoraOverrideError):
        parse_workflow_lora_overrides(value)

    class Object(dict[str, object]):
        pass

    with pytest.raises(WorkflowLoraOverrideError):
        parse_workflow_lora_overrides(Object(version=1, targets=[]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("activation_binding_sha256", None),
        ("activation_witness_sha256", "A" * 64),
        ("api_graph_sha256", "a" * 63),
        ("workflow_definition_id", "x" * 65),
        ("workflow_revision_id", "x" * 41),
        ("workflow_variant_key", "x" * 101),
        ("workflow_family_id", "bad\nfamily"),
        ("slot_contract_version", True),
        ("slot_contract_version", 2),
    ],
)
def test_complete_target_witness_is_required_and_bounded(field: str, value: object) -> None:
    target = _target()
    target[field] = value

    with pytest.raises(WorkflowLoraOverrideError):
        parse_workflow_lora_overrides(_envelope(target))


def test_exact_null_family_and_variant_are_valid_not_wildcards() -> None:
    parsed = parse_workflow_lora_overrides(_envelope(_target(family=None, variant=None)))

    witness = parsed.targets[0].witness
    assert witness.workflow_family_id is None
    assert witness.workflow_variant_key is None


@pytest.mark.parametrize(
    "value",
    [True, False, None, "1", [], {}, float("nan"), float("inf"), -float("inf"), 5, -5, 10**1000],
)
def test_strengths_reject_bools_null_non_numbers_nonfinite_and_out_of_range(
    value: object,
) -> None:
    with pytest.raises(WorkflowLoraOverrideError) as caught:
        parse_workflow_lora_overrides(
            _envelope(_target(overrides=[_slot_override(changes={"model_strength": value})]))
        )

    assert caught.value.code == "invalid_workflow_lora_override_strength"


@pytest.mark.parametrize("value", [-MAX_LORA_STRENGTH, MAX_LORA_STRENGTH, 0, -0.0, 1])
def test_strength_range_endpoints_and_json_integers_are_normalized(value: int | float) -> None:
    parsed = parse_workflow_lora_overrides(
        _envelope(_target(overrides=[_slot_override(changes={"model_strength": value})]))
    )

    change = parsed.targets[0].overrides[0].changes[0]
    assert type(change.value) is float
    assert change.value == float(value)
    if value == 0:
        assert json.dumps(workflow_lora_overrides_payload(parsed), allow_nan=False).find("-0.0") < 0


@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}])
def test_enabled_requires_an_exact_boolean(value: object) -> None:
    with pytest.raises(WorkflowLoraOverrideError) as caught:
        parse_workflow_lora_overrides(
            _envelope(_target(overrides=[_slot_override(changes={"enabled": value})]))
        )

    assert caught.value.code == "invalid_workflow_lora_override_enabled"


def test_empty_target_empty_changes_and_duplicate_records_refuse() -> None:
    with pytest.raises(WorkflowLoraOverrideError) as empty_target:
        parse_workflow_lora_overrides(_envelope(_target(overrides=[])))
    assert empty_target.value.code == "empty_workflow_lora_override_target"

    with pytest.raises(WorkflowLoraOverrideError) as empty_changes:
        parse_workflow_lora_overrides(_envelope(_target(overrides=[_slot_override(changes={})])))
    assert empty_changes.value.code == "empty_workflow_lora_override_changes"

    target = _target()
    with pytest.raises(WorkflowLoraOverrideError) as duplicate_target:
        parse_workflow_lora_overrides(_envelope(target, deepcopy(target)))
    assert duplicate_target.value.code == "duplicate_workflow_lora_override_target"

    duplicate = _slot_override()
    with pytest.raises(WorkflowLoraOverrideError) as duplicate_slot:
        parse_workflow_lora_overrides(
            _envelope(_target(overrides=[duplicate, deepcopy(duplicate)]))
        )
    assert duplicate_slot.value.code == "duplicate_workflow_lora_override_slot"


def test_target_per_target_total_and_canonical_byte_bounds_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [
        _target(revision=f"wfrev-{index}", digest_offset=index * 10)
        for index in range(MAX_WORKFLOW_LORA_OVERRIDE_TARGETS)
    ]
    assert len(parse_workflow_lora_overrides(_envelope(*targets)).targets) == (
        MAX_WORKFLOW_LORA_OVERRIDE_TARGETS
    )
    with pytest.raises(WorkflowLoraOverrideError) as too_many_targets:
        parse_workflow_lora_overrides(
            _envelope(
                *targets,
                _target(revision="wfrev-over", digest_offset=500),
            )
        )
    assert too_many_targets.value.code == "too_many_workflow_lora_override_targets"

    sixty_four = [_slot_override(index) for index in range(1, 65)]
    assert (
        len(
            parse_workflow_lora_overrides(_envelope(_target(overrides=sixty_four)))
            .targets[0]
            .overrides
        )
        == MAX_WORKFLOW_LORA_OVERRIDES_PER_TARGET
    )
    with pytest.raises(WorkflowLoraOverrideError) as per_target:
        parse_workflow_lora_overrides(
            _envelope(_target(overrides=[*sixty_four, _slot_override(65)]))
        )
    assert per_target.value.code == "too_many_workflow_lora_overrides_for_target"

    four_full_targets = [
        _target(
            revision=f"wfrev-full-{index}",
            digest_offset=100 + index * 10,
            overrides=deepcopy(sixty_four),
        )
        for index in range(4)
    ]
    assert (
        sum(
            len(item.overrides)
            for item in parse_workflow_lora_overrides(_envelope(*four_full_targets)).targets
        )
        == MAX_WORKFLOW_LORA_OVERRIDES_TOTAL
    )
    with pytest.raises(WorkflowLoraOverrideError) as total:
        parse_workflow_lora_overrides(
            _envelope(
                *four_full_targets,
                _target(
                    revision="wfrev-full-over",
                    digest_offset=900,
                    overrides=[_slot_override()],
                ),
            )
        )
    assert total.value.code == "too_many_workflow_lora_overrides"

    monkeypatch.setattr(overrides_module, "MAX_WORKFLOW_LORA_OVERRIDE_CANONICAL_BYTES", 64)
    with pytest.raises(WorkflowLoraOverrideError) as too_large:
        parse_workflow_lora_overrides(_envelope(_target()))
    assert too_large.value.code == "workflow_lora_overrides_too_large"


def test_parsed_values_are_frozen_and_typed_values_are_revalidated() -> None:
    parsed = parse_workflow_lora_overrides(_envelope(_target()))

    with pytest.raises(FrozenInstanceError):
        parsed.__setattr__("version", 2)

    malformed = replace(parsed, version=True)
    with pytest.raises(WorkflowLoraOverrideError):
        workflow_lora_overrides_payload(malformed)

    duplicate_change = parsed.targets[0].overrides[0].changes[0]
    duplicate_override = replace(
        parsed.targets[0].overrides[0],
        changes=(duplicate_change, duplicate_change),
    )
    duplicate_target = replace(parsed.targets[0], overrides=(duplicate_override,))
    with pytest.raises(WorkflowLoraOverrideError) as duplicate:
        workflow_lora_overrides_payload(replace(parsed, targets=(duplicate_target,)))
    assert duplicate.value.code == "duplicate_workflow_lora_override_field"

    invalid_field = replace(duplicate_change, field=cast(Any, ["model_strength"]))
    invalid_override = replace(parsed.targets[0].overrides[0], changes=(invalid_field,))
    invalid_target = replace(parsed.targets[0], overrides=(invalid_override,))
    with pytest.raises(WorkflowLoraOverrideError):
        workflow_lora_overrides_payload(replace(parsed, targets=(invalid_target,)))


def test_every_typed_boundary_refuses_subclasses() -> None:
    parsed = parse_workflow_lora_overrides(_envelope(_target()))
    target = parsed.targets[0]
    slot_override = target.overrides[0]
    change = slot_override.changes[0]
    witness = target.witness
    slot = _catalog_slot()
    catalog = WorkflowLoraOverrideCatalog(witness, (slot,))
    layer = _layer("turn", _target())
    resolution = resolve_workflow_lora_override_layers(
        catalog=catalog,
        layers=(layer,),
    )
    resolved_override = resolution.overrides[0]
    resolved_change = resolved_override.changes[0]
    binding = cast(WorkflowLoraAssetBinding, slot.asset_binding)

    hostile_calls: tuple[Callable[[], object], ...] = (
        lambda: workflow_lora_overrides_payload(cast(Any, _subclass_copy(parsed))),
        lambda: workflow_lora_overrides_payload(
            replace(parsed, targets=(cast(Any, _subclass_copy(target)),))
        ),
        lambda: workflow_lora_overrides_payload(
            replace(
                parsed,
                targets=(
                    replace(
                        target,
                        witness=cast(Any, _subclass_copy(witness)),
                    ),
                ),
            )
        ),
        lambda: workflow_lora_overrides_payload(
            replace(
                parsed,
                targets=(
                    replace(
                        target,
                        overrides=(cast(Any, _subclass_copy(slot_override)),),
                    ),
                ),
            )
        ),
        lambda: workflow_lora_overrides_payload(
            replace(
                parsed,
                targets=(
                    replace(
                        target,
                        overrides=(
                            replace(
                                slot_override,
                                changes=(cast(Any, _subclass_copy(change)),),
                            ),
                        ),
                    ),
                ),
            )
        ),
        lambda: resolve_workflow_lora_override_layers(
            catalog=cast(Any, _subclass_copy(catalog)),
            layers=(layer,),
        ),
        lambda: resolve_workflow_lora_override_layers(
            catalog=catalog,
            layers=(cast(Any, _subclass_copy(layer)),),
        ),
        lambda: resolve_workflow_lora_override_layers(
            catalog=WorkflowLoraOverrideCatalog(
                witness,
                (cast(Any, _subclass_copy(slot)),),
            ),
            layers=(layer,),
        ),
        lambda: resolve_workflow_lora_override_layers(
            catalog=WorkflowLoraOverrideCatalog(
                witness,
                (
                    replace(
                        slot,
                        asset_binding=cast(Any, _subclass_copy(binding)),
                    ),
                ),
            ),
            layers=(layer,),
        ),
        lambda: workflow_lora_override_resolution_payload(cast(Any, _subclass_copy(resolution))),
        lambda: workflow_lora_override_resolution_payload(
            replace(
                resolution,
                overrides=(cast(Any, _subclass_copy(resolved_override)),),
            )
        ),
        lambda: workflow_lora_override_resolution_payload(
            replace(
                resolution,
                overrides=(
                    replace(
                        resolved_override,
                        changes=(cast(Any, _subclass_copy(resolved_change)),),
                    ),
                ),
            )
        ),
    )

    for hostile_call in hostile_calls:
        with pytest.raises(WorkflowLoraOverrideError):
            hostile_call()


def test_public_slot_type_refusal_does_not_evaluate_hostile_properties() -> None:
    class PropertyTrap:
        @property
        def slot_id(self) -> str:
            raise AssertionError("a non-exact slot property was evaluated")

    with pytest.raises(WorkflowLoraOverrideError) as caught:
        resolve_workflow_lora_override_layers(
            catalog=WorkflowLoraOverrideCatalog(
                _parsed_target(),
                (cast(Any, PropertyTrap()),),
            ),
            layers=(),
        )

    assert caught.value.code == "invalid_workflow_lora_override_catalog"


def test_every_exact_target_identity_component_participates_in_the_digest() -> None:
    baseline = parse_workflow_lora_overrides(_envelope(_target()))
    baseline_digest = workflow_lora_overrides_sha256(baseline)
    mutations: dict[str, object] = {
        "workflow_family_id": "wffamily-other",
        "workflow_definition_id": "workflow-other",
        "workflow_variant_key": "video",
        "workflow_revision_id": "wfrev-other",
        "revision_scope_sha256": _digest(201),
        "api_graph_sha256": _digest(202),
        "dependency_contract_sha256": _digest(203),
        "activation_binding_sha256": _digest(204),
        "activation_witness_sha256": _digest(205),
    }

    for field, value in mutations.items():
        target = _target()
        target[field] = value
        changed = parse_workflow_lora_overrides(_envelope(target))
        assert workflow_lora_overrides_sha256(changed) != baseline_digest, field

    changed_slot = parse_workflow_lora_overrides(_envelope(_target(overrides=[_slot_override(2)])))
    changed_authority = parse_workflow_lora_overrides(
        _envelope(_target(overrides=[_slot_override(authority=_digest(250))]))
    )
    changed_value = parse_workflow_lora_overrides(
        _envelope(_target(overrides=[_slot_override(changes={"model_strength": 0.8})]))
    )
    assert (
        len(
            {
                baseline_digest,
                workflow_lora_overrides_sha256(changed_slot),
                workflow_lora_overrides_sha256(changed_authority),
                workflow_lora_overrides_sha256(changed_value),
            }
        )
        == 4
    )


def test_layer_resolution_is_field_wise_in_closed_precedence_order() -> None:
    witness = _parsed_target()
    slot = _catalog_slot()
    catalog = WorkflowLoraOverrideCatalog(witness, (slot,))
    project = _layer(
        "project",
        _target(overrides=[_slot_override(changes={"model_strength": 0.25, "clip_strength": 0.4})]),
    )
    chat = _layer(
        "chat",
        _target(overrides=[_slot_override(changes={"clip_strength": 0.7})]),
    )
    # Equal to the authored value, but it deliberately masks the lower project
    # value and must retain the explicit turn origin.
    turn = _layer(
        "turn",
        _target(overrides=[_slot_override(changes={"model_strength": 1.0})]),
    )

    resolution = resolve_workflow_lora_override_layers(
        catalog=catalog,
        layers=(turn, project, chat),
    )

    assert resolution.version == 1
    assert resolution.target == witness
    assert resolution.inactive_targets == ()
    assert [
        (change.field, change.value, change.origin) for change in resolution.overrides[0].changes
    ] == [
        ("model_strength", 1.0, "turn"),
        ("clip_strength", 0.7, "chat"),
    ]


def test_all_eight_origins_are_closed_ordered_and_field_wise() -> None:
    witness = _parsed_target()
    catalog = WorkflowLoraOverrideCatalog(witness, (_catalog_slot(),))
    origins = (
        "profile_request",
        "default_preset",
        "project_preset",
        "project",
        "chat_preset",
        "chat",
        "turn_preset",
        "turn",
    )
    assert origins == WORKFLOW_LORA_OVERRIDE_ORIGINS
    layers = tuple(
        _layer(
            origin,
            _target(overrides=[_slot_override(changes={"model_strength": index / 10})]),
        )
        for index, origin in enumerate(origins, start=1)
    )

    resolution = resolve_workflow_lora_override_layers(
        catalog=catalog,
        layers=tuple(reversed(layers)),
    )

    assert resolution.overrides[0].changes[0].value == 0.8
    assert resolution.overrides[0].changes[0].origin == "turn"

    with pytest.raises(WorkflowLoraOverrideError) as invalid:
        resolve_workflow_lora_override_layers(
            catalog=catalog,
            layers=(_layer("profile_load", _target()),),
        )
    assert invalid.value.code == "invalid_workflow_lora_override_layer"

    with pytest.raises(WorkflowLoraOverrideError) as duplicate:
        resolve_workflow_lora_override_layers(
            catalog=catalog,
            layers=(layers[0], layers[0]),
        )
    assert duplicate.value.code == "duplicate_workflow_lora_override_layer"

    hostile_origin = WorkflowLoraOverrideLayer(cast(Any, []), layers[0].overrides)
    with pytest.raises(WorkflowLoraOverrideError) as hostile:
        resolve_workflow_lora_override_layers(catalog=catalog, layers=(hostile_origin,))
    assert hostile.value.code == "invalid_workflow_lora_override_layer"


def test_other_workflow_and_revision_targets_are_retained_as_inactive_without_bleed() -> None:
    witness = _parsed_target()
    catalog = WorkflowLoraOverrideCatalog(witness, (_catalog_slot(),))
    layer = _layer(
        "project",
        _target(definition="workflow-other", revision="wfrev-other", digest_offset=20),
        _target(revision="wfrev-other", digest_offset=30),
    )

    resolution = resolve_workflow_lora_override_layers(catalog=catalog, layers=(layer,))

    assert resolution.overrides == ()
    assert {
        (item.target.workflow_definition_id, item.reason) for item in resolution.inactive_targets
    } == {
        ("workflow-one", "different_revision"),
        ("workflow-other", "different_workflow"),
    }


def test_current_definition_and_revision_with_any_stale_witness_refuses() -> None:
    witness = _parsed_target()
    catalog = WorkflowLoraOverrideCatalog(witness, (_catalog_slot(),))
    stale = _target()
    stale["activation_witness_sha256"] = _digest(999)

    with pytest.raises(WorkflowLoraOverrideError) as caught:
        resolve_workflow_lora_override_layers(
            catalog=catalog,
            layers=(_layer("chat", stale),),
        )

    assert caught.value.code == "stale_workflow_lora_override_target"


@pytest.mark.parametrize(
    ("slot", "override", "code"),
    [
        (
            _catalog_slot(),
            _slot_override(authority=_digest(999)),
            "stale_workflow_lora_override_slot",
        ),
        (
            _catalog_slot(
                editability="required_locked",
                dependency_required=True,
                editable_fields=(),
            ),
            _slot_override(),
            "workflow_lora_override_locked",
        ),
        (
            _catalog_slot(
                editability="detected_read_only",
                dependency_required=None,
                editable_fields=(),
                has_asset=False,
            ),
            _slot_override(),
            "workflow_lora_override_locked",
        ),
        (
            _catalog_slot(editable_fields=("model_strength",)),
            _slot_override(changes={"clip_strength": 0.5}),
            "unsupported_workflow_lora_override_field",
        ),
        (
            _catalog_slot(default_enabled=False, editable_fields=("enabled", "model_strength")),
            _slot_override(changes={"enabled": True}),
            "workflow_lora_override_enable_unsupported",
        ),
    ],
)
def test_resolution_requires_exact_editable_slot_and_loader_authority(
    slot: WorkflowLoraSlot,
    override: dict[str, object],
    code: str,
) -> None:
    witness = _parsed_target()
    catalog = WorkflowLoraOverrideCatalog(witness, (slot,))

    with pytest.raises(WorkflowLoraOverrideError) as caught:
        resolve_workflow_lora_override_layers(
            catalog=catalog,
            layers=(_layer("turn", _target(overrides=[override])),),
        )

    assert caught.value.code == code


def test_unknown_slot_and_duplicate_catalog_slots_refuse() -> None:
    witness = _parsed_target()
    slot = _catalog_slot()

    with pytest.raises(WorkflowLoraOverrideError) as unknown:
        resolve_workflow_lora_override_layers(
            catalog=WorkflowLoraOverrideCatalog(witness, (slot,)),
            layers=(_layer("turn", _target(overrides=[_slot_override(2)])),),
        )
    assert unknown.value.code == "stale_workflow_lora_override_slot"

    with pytest.raises(WorkflowLoraOverrideError) as duplicate:
        resolve_workflow_lora_override_layers(
            catalog=WorkflowLoraOverrideCatalog(witness, (slot, slot)),
            layers=(),
        )
    assert duplicate.value.code == "duplicate_workflow_lora_override_catalog_slot"

    repeated_position = replace(_catalog_slot(2), position=slot.position)
    with pytest.raises(WorkflowLoraOverrideError) as position:
        resolve_workflow_lora_override_layers(
            catalog=WorkflowLoraOverrideCatalog(
                witness,
                (slot, repeated_position),
            ),
            layers=(),
        )
    assert position.value.code == "invalid_workflow_lora_override_catalog"


def test_enabled_can_only_toggle_an_authored_on_optional_audited_slot() -> None:
    witness = _parsed_target()
    slot = _catalog_slot(
        loader_contract="rgthree-power-lora-loader-v1",
        editable_fields=("enabled", "model_strength"),
        clip_strength=1.0,
    )
    catalog = WorkflowLoraOverrideCatalog(witness, (slot,))
    disabled = _layer(
        "project",
        _target(
            overrides=[
                _slot_override(
                    loader_contract="rgthree-power-lora-loader-v1",
                    changes={"enabled": False},
                )
            ]
        ),
    )
    authored_reset = _layer(
        "turn",
        _target(
            overrides=[
                _slot_override(
                    loader_contract="rgthree-power-lora-loader-v1",
                    changes={"enabled": True},
                )
            ]
        ),
    )

    lower = resolve_workflow_lora_override_layers(catalog=catalog, layers=(disabled,))
    assert (lower.overrides[0].changes[0].value, lower.overrides[0].changes[0].origin) == (
        False,
        "project",
    )
    reset = resolve_workflow_lora_override_layers(
        catalog=catalog,
        layers=(authored_reset, disabled),
    )
    assert (reset.overrides[0].changes[0].value, reset.overrides[0].changes[0].origin) == (
        True,
        "turn",
    )


def test_model_only_and_coupled_slots_never_gain_invented_fields() -> None:
    witness = _parsed_target()
    model_only = _catalog_slot(
        editable_fields=("model_strength",),
        clip_strength=None,
    )
    coupled = _catalog_slot(
        editable_fields=("enabled", "model_strength"),
        loader_contract="rgthree-power-lora-loader-v1",
    )

    for slot, field in (
        (model_only, "enabled"),
        (model_only, "clip_strength"),
        (coupled, "clip_strength"),
    ):
        loader = cast(str, slot.loader_contract)
        with pytest.raises(WorkflowLoraOverrideError) as caught:
            resolve_workflow_lora_override_layers(
                catalog=WorkflowLoraOverrideCatalog(witness, (slot,)),
                layers=(
                    _layer(
                        "turn",
                        _target(
                            overrides=[
                                _slot_override(
                                    loader_contract=loader,
                                    changes={field: False if field == "enabled" else 0.5},
                                )
                            ]
                        ),
                    ),
                ),
            )
        assert caught.value.code == "unsupported_workflow_lora_override_field"


def test_two_occurrences_with_the_same_asset_remain_independent_slots() -> None:
    witness = _parsed_target()
    shared_asset = _asset_binding(1)
    first = _catalog_slot(1, asset=shared_asset)
    second = _catalog_slot(2, asset=shared_asset)
    layer = _layer(
        "turn",
        _target(
            overrides=[
                _slot_override(1, changes={"model_strength": 0.4}),
                _slot_override(2, changes={"clip_strength": 0.6}),
            ]
        ),
    )

    resolution = resolve_workflow_lora_override_layers(
        catalog=WorkflowLoraOverrideCatalog(witness, (first, second)),
        layers=(layer,),
    )

    assert [(item.slot_id, item.changes[0].field) for item in resolution.overrides] == [
        (_slot_id(1), "model_strength"),
        (_slot_id(2), "clip_strength"),
    ]


def test_resolved_plan_digest_includes_winning_values_origins_and_exact_target() -> None:
    witness = _parsed_target()
    catalog = WorkflowLoraOverrideCatalog(witness, (_catalog_slot(),))
    project = _layer(
        "project",
        _target(overrides=[_slot_override(changes={"model_strength": 0.5})]),
    )
    turn = _layer(
        "turn",
        _target(overrides=[_slot_override(changes={"model_strength": 1.0})]),
    )
    resolution = resolve_workflow_lora_override_layers(
        catalog=catalog,
        layers=(turn, project),
    )

    payload = workflow_lora_override_resolution_payload(resolution)
    assert payload["target"] == {
        key: value for key, value in _target().items() if key != "overrides"
    }
    changes = cast(list[dict[str, Any]], payload["overrides"])[0]["changes"]
    assert changes == {"model_strength": {"value": 1.0, "origin": "turn"}}
    expected = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert workflow_lora_override_resolution_sha256(resolution) == expected

    project_only = resolve_workflow_lora_override_layers(catalog=catalog, layers=(project,))
    assert workflow_lora_override_resolution_sha256(project_only) != expected
    same_value_different_origin = resolve_workflow_lora_override_layers(
        catalog=catalog,
        layers=(
            _layer(
                "chat",
                _target(overrides=[_slot_override(changes={"model_strength": 1.0})]),
            ),
        ),
    )
    assert workflow_lora_override_resolution_sha256(same_value_different_origin) != expected


def test_inactive_targets_do_not_change_the_effective_resolution_digest() -> None:
    witness = _parsed_target()
    catalog = WorkflowLoraOverrideCatalog(witness, (_catalog_slot(),))
    active = _layer("turn", _target())
    with_inactive = _layer(
        "turn",
        _target(),
        _target(definition="workflow-other", revision="wfrev-other", digest_offset=20),
    )

    active_resolution = resolve_workflow_lora_override_layers(catalog=catalog, layers=(active,))
    inactive_resolution = resolve_workflow_lora_override_layers(
        catalog=catalog,
        layers=(with_inactive,),
    )

    assert inactive_resolution.inactive_targets
    assert workflow_lora_override_resolution_sha256(inactive_resolution) == (
        workflow_lora_override_resolution_sha256(active_resolution)
    )


def test_inactive_evidence_is_validated_even_though_it_is_not_hashed() -> None:
    witness = _parsed_target()
    resolution = resolve_workflow_lora_override_layers(
        catalog=WorkflowLoraOverrideCatalog(witness, (_catalog_slot(),)),
        layers=(
            _layer(
                "turn",
                _target(definition="workflow-other", digest_offset=20),
            ),
        ),
    )
    inactive = resolution.inactive_targets[0]
    malformed = (
        replace(resolution, inactive_targets=cast(Any, [inactive])),
        replace(
            resolution,
            inactive_targets=(cast(Any, _subclass_copy(inactive)),),
        ),
        replace(
            resolution,
            inactive_targets=(
                replace(inactive, origin=cast(Any, type("Text", (str,), {})("turn"))),
            ),
        ),
        replace(
            resolution,
            inactive_targets=(replace(inactive, reason=cast(Any, "different_revision")),),
        ),
        replace(
            resolution,
            inactive_targets=(
                replace(
                    inactive,
                    target=cast(Any, _subclass_copy(inactive.target)),
                ),
            ),
        ),
    )

    for value in malformed:
        with pytest.raises(WorkflowLoraOverrideError):
            workflow_lora_override_resolution_payload(value)


def test_pure_contract_exposes_no_graph_mutation_location_or_fragment() -> None:
    parsed = parse_workflow_lora_overrides(_envelope(_target()))
    serialized = json.dumps(workflow_lora_overrides_payload(parsed), sort_keys=True)

    for private_name in ("node_id", "locator", "path", "api_graph_json", "ui_graph_json"):
        assert private_name not in serialized
    resolution = resolve_workflow_lora_override_layers(
        catalog=WorkflowLoraOverrideCatalog(_parsed_target(), (_catalog_slot(),)),
        layers=(_layer("turn", _target()),),
    )
    assert not hasattr(resolution, "graph")
    assert not hasattr(resolution.overrides[0], "node_id")
    assert "private target" in (resolve_workflow_lora_override_layers.__doc__ or "")
