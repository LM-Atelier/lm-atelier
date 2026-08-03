from __future__ import annotations

from copy import deepcopy

import pytest

import local_lm.workflow_dependencies as dependency_module
from local_lm.workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyError,
    WorkflowDependencyRequirement,
    WorkflowDependencySlotContract,
    legacy_workflow_dependency_view,
    parse_workflow_dependency_contract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_sha256,
)


def _contract() -> dict[str, object]:
    return {
        "version": 1,
        "slots": [
            {
                "name": "primary",
                "resource_kind": "model_profile",
                "required": True,
                "satisfaction": "any_of",
                "requirements": [
                    {
                        "key": "balanced",
                        "constraints": {
                            "role": "image",
                            "engine": "comfyui",
                            "model": {
                                "components": [
                                    {
                                        "kind": "diffusion_model",
                                        "runtime_reference": "models/quality.safetensors",
                                        "sha256": "a" * 64,
                                        "target_folder": "diffusion_models",
                                    },
                                    {
                                        "kind": "vae",
                                        "runtime_reference": "vae/quality.safetensors",
                                        "sha256": "b" * 64,
                                        "target_folder": "vae",
                                    },
                                ]
                            },
                        },
                    },
                    {
                        "key": "fast",
                        "constraints": {"role": "image", "engine": "comfyui"},
                    },
                ],
            },
            {
                "name": "encoders",
                "resource_kind": "model_install",
                "required": True,
                "satisfaction": "all_of",
                "requirements": [
                    {"key": "clip", "constraints": {"role": "image"}},
                    {"key": "t5", "constraints": {"role": "image"}},
                ],
            },
            {
                "name": "style",
                "resource_kind": "model_asset",
                "required": False,
                "satisfaction": "any_of",
                "requirements": [{"key": "optional_lora", "constraints": {"asset_kind": "lora"}}],
            },
        ],
    }


def _assert_error(code: str, value: object) -> None:
    with pytest.raises(WorkflowDependencyError) as raised:
        parse_workflow_dependency_contract(value)
    assert raised.value.code == code


def test_parses_required_optional_all_of_and_any_of_slots() -> None:
    contract = parse_workflow_dependency_contract(_contract())

    assert [slot.name for slot in contract.slots] == ["encoders", "primary", "style"]
    assert contract.slots[0].satisfaction == "all_of"
    assert contract.slots[1].required is True
    assert [item.key for item in contract.slots[1].requirements] == ["balanced", "fast"]
    assert contract.slots[2].required is False
    assert len(workflow_dependency_contract_sha256(contract)) == 64
    assert len(workflow_dependency_slot_sha256(contract.slots[0])) == 64


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"version": "1"}, "invalid_workflow_dependencies"),
        ({"slots.0.required": 1}, "invalid_workflow_dependencies"),
        ({"slots.0.satisfaction": "one_of"}, "invalid_workflow_dependencies"),
        ({"slots.0.requirements": []}, "invalid_workflow_dependencies"),
        ({"slots.0.resource_kind": "checkpoint"}, "invalid_workflow_dependencies"),
        (
            {"slots.0.requirements.0.constraints.model_install_id": "model_local"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.local_path": "C:/models/private"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.password": "do-not-store"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.authToken": "do-not-store"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.hf_token": "do-not-store"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.tokens": ["do-not-store"]},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.api_keys": ["do-not-store"]},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.location": "C:relative-model"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.location": f"{chr(92)}rooted-model"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.location": "../model"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.location": "file:///tmp/model"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.location": "%USERPROFILE%/models"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.location": "$HOME/models"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.location": "${HOME}/models"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.location": "$env:USERPROFILE/models"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.location": "${env:USERPROFILE}/models"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.location": "~someone/models"},
            "nonportable_dependency_data",
        ),
        (
            {
                "slots.0.requirements.0.constraints.source_url": "https://alice:secret@example.com/repo"
            },
            "nonportable_dependency_data",
        ),
        (
            {
                "slots.0.requirements.0.constraints.source_url": "https://example.com/repo?access_token=secret"
            },
            "nonportable_dependency_data",
        ),
        (
            {
                "slots.0.requirements.0.constraints.source_url": "https://example.com/repo#access_token=secret"
            },
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.source_url": "callback#access_token=secret"},
            "nonportable_dependency_data",
        ),
        (
            {"slots.0.requirements.0.constraints.kind": "model_asset"},
            "invalid_workflow_dependencies",
        ),
        (
            {"slots.0.requirements.0.constraints.kind": None},
            "invalid_workflow_dependencies",
        ),
        (
            {"slots.0.requirements.0.constraints.label": f"invalid{chr(127)}text"},
            "invalid_portable_dependency_data",
        ),
    ],
)
def test_invalid_or_nonportable_contracts_fail_closed(change: dict[str, object], code: str) -> None:
    value = _contract()
    for dotted, replacement in change.items():
        target: object = value
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]  # type: ignore[index]
        if isinstance(target, list):
            target[int(parts[-1])] = replacement
        else:
            target[parts[-1]] = replacement  # type: ignore[index]

    _assert_error(code, value)


def test_rejects_duplicate_slots_and_requirements() -> None:
    duplicated_slot = _contract()
    slots = duplicated_slot["slots"]
    assert isinstance(slots, list)
    slots.append(deepcopy(slots[0]))
    _assert_error("duplicate_dependency_slot", duplicated_slot)

    duplicated_requirement = _contract()
    first_slot = duplicated_requirement["slots"]
    assert isinstance(first_slot, list) and isinstance(first_slot[0], dict)
    requirements = first_slot[0]["requirements"]
    assert isinstance(requirements, list)
    requirements.append(deepcopy(requirements[0]))
    _assert_error("duplicate_dependency_requirement", duplicated_requirement)


def test_rejects_oversized_aggregate_dependency_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dependency_module, "MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS", 2)

    _assert_error("too_many_workflow_dependencies", _contract())


def test_dependency_digest_is_order_independent_but_semantic() -> None:
    original = _contract()
    reversed_value = deepcopy(original)
    slots = reversed_value["slots"]
    assert isinstance(slots, list)
    slots.reverse()
    for slot in slots:
        assert isinstance(slot, dict)
        requirements = slot["requirements"]
        assert isinstance(requirements, list)
        requirements.reverse()
    primary = next(slot for slot in slots if isinstance(slot, dict) and slot["name"] == "primary")
    requirements = primary["requirements"]
    assert isinstance(requirements, list) and isinstance(requirements[0], dict)
    balanced = next(
        requirement
        for requirement in requirements
        if isinstance(requirement, dict) and requirement["key"] == "balanced"
    )
    constraints = balanced["constraints"]
    assert isinstance(constraints, dict)
    balanced["constraints"] = dict(reversed(list(constraints.items())))
    model = balanced["constraints"]["model"]
    assert isinstance(model, dict)
    components = model["components"]
    assert isinstance(components, list)
    components.reverse()

    first = parse_workflow_dependency_contract(original)
    second = parse_workflow_dependency_contract(reversed_value)

    assert first == second
    assert workflow_dependency_contract_sha256(first) == workflow_dependency_contract_sha256(second)

    changed = deepcopy(original)
    changed_slots = changed["slots"]
    assert isinstance(changed_slots, list) and isinstance(changed_slots[0], dict)
    changed_slots[0]["required"] = False
    assert workflow_dependency_contract_sha256(
        parse_workflow_dependency_contract(changed)
    ) != workflow_dependency_contract_sha256(first)


def test_direct_contract_construction_is_revalidated_before_hashing() -> None:
    direct = WorkflowDependencyContract(
        version=1,
        slots=(
            WorkflowDependencySlotContract(
                name="primary",
                resource_kind="runtime",
                required=True,
                satisfaction="any_of",
                requirements=(
                    WorkflowDependencyRequirement("second", {"engine": "comfyui"}),
                    WorkflowDependencyRequirement("first", {"engine": "comfyui"}),
                ),
            ),
        ),
    )
    canonical = parse_workflow_dependency_contract(
        {
            "version": 1,
            "slots": [
                {
                    "name": "primary",
                    "resource_kind": "runtime",
                    "required": True,
                    "satisfaction": "any_of",
                    "requirements": [
                        {"key": "first", "constraints": {"engine": "comfyui"}},
                        {"key": "second", "constraints": {"engine": "comfyui"}},
                    ],
                }
            ],
        }
    )

    assert workflow_dependency_contract_sha256(direct) == workflow_dependency_contract_sha256(
        canonical
    )


def test_signed_float_zero_has_one_canonical_contract_digest() -> None:
    negative_zero = _contract()
    slots = negative_zero["slots"]
    assert isinstance(slots, list) and isinstance(slots[0], dict)
    requirements = slots[0]["requirements"]
    assert isinstance(requirements, list) and isinstance(requirements[0], dict)
    constraints = requirements[0]["constraints"]
    assert isinstance(constraints, dict)
    constraints["weight"] = -0.0
    positive_zero = deepcopy(negative_zero)
    positive_slots = positive_zero["slots"]
    assert isinstance(positive_slots, list) and isinstance(positive_slots[0], dict)
    positive_requirements = positive_slots[0]["requirements"]
    assert isinstance(positive_requirements, list) and isinstance(positive_requirements[0], dict)
    positive_constraints = positive_requirements[0]["constraints"]
    assert isinstance(positive_constraints, dict)
    positive_constraints["weight"] = 0.0

    assert workflow_dependency_contract_sha256(
        parse_workflow_dependency_contract(negative_zero)
    ) == workflow_dependency_contract_sha256(parse_workflow_dependency_contract(positive_zero))


def test_legacy_view_does_not_promote_flat_components_into_slots() -> None:
    view = legacy_workflow_dependency_view(
        {
            "model_install_ids": ["model_b", "model_a", "model_a"],
            "models": ["portable-name", {"id": "legacy-shape"}],
            "custom_nodes": [{"source_url": "https://github.com/example/node"}],
            "model_components": [
                {"target_folder": "unet", "sha256": "A" * 64},
                {"target_folder": "unet", "sha256": "A" * 64},
            ],
            "extensions": {"lora": {}},
        }
    )

    assert view.model_install_ids == ("model_a", "model_b")
    assert view.model_components == ({"target_folder": "unet", "sha256": "a" * 64},)
    assert view.model_references == ("portable-name", {"id": "legacy-shape"})
    assert view.unsupported_keys == ("extensions",)
