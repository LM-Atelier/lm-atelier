"""Preset contracts distinguish inherited, automatic and explicit choices."""

import importlib
import importlib.util
from types import ModuleType

import pytest
from pydantic import ValidationError


def _contract() -> ModuleType:
    name = "local_lm.workflow_use_case_presets_v1"
    assert importlib.util.find_spec(name) is not None, "Use-case preset contracts are absent."
    return importlib.import_module(name)


@pytest.mark.parametrize("key", ["prompt", "negative_prompt"])
def test_prompt_keys_cannot_enter_execution_recipes(key: str) -> None:
    with pytest.raises(ValidationError, match="workflow-use-case-preset-prompt-setting-forbidden"):
        _contract().WorkflowUseCasePresetCreate.model_validate(
            {
                "name": "Example",
                "use_case": "image_generation",
                "settings_json": {key: "neutral fixture"},
            }
        )


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "inherit"},
        {"mode": "automatic"},
        {"mode": "preset", "preset_id": "wfucpreset_example"},
    ],
)
def test_choice_modes_roundtrip_without_inventing_a_target(value: dict[str, object]) -> None:
    contract = _contract()
    result = contract.WorkflowUseCaseChoice.model_validate(value)
    assert result.model_dump(mode="json") == value


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"mode": "preset"},
        {"mode": "preset", "preset_id": None},
        {"mode": "preset", "preset_id": ""},
        {"mode": "preset", "preset_id": " "},
        {"mode": "preset", "preset_id": 123},
        {"mode": "preset", "preset_id": "x" * 41},
        {"mode": "inherit", "preset_id": "wfucpreset_example"},
        {"mode": "automatic", "preset_id": "wfucpreset_example"},
        {"mode": "unknown"},
        {"mode": "inherit", "settings_json": {}},
    ],
)
def test_impossible_selection_shapes_are_refused(value: dict[str, object]) -> None:
    contract = _contract()
    with pytest.raises(ValidationError):
        contract.WorkflowUseCaseChoice.model_validate(value)


@pytest.mark.parametrize(
    "use_case",
    [
        "chat",
        "image_generation",
        "image_edit",
        "image_inpaint",
        "image_outpaint",
        "image_upscale",
        "video_generation",
        "video_animate",
    ],
)
def test_preset_keys_use_the_classifier_vocabulary(use_case: str) -> None:
    contract = _contract()
    value = contract.WorkflowUseCasePresetCreate.model_validate(
        {"name": "  Small output  ", "use_case": use_case, "settings_json": {"seed": 7}}
    )
    assert value.name == "Small output"
    assert value.model_dump(mode="json")["use_case"] == use_case
    assert value.enabled and not value.is_default
    assert value.settings_json == {"seed": 7}


@pytest.mark.parametrize(
    "change",
    [
        {"name": ""},
        {"name": " "},
        {"name": "x" * 201},
        {"name": 123},
        {"use_case": "unknown"},
        {"enabled": "yes"},
        {"enabled": 1},
        {"is_default": "yes"},
        {"builtin": True},
        {"workflow_family_id": "family"},
        {"settings_json": []},
        {"settings_json": None},
        {"enabled": False, "is_default": True},
    ],
)
def test_malformed_recipe_contracts_are_refused(change: dict[str, object]) -> None:
    contract = _contract()
    with pytest.raises(ValidationError):
        contract.WorkflowUseCasePresetCreate.model_validate(
            {"name": "Small output", "use_case": "image_generation", **change}
        )


def test_new_recipes_do_not_share_a_mutable_settings_default() -> None:
    contract = _contract()
    first = contract.WorkflowUseCasePresetCreate(name="First", use_case="image_generation")
    second = contract.WorkflowUseCasePresetCreate(name="Second", use_case="image_generation")
    first.settings_json["seed"] = 7
    assert second.settings_json == {}
