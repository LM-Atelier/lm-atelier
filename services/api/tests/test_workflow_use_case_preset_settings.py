"""Refuse incompatible recipe values without silently dropping a setting."""

from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

import pytest

from local_lm.models import WorkflowRevision
from local_lm.schemas import SettingField
from local_lm.settings_registry import builtin_settings_for_role
from local_lm.workflow_use_case_preset_resolution import ResolvedWorkflowUseCasePreset
from local_lm.workflow_use_cases_v1 import WorkflowUseCase


def _validator() -> ModuleType:
    name = "local_lm.workflow_use_case_preset_settings"
    assert find_spec(name) is not None, "Strict recipe settings admission is absent."
    return import_module(name)


def _preset(**values: object) -> ResolvedWorkflowUseCasePreset:
    return ResolvedWorkflowUseCasePreset(
        WorkflowUseCase.IMAGE_GENERATION, "preset", "workspace", "preset", "Example", dict(values)
    )


def _revision(schema: dict | None = None) -> WorkflowRevision:
    return WorkflowRevision(id="revision", workflow_id="workflow", input_schema_json=schema or {})


def _admit(
    preset: ResolvedWorkflowUseCasePreset,
    revision: WorkflowRevision | None = None,
    fields: list[SettingField] | None = None,
):
    return _validator().validate_workflow_use_case_preset_settings(
        preset,
        revision or _revision(),
        expected_revision_id="revision",
        use_case=WorkflowUseCase.IMAGE_GENERATION,
        fields=fields if fields is not None else builtin_settings_for_role("image"),
    )


@pytest.mark.parametrize("key", ["prompt", "negative_prompt"])
def test_stored_prompt_settings_are_refused_with_a_fixed_reason(key: str) -> None:
    module = _validator()
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(_preset(**{key: "neutral fixture"}))
    assert caught.value.code == "workflow-use-case-preset-prompt-setting-forbidden"
    assert str(caught.value) == caught.value.code


def test_recipe_settings_are_bound_to_an_exact_revision_and_detached() -> None:
    schema = {"properties": {"options": {"type": "object", "default": {"sizes": [256]}}}}
    recipe = _preset(seed=9, options={"sizes": [512]})
    result = _admit(recipe, _revision(schema))
    assert result.workflow_revision_id == "revision"
    assert result.preset.preset_id == "preset" and result.preset.scope == "workspace"
    assert result.preset.settings_json == {"seed": 9, "options": {"sizes": [512]}}
    recipe.settings_json["options"]["sizes"].append(1024)
    schema["properties"]["options"]["default"]["sizes"].append(2048)
    assert result.preset.settings_json["options"] == {"sizes": [512]}


@pytest.mark.parametrize("values", [{"unknown_control": 3}, {"seed": 4, "unknown_control": 3}])
def test_unknown_recipe_settings_refuse_the_entire_layer(values: dict) -> None:
    module = _validator()
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(_preset(**values))
    assert caught.value.code == "workflow-use-case-preset-setting-unsupported"
    assert str(caught.value) == caught.value.code
    assert "unknown_control" not in str(caught.value)


@pytest.mark.parametrize("value", [True, "9", 10**400, float("nan"), 1.5])
def test_invalid_recipe_values_refuse_instead_of_coercing(value: object) -> None:
    module = _validator()
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(_preset(seed=value))
    assert caught.value.code == "workflow-use-case-preset-settings-invalid"


@pytest.mark.parametrize("scope,available", [("load", True), ("workflow", False)])
def test_unavailable_or_load_settings_cannot_enter_a_recipe(scope: str, available: bool) -> None:
    field = SettingField.model_validate(
        dict(
            key="control",
            label="Control",
            type="integer",
            default=1,
            scope=scope,
            available=available,
        )
    )
    module = _validator()
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(_preset(control=2), fields=[field])
    assert caught.value.code == "workflow-use-case-preset-setting-unavailable"


def test_read_only_custom_controls_cannot_be_overridden() -> None:
    schema = {"properties": {"control": {"type": "integer", "default": 2, "readOnly": True}}}
    module = _validator()
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(_preset(control=3), _revision(schema))
    assert caught.value.code == "workflow-use-case-preset-setting-unavailable"


def test_graph_owned_controls_cannot_receive_recipe_overrides() -> None:
    schema = {
        "properties": {},
        "x-lm-atelier-graph-settings": {"version": 1, "bindings": []},
    }
    module = _validator()
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(_preset(seed=3), _revision(schema))
    assert caught.value.code == "workflow-use-case-preset-setting-unavailable"


@pytest.mark.parametrize("value", [12, 17, 24])
def test_workflow_bounds_and_multiples_are_enforced(value: int) -> None:
    schema = {
        "properties": {
            "control": {
                "type": "integer",
                "default": 16,
                "minimum": 16,
                "maximum": 20,
                "multipleOf": 4,
            }
        }
    }
    module = _validator()
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(_preset(control=value), _revision(schema))
    assert caught.value.code == "workflow-use-case-preset-settings-invalid"
    assert _admit(_preset(control=20), _revision(schema)).preset.settings_json == {"control": 20}


def test_duplicate_engine_fields_and_invalid_workflow_schema_refuse() -> None:
    module = _validator()
    fields = builtin_settings_for_role("image")
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(_preset(seed=9), fields=[*fields, fields[0]])
    assert caught.value.code == "workflow-use-case-preset-schema-invalid"
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(
            _preset(seed=9), _revision({"properties": {"seed": {"type": "object", "default": {}}}})
        )
    assert caught.value.code == "workflow-use-case-preset-schema-invalid"


def test_wrong_revision_and_use_case_are_refused() -> None:
    module = _validator()
    revision = _revision()
    revision.id = "different"
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(_preset(), revision)
    assert caught.value.code == "workflow-use-case-preset-revision-mismatch"
    wrong = ResolvedWorkflowUseCasePreset(
        WorkflowUseCase.IMAGE_EDIT, "preset", "chat", "preset", "Example", {}
    )
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(wrong)
    assert caught.value.code == "workflow-use-case-preset-mismatch"


@pytest.mark.parametrize(
    "mode,scope", [("automatic", "chat"), ("automatic", "project"), ("unconfigured", None)]
)
def test_automatic_and_unconfigured_contribute_no_settings(mode: str, scope: str | None) -> None:
    recipe = ResolvedWorkflowUseCasePreset(WorkflowUseCase.IMAGE_GENERATION, mode, scope)
    result = _admit(recipe)
    assert result.preset.mode == mode and result.preset.scope == scope
    assert result.preset.settings_json == {} and result.preset.preset_id is None


@pytest.mark.parametrize(
    "recipe",
    [
        ResolvedWorkflowUseCasePreset(
            WorkflowUseCase.IMAGE_GENERATION, "automatic", "chat", settings_json={"seed": 4}
        ),
        ResolvedWorkflowUseCasePreset(WorkflowUseCase.IMAGE_GENERATION, "unconfigured", "chat"),
        ResolvedWorkflowUseCasePreset(WorkflowUseCase.IMAGE_GENERATION, "automatic", "workspace"),
        ResolvedWorkflowUseCasePreset(WorkflowUseCase.IMAGE_GENERATION, "preset", "workspace"),
    ],
)
def test_inconsistent_resolution_snapshots_are_refused(
    recipe: ResolvedWorkflowUseCasePreset,
) -> None:
    module = _validator()
    with pytest.raises(module.WorkflowUseCasePresetSettingsError) as caught:
        _admit(recipe)
    assert caught.value.code == "workflow-use-case-preset-invalid"
