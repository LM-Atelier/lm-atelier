"""Admit every recipe value against one exact workflow revision."""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Literal

from .models import WorkflowRevision
from .schemas import SettingField
from .settings_registry import validate_settings, workflow_settings
from .workflow_use_case_preset_resolution import ResolvedWorkflowUseCasePreset
from .workflow_use_case_presets_v1 import PROMPT_SETTING_KEYS
from .workflow_use_cases_v1 import WorkflowUseCase

SettingsRefusal = Literal[
    "workflow-use-case-preset-invalid",
    "workflow-use-case-preset-mismatch",
    "workflow-use-case-preset-revision-mismatch",
    "workflow-use-case-preset-schema-invalid",
    "workflow-use-case-preset-setting-unsupported",
    "workflow-use-case-preset-setting-unavailable",
    "workflow-use-case-preset-settings-invalid",
    "workflow-use-case-preset-prompt-setting-forbidden",
]


class WorkflowUseCasePresetSettingsError(ValueError):
    def __init__(self, code: SettingsRefusal) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ValidatedWorkflowUseCasePresetSettings:
    workflow_revision_id: str
    preset: ResolvedWorkflowUseCasePreset


def _consistent(preset: ResolvedWorkflowUseCasePreset) -> bool:
    if not isinstance(preset.settings_json, dict):
        return False
    if preset.mode == "preset":
        return (
            preset.scope in {"chat", "project", "workspace"}
            and isinstance(preset.preset_id, str)
            and 0 < len(preset.preset_id) <= 40
            and bool(preset.preset_id.strip())
            and isinstance(preset.preset_name, str)
            and 0 < len(preset.preset_name) <= 200
            and bool(preset.preset_name.strip())
        )
    return (
        not preset.settings_json
        and preset.preset_id is None
        and preset.preset_name is None
        and (
            (preset.mode == "automatic" and preset.scope in {"chat", "project"})
            or (preset.mode == "unconfigured" and preset.scope is None)
        )
    )


def validate_workflow_use_case_preset_settings(
    preset: ResolvedWorkflowUseCasePreset,
    revision: WorkflowRevision,
    *,
    expected_revision_id: str,
    use_case: WorkflowUseCase,
    fields: Iterable[SettingField],
    accepts_added_loras: bool = False,
) -> ValidatedWorkflowUseCasePresetSettings:
    """Validate the recipe layer without choosing a family or adding defaults.

    The caller supplies the selected engine's fields and exact revision. Trust,
    readiness and structural input eligibility remain separate admission checks.
    No saved value is aliased, coerced or silently removed.
    """
    if (
        not isinstance(expected_revision_id, str)
        or not expected_revision_id
        or revision.id != expected_revision_id
    ):
        raise WorkflowUseCasePresetSettingsError("workflow-use-case-preset-revision-mismatch")
    if not isinstance(preset.use_case, WorkflowUseCase) or preset.use_case != use_case:
        raise WorkflowUseCasePresetSettingsError("workflow-use-case-preset-mismatch")
    if not _consistent(preset):
        raise WorkflowUseCasePresetSettingsError("workflow-use-case-preset-invalid")
    if PROMPT_SETTING_KEYS.intersection(preset.settings_json):
        raise WorkflowUseCasePresetSettingsError(
            "workflow-use-case-preset-prompt-setting-forbidden"
        )
    if preset.mode != "preset":
        return ValidatedWorkflowUseCasePresetSettings(
            revision.id, replace(preset, settings_json={})
        )

    engine_fields = list(fields)
    schema = revision.input_schema_json
    if schema is not None and not isinstance(schema, Mapping):
        raise WorkflowUseCasePresetSettingsError("workflow-use-case-preset-schema-invalid")
    try:
        validate_settings({}, engine_fields)
        definitions = workflow_settings(
            engine_fields, schema, accepts_added_loras=accepts_added_loras
        )
        validate_settings({}, definitions)
    except (ValueError, TypeError, OverflowError):
        raise WorkflowUseCasePresetSettingsError(
            "workflow-use-case-preset-schema-invalid"
        ) from None
    by_key = {field.key: field for field in definitions}
    properties = schema.get("properties", {}) if schema else {}
    for key in preset.settings_json:
        declaration = properties.get(key) if isinstance(properties, Mapping) else None
        if isinstance(declaration, Mapping) and declaration.get("readOnly") is True:
            raise WorkflowUseCasePresetSettingsError("workflow-use-case-preset-setting-unavailable")
        field = by_key.get(key)
        if field is None:
            raise WorkflowUseCasePresetSettingsError("workflow-use-case-preset-setting-unsupported")
        if field.scope == "load" or not field.available:
            raise WorkflowUseCasePresetSettingsError("workflow-use-case-preset-setting-unavailable")
    try:
        validated = validate_settings(preset.settings_json, definitions)
    except (ValueError, TypeError, OverflowError):
        raise WorkflowUseCasePresetSettingsError(
            "workflow-use-case-preset-settings-invalid"
        ) from None
    return ValidatedWorkflowUseCasePresetSettings(
        revision.id, replace(preset, settings_json=deepcopy(validated))
    )
