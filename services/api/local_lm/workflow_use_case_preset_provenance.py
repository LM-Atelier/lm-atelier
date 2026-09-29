"""Retain admitted recipe values without consulting mutable recipe records."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from .workflow_use_case_preset_resolution import PresetScope, ResolvedWorkflowUseCasePreset
from .workflow_use_case_presets_v1 import PROMPT_SETTING_KEYS
from .workflow_use_cases_v1 import WorkflowUseCase

if TYPE_CHECKING:
    from .workflow_use_case_preset_admission import AdmittedWorkflowUseCasePreset

SnapshotId = Annotated[str, StringConstraints(min_length=1, max_length=40)]


class WorkflowUseCasePresetSnapshot(BaseModel):
    """Describe the recipe layer accepted for one exact execution revision."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    version: Literal[1] = 1
    workflow_revision_id: SnapshotId
    use_case: WorkflowUseCase = Field(strict=False)
    mode: Literal["unconfigured", "automatic", "preset"]
    scope: PresetScope | None
    preset_id: SnapshotId | None
    preset_name: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None
    settings_json: dict[str, JsonValue]
    upscale_kind: Literal["model", "resample"] | None

    @field_validator("version", mode="before")
    @classmethod
    def require_integer_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("workflow-use-case-preset-snapshot-invalid")
        return value

    @model_validator(mode="after")
    def require_consistent_receipt(self) -> Self:
        valid = self.workflow_revision_id == self.workflow_revision_id.strip()
        if self.mode == "preset":
            valid = (
                valid
                and self.scope is not None
                and self.preset_id is not None
                and self.preset_id == self.preset_id.strip()
                and self.preset_name is not None
                and bool(self.preset_name.strip())
            )
        else:
            valid = (
                valid
                and self.preset_id is None
                and self.preset_name is None
                and not self.settings_json
                and (
                    (self.mode == "automatic" and self.scope in {"chat", "project"})
                    or (self.mode == "unconfigured" and self.scope is None)
                )
            )
        if (
            not valid
            or PROMPT_SETTING_KEYS.intersection(self.settings_json)
            or (self.upscale_kind is not None and self.use_case != WorkflowUseCase.IMAGE_UPSCALE)
        ):
            raise ValueError("workflow-use-case-preset-snapshot-invalid")
        return self

    def resolved_preset(self) -> ResolvedWorkflowUseCasePreset:
        """Restore detached values for fresh input, setting and revision admission."""
        return ResolvedWorkflowUseCasePreset(
            self.use_case,
            self.mode,
            self.scope,
            self.preset_id,
            self.preset_name,
            deepcopy(self.settings_json),
        )


def read_workflow_use_case_preset(
    value: object, *, workflow_revision_id: str | None
) -> WorkflowUseCasePresetSnapshot | None:
    """Read an optional receipt, refusing malformed or mismatched provenance."""
    if value is None:
        return None
    try:
        snapshot = WorkflowUseCasePresetSnapshot.model_validate(value).model_copy(deep=True)
    except ValidationError:
        raise ValueError("workflow-use-case-preset-snapshot-invalid") from None
    if snapshot.workflow_revision_id != workflow_revision_id:
        raise ValueError("workflow-use-case-preset-snapshot-revision-mismatch")
    return snapshot


def capture_workflow_use_case_preset(
    admission: AdmittedWorkflowUseCasePreset,
) -> WorkflowUseCasePresetSnapshot:
    """Copy the admitted recipe layer separately from final overridden settings."""
    preset = admission.preset
    snapshot = read_workflow_use_case_preset(
        {
            "workflow_revision_id": admission.workflow_revision_id,
            "use_case": preset.use_case,
            "mode": preset.mode,
            "scope": preset.scope,
            "preset_id": preset.preset_id,
            "preset_name": preset.preset_name,
            "settings_json": preset.settings_json,
            "upscale_kind": admission.upscale_kind,
        },
        workflow_revision_id=admission.workflow_revision_id,
    )
    assert snapshot is not None
    return snapshot
