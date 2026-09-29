"""Define use-case recipes and choices separately from role-based settings."""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    StringConstraints,
    model_validator,
)

from .workflow_use_cases_v1 import WorkflowUseCase

PROMPT_SETTING_KEYS = frozenset({"prompt", "negative_prompt"})

PresetName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
PresetId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]


class _PresetContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class WorkflowUseCasePresetCreate(_PresetContract):
    """Describe a recipe whose settings still need exact-revision admission."""

    name: PresetName
    use_case: WorkflowUseCase = Field(strict=False)
    settings_json: dict[str, JsonValue] = Field(default_factory=dict)
    enabled: bool = True
    is_default: bool = False

    @model_validator(mode="after")
    def require_enabled_default(self) -> Self:
        if self.is_default and not self.enabled:
            raise ValueError("workflow-use-case-default-disabled")
        return self

    @model_validator(mode="after")
    def require_settings_without_prompts(self) -> Self:
        if PROMPT_SETTING_KEYS.intersection(self.settings_json):
            raise ValueError("workflow-use-case-preset-prompt-setting-forbidden")
        return self


class InheritedWorkflowUseCaseChoice(_PresetContract):
    mode: Literal["inherit"]


class AutomaticWorkflowUseCaseChoice(_PresetContract):
    mode: Literal["automatic"]


class ExplicitWorkflowUseCaseChoice(_PresetContract):
    mode: Literal["preset"]
    preset_id: PresetId


class WorkflowUseCaseChoice(
    RootModel[
        Annotated[
            InheritedWorkflowUseCaseChoice
            | AutomaticWorkflowUseCaseChoice
            | ExplicitWorkflowUseCaseChoice,
            Field(discriminator="mode"),
        ]
    ]
):
    """Keep inherited absence distinct from Automatic and a selected recipe."""
