"""Compare two generation choices against one frozen request.

A comparison holds everything but the choices fixed: one prompt, one negative
prompt, one output shape, one output per choice and one seed policy. Each
choice is an exact model profile and workflow revision with its own settings.
Both are checked together before anything is accepted, so one choice that
cannot run means the comparison accepts no work at all.
"""

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_serializer,
    model_validator,
)

from .output_geometry import MAX_DIMENSION, PresetId

CONTRACT_VERSION: Literal[1] = 1
# The scope of the one hidden chat a started comparison runs in.
EXPERIMENT_CHAT_SCOPE = "experiment"
# The largest seed every image workflow accepts; -1 means "choose one" to them,
# and a comparison never leaves that choice to the run.
MAX_SEED = 2_147_483_647
MAX_ARM_SETTINGS = 64


class SeedPolicyKind(StrEnum):
    """How the seed each choice runs with is chosen.

    ``fixed_numeric`` gives both choices one number and is offered only when
    both run one model family, because only there does a number describe the
    same starting noise. ``same_recorded_number`` gives both choices one number
    without that claim. ``independent_deterministic`` gives each choice its own
    number derived from one base, and ``random_per_trial`` draws one per choice.
    """

    FIXED_NUMERIC = "fixed_numeric"
    SAME_RECORDED_NUMBER = "same_recorded_number"
    INDEPENDENT_DETERMINISTIC = "independent_deterministic"
    RANDOM_PER_TRIAL = "random_per_trial"


class GenerationExperimentRefusalCode(StrEnum):
    ARM_PROFILE_UNAVAILABLE = "arm-profile-unavailable"
    ARM_WORKFLOW_UNAVAILABLE = "arm-workflow-unavailable"
    ARM_OPERATION_MISMATCH = "arm-operation-mismatch"
    ARM_WORKFLOW_UNTRUSTED = "arm-workflow-untrusted"
    ARM_ACTIVATION_NOT_READY = "arm-activation-not-ready"
    ARM_PACKAGE_MISSING = "arm-package-missing"
    ARM_MODEL_MISMATCH = "arm-model-mismatch"
    ARM_ENGINE_UNAVAILABLE = "arm-engine-unavailable"
    ARM_SETTING_UNSUPPORTED = "arm-setting-unsupported"
    ARM_SETTING_INVALID = "arm-setting-invalid"
    COMMON_INPUT_OVERRIDDEN = "common-input-overridden"
    ARM_INPUT_UNSUPPORTED = "arm-input-unsupported"
    ARM_SEED_UNSUPPORTED = "arm-seed-unsupported"
    ARM_GEOMETRY_UNREACHABLE = "arm-geometry-unreachable"
    ARM_LORA_REFUSED = "arm-lora-refused"
    SEED_FAMILY_UNPROVEN = "seed-family-unproven"
    ARMS_IDENTICAL = "arms-identical"
    EXPERIMENT_TOO_LARGE = "experiment-too-large"
    ARM_CHANGED = "arm-changed"


# One fixed sentence per refusal. A refusal never repeats what was asked for,
# so a prompt cannot reach a response, a log or an error report through one.
REFUSAL_MESSAGES: dict[GenerationExperimentRefusalCode, str] = {
    GenerationExperimentRefusalCode.ARM_PROFILE_UNAVAILABLE: (
        "This model is not available for pictures. Choose an installed image model."
    ),
    GenerationExperimentRefusalCode.ARM_WORKFLOW_UNAVAILABLE: (
        "This workflow cannot run on the active media engine. Choose another workflow."
    ),
    GenerationExperimentRefusalCode.ARM_OPERATION_MISMATCH: (
        "This workflow does not make a picture from words. Choose a text-to-image workflow."
    ),
    GenerationExperimentRefusalCode.ARM_WORKFLOW_UNTRUSTED: (
        "This workflow has not been reviewed. Review it before comparing it."
    ),
    GenerationExperimentRefusalCode.ARM_ACTIVATION_NOT_READY: (
        "This workflow's dependencies are not ready yet."
    ),
    GenerationExperimentRefusalCode.ARM_PACKAGE_MISSING: (
        "This workflow needs a package that is not installed."
    ),
    GenerationExperimentRefusalCode.ARM_MODEL_MISMATCH: (
        "This workflow is bound to a different model. Choose the model it runs."
    ),
    GenerationExperimentRefusalCode.ARM_ENGINE_UNAVAILABLE: (
        "The engine this model runs on is not configured."
    ),
    GenerationExperimentRefusalCode.ARM_SETTING_UNSUPPORTED: ("This workflow has no such setting."),
    GenerationExperimentRefusalCode.ARM_SETTING_INVALID: (
        "This setting's value is outside what the workflow accepts."
    ),
    GenerationExperimentRefusalCode.COMMON_INPUT_OVERRIDDEN: (
        "The negative prompt, seed, size and number of pictures are shared by both choices "
        "and cannot be set for one."
    ),
    GenerationExperimentRefusalCode.ARM_INPUT_UNSUPPORTED: (
        "A comparison makes new pictures from words only; selections, relighting and "
        "extending are not part of it."
    ),
    GenerationExperimentRefusalCode.ARM_SEED_UNSUPPORTED: (
        "This workflow does not accept a seed, so its result could not be reproduced."
    ),
    GenerationExperimentRefusalCode.ARM_GEOMETRY_UNREACHABLE: (
        "This workflow cannot be shown to make the chosen shape. Choose an exact size instead."
    ),
    GenerationExperimentRefusalCode.ARM_LORA_REFUSED: (
        "A LoRA chosen here cannot run with this workflow."
    ),
    GenerationExperimentRefusalCode.SEED_FAMILY_UNPROVEN: (
        "One seed means the same starting point only within one model family, and these "
        "choices are not shown to share one. Use the same recorded number instead."
    ),
    GenerationExperimentRefusalCode.ARMS_IDENTICAL: (
        "Both choices are the same and would run with the same seed. Change one of them."
    ),
    GenerationExperimentRefusalCode.EXPERIMENT_TOO_LARGE: (
        "These two pictures together are larger than one request may be."
    ),
    GenerationExperimentRefusalCode.ARM_CHANGED: (
        "This choice would now run differently from when the comparison was accepted. "
        "Check it again."
    ),
}

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PresetGeometry(_Contract):
    """A shape, worked out per choice from what its workflow can be shown to make."""

    mode: Literal["preset"]
    preset_id: PresetId


class SizeGeometry(_Contract):
    """One exact size, the same for both choices."""

    mode: Literal["size"]
    width: int = Field(ge=1, le=MAX_DIMENSION)
    height: int = Field(ge=1, le=MAX_DIMENSION)


Geometry = Annotated[PresetGeometry | SizeGeometry, Field(discriminator="mode")]


class SeedPolicy(_Contract):
    kind: SeedPolicyKind = Field(strict=False)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)

    @model_validator(mode="after")
    def require_a_fixed_number(self) -> Self:
        if self.kind == SeedPolicyKind.FIXED_NUMERIC and self.seed is None:
            raise ValueError("A fixed seed needs its number.")
        if self.kind == SeedPolicyKind.RANDOM_PER_TRIAL and self.seed is not None:
            raise ValueError("A random seed per picture takes no number.")
        return self


class GenerationExperimentPreference(StrEnum):
    """What the person said of the two pictures.

    Their own judgement, kept as said: nothing ranks a choice by it.
    """

    PREFERRED = "preferred"
    TIED = "tied"
    UNSUITABLE = "unsuitable"


class GenerationExperimentEvaluationMode(StrEnum):
    """Whether the pictures were shown with their choices named when it was said."""

    UNBLINDED = "unblinded"
    BLIND = "blind"


class ExperimentArmRequest(_Contract):
    """One choice: an exact model, an exact workflow revision and its own settings."""

    label: Label
    profile_id: Identifier
    workflow_revision_id: Identifier
    settings: dict[str, JsonValue] = Field(default_factory=dict, max_length=MAX_ARM_SETTINGS)


class GenerationExperimentRequest(_Contract):
    name: Name
    operation: Literal["text_to_image"]
    prompt: str = Field(min_length=1, max_length=200_000)
    negative_prompt: str = Field(default="", max_length=20_000)
    geometry: Geometry
    seed_policy: SeedPolicy
    arms: list[ExperimentArmRequest] = Field(min_length=2, max_length=2)
    # Blind hides which picture each choice made until the person says which
    # they prefer, so it is chosen before any picture exists.
    evaluation_mode: GenerationExperimentEvaluationMode = Field(
        default=GenerationExperimentEvaluationMode.UNBLINDED, strict=False
    )

    @model_validator(mode="after")
    def require_distinct_labels(self) -> Self:
        labels = {arm.label.casefold() for arm in self.arms}
        if len(labels) != len(self.arms):
            raise ValueError("Give each choice its own label.")
        return self


class RefusalAlternativeOut(BaseModel):
    """What would be accepted instead, where one exact alternative exists."""

    seed_policy: SeedPolicyKind | None = None
    profile_id: str | None = None


class ExperimentRefusalOut(BaseModel):
    code: GenerationExperimentRefusalCode
    arm_ordinal: int | None = None
    setting: str | None = None
    alternative: RefusalAlternativeOut | None = None
    message: str


class ResourceEvidenceOut(BaseModel):
    """One resource figure with where it came from.

    Only the admission formula's estimate is known before anything runs. Memory
    and time are not estimated at all, so they are absent rather than guessed.
    """

    resource: Literal["work_units", "output_bytes"]
    value: int
    unit: Literal["work_units", "bytes"]
    kind: Literal["measured", "estimated"]
    source: Literal["admission_formula"]
    confidence: Literal["heuristic"]


class ArmPreflightOut(BaseModel):
    ordinal: int
    label: str
    outcome: Literal["compatible", "refused"]
    profile_id: str
    profile_name: str | None = None
    workflow_revision_id: str
    workflow_version: int | None = None
    workflow_activation_id: str | None = None
    model_family: str | None = None
    width: int | None = None
    height: int | None = None
    effective_settings: dict[str, JsonValue] = Field(default_factory=dict)
    # Appended to the prompt for this choice's model and LoRAs, as a turn does.
    trigger_words_applied: list[str] = Field(default_factory=list)
    snapshot_sha256: str | None = None


class GenerationExperimentPreflightOut(BaseModel):
    outcome: Literal["compatible", "refused"]
    preflight_sha256: str | None = None
    seed_equivalence: Literal["same_family", "none"]
    confirmation_required: bool
    estimate: list[ResourceEvidenceOut] = Field(default_factory=list)
    arms: list[ArmPreflightOut]
    refusals: list[ExperimentRefusalOut] = Field(default_factory=list)


def canonical_sha256(value: Any) -> str:
    """One digest for one value, however its keys happen to be ordered."""

    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class GenerationExperimentState(StrEnum):
    READY = "ready"
    STARTED = "started"


class GenerationExperimentTrialState(StrEnum):
    PLANNED = "planned"
    STARTED = "started"


# Where a started picture's work stands, read from its step when the comparison
# is read; "removed" when its work was deleted after it started.
TrialWorkStatus = Literal[
    "queued",
    "running",
    "paused",
    "complete",
    "failed",
    "cancelled",
    "interrupted",
    "blocked",
    "removed",
]


Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class GenerationExperimentCreate(GenerationExperimentRequest):
    """Accept a comparison the preflight answered for, exactly as it answered.

    The digest is the one the preflight returned. If either choice would now
    run differently, it no longer matches and nothing is accepted.
    """

    idempotency_key: str = Field(min_length=1, max_length=200)
    preflight_sha256: Digest


class GenerationExperimentStart(_Contract):
    """Start an accepted comparison: queue one picture per choice, exactly as accepted.

    The digest is the comparison's own, so a start aimed at a record that has
    changed since it was read starts nothing.
    """

    idempotency_key: str = Field(min_length=1, max_length=200)
    snapshot_sha256: Digest
    confirm_expensive: bool = False


class ExperimentTrialOut(BaseModel):
    id: str
    ordinal: int
    seed: int
    state: GenerationExperimentTrialState
    work_step_id: str | None = None
    run_id: str | None = None
    job_id: str | None = None
    status: TrialWorkStatus | None = None


class ExperimentArmOut(BaseModel):
    id: str
    ordinal: int
    label: str
    profile_id: str
    profile_name: str | None = None
    workflow_revision_id: str
    workflow_version: int | None = None
    workflow_activation_id: str | None = None
    model_family: str | None = None
    width: int
    height: int
    effective_settings: dict[str, JsonValue]
    trigger_words_applied: list[str]
    snapshot_sha256: str
    trials: list[ExperimentTrialOut]


class GenerationExperimentEvaluationCreate(_Contract):
    """One choice preferred, or a tie, or neither suiting, with an optional short note."""

    preference: GenerationExperimentPreference = Field(strict=False)
    # The choice preferred, by its ordinal; given with a preference and only then.
    arm_ordinal: int | None = Field(default=None, ge=1, le=2)
    note: Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)] | None = None

    @model_validator(mode="after")
    def require_a_choice_only_when_preferred(self) -> Self:
        preferred = self.preference == GenerationExperimentPreference.PREFERRED
        if preferred != (self.arm_ordinal is not None):
            raise ValueError(
                "A preference names the choice preferred, and a tie or neither does not."
            )
        return self


class ExperimentEvaluationOut(BaseModel):
    preference: GenerationExperimentPreference
    mode: GenerationExperimentEvaluationMode
    arm_ordinal: int | None = None
    note: str | None = None
    created_at: datetime

    @field_serializer("created_at", when_used="json")
    def serialize_time_as_utc(self, value: datetime) -> str:
        """The database keeps it without a zone and it is UTC; say so."""

        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat().replace("+00:00", "Z")


class GenerationExperimentOut(BaseModel):
    id: str
    name: str
    state: GenerationExperimentState
    operation: Literal["text_to_image"]
    prompt: str
    negative_prompt: str
    geometry: Geometry
    seed_policy: SeedPolicy
    seed_equivalence: Literal["same_family", "none"]
    preflight_sha256: str
    snapshot_sha256: str
    estimate: list[ResourceEvidenceOut]
    created_at: datetime
    work_plan_id: str | None = None
    started_at: datetime | None = None
    arms: list[ExperimentArmOut]
    # The latest thing said of the pictures; none until something is.
    evaluation: ExperimentEvaluationOut | None = None
    evaluation_mode: GenerationExperimentEvaluationMode = (
        GenerationExperimentEvaluationMode.UNBLINDED
    )
    # True while a blind comparison waits for its blind saying: until then no
    # answer links a picture to the choice that made it.
    blind_pending: bool = False

    @field_serializer("created_at", "started_at", when_used="json")
    def serialize_times_as_utc(self, value: datetime | None) -> str | None:
        """The database keeps them without a zone and they are UTC; say so in every answer."""

        if value is None:
            return None
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat().replace("+00:00", "Z")


# Why a setting a choice ran with is not in the recipe drafted from it: one
# fixed sentence each, so a value never reaches the answer through a reason.
RECIPE_LEFT_OUT_MESSAGES: dict[str, str] = {
    "recipe-prompt": "A recipe never holds the words; each request brings its own.",
    "recipe-seed": "A recipe leaves the seed to each request, so each one comes out new.",
    "recipe-output-count": "A recipe leaves how many results to make to each request.",
    "recipe-matched-loras": (
        "These were chosen for this request's words; each request chooses its own."
    ),
    "comparison-adapted": (
        "The comparison changed this for its own pictures, so the recipe leaves it to each request."
    ),
    "workflow-use-case-preset-setting-unavailable": "This workflow does not let a recipe set this.",
    "workflow-use-case-preset-settings-invalid": (
        "This workflow does not accept this value from a recipe."
    ),
    "recipe-unsupported": "A recipe for this workflow cannot hold this setting.",
}


class RecipeDraftLeftOut(BaseModel):
    setting: str
    reason: str
    message: str


class GenerationExperimentRecipeDraftOut(BaseModel):
    """A recipe to review before saving: the settings one choice ran with, as a recipe holds them.

    The model and workflow are named, not held: a recipe has neither, and both
    already exist to be chosen beside it.
    """

    experiment_id: str
    arm_ordinal: int
    use_case: Literal["image_generation"]
    name: str
    settings_json: dict[str, JsonValue]
    left_out: list[RecipeDraftLeftOut]
    profile_id: str
    profile_name: str | None = None
    workflow_id: str
    # The family a chat chooses its workflow by; none for a workflow outside one.
    workflow_family_id: str | None = None
    workflow_revision_id: str
    workflow_name: str
    workflow_version: int | None = None


class BlindPositionOut(BaseModel):
    """One picture in a blind view, known only by where it is shown."""

    position: int
    status: TrialWorkStatus | None = None
    ready: bool


class BlindRevealOut(BaseModel):
    position: int
    arm_ordinal: int
    label: str


class GenerationExperimentBlindViewOut(BaseModel):
    """A blind comparison as one viewing shows it: its own order, and no choice named.

    The reveal comes with the saying and not before.
    """

    id: str
    experiment_id: str
    positions: list[BlindPositionOut]
    evaluation: ExperimentEvaluationOut | None = None
    reveal: list[BlindRevealOut] | None = None


class GenerationExperimentBlindEvaluationCreate(_Contract):
    """A preference said in a blind view: a picture by its position, or a tie, or neither."""

    preference: GenerationExperimentPreference = Field(strict=False)
    position: int | None = Field(default=None, ge=1, le=2)
    note: Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)] | None = None

    @model_validator(mode="after")
    def require_a_picture_only_when_preferred(self) -> Self:
        preferred = self.preference == GenerationExperimentPreference.PREFERRED
        if preferred != (self.position is not None):
            raise ValueError(
                "A preference names the picture preferred, and a tie or neither does not."
            )
        return self
