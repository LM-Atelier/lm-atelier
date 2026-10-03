"""A recipe drafted from one finished generation, so a setup that worked can be kept.

A recipe holds settings only, and never what one request brings for itself:
its words, its seed, how many results it makes, LoRAs matched to its words, or
its pictures. The generation's model and workflow already exist, so the draft
names them beside the settings. Each setting is checked the way a turn
checks a saved recipe against that exact workflow, and anything a recipe there
cannot hold is left out with its reason. Nothing is written: the person
reviews the draft and saves it through the ordinary recipe editor. The
generation and its record are never changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, JsonValue
from sqlalchemy.orm import Session

from .auxiliary_assets import revision_accepts_added_loras
from .domain import Operation, PartType, RunStatus, operation_model_role
from .engines import EngineNotConfiguredError, EngineSchemaUnavailableError
from .generation_experiment_recipe import admitted_recipe_settings, recipe_left_out
from .generation_experiments_v1 import RecipeDraftLeftOut
from .models import Message, ModelProfile, Run, WorkflowDefinition, WorkflowRevision
from .output_recipe import prompt_taken_back
from .workflow_graph_settings_v1 import GRAPH_SETTING_INPUT_NAMES, workflow_graph_settings
from .workflow_use_case_preset_admission import (
    WorkflowUseCasePresetAdmissionError,
    admit_workflow_use_case_preset,
)
from .workflow_use_case_preset_resolution import ResolvedWorkflowUseCasePreset
from .workflow_use_case_presets_v1 import PROMPT_SETTING_KEYS
from .workflow_use_cases_v1 import WorkflowUseCase, WorkflowUseCaseInputs

if TYPE_CHECKING:
    from .orchestrator import ConversationOrchestrator

#: The generations whose setup a recipe can carry. An edit is made from its
#: own picture and selection, so its setup is kept as a Studio recipe instead.
DRAFTED_USE_CASES: Final = {
    Operation.TEXT_TO_IMAGE: WorkflowUseCase.IMAGE_GENERATION,
    Operation.TEXT_TO_VIDEO: WorkflowUseCase.VIDEO_GENERATION,
    Operation.IMAGE_TO_VIDEO: WorkflowUseCase.VIDEO_ANIMATE,
}
#: Seeds that belong to one request rather than to a setup, whatever workflow
#: binds them; a workflow can also bind its own under generated names.
_SEEDS: Final = frozenset({"seed", "noise_seed"})
#: A run records a batch size of one whenever its request made several pictures,
#: one run each, so the value it holds never says how many the setup makes.
_OUTPUT_COUNT: Final = "batch_size"
_NAME_LIMIT: Final = 200
#: The longest instruction an edit recipe holds, as the recipe route accepts it.
_INSTRUCTION_LIMIT: Final = 20_000


class OutputRecipeDraftOut(BaseModel):
    """A recipe to review before saving: the settings a generation ran with, as a recipe holds them.

    The model and workflow are named, not held: a recipe has neither, and both
    already exist to be chosen beside it.
    """

    run_id: str
    use_case: Literal["image_generation", "video_generation", "video_animate"]
    name: str
    settings_json: dict[str, JsonValue]
    left_out: list[RecipeDraftLeftOut]
    profile_id: str | None = None
    profile_name: str | None = None
    workflow_id: str
    # The family a chat chooses its workflow by; none for a workflow outside one.
    workflow_family_id: str | None = None
    workflow_revision_id: str
    workflow_name: str
    workflow_version: int


class OutputRecipeDraftRefused(Exception):
    """Why no recipe can be drafted from this generation, as a status, a code and a sentence."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.message = message


async def output_recipe_draft(
    orchestrator: ConversationOrchestrator, session: Session, run_id: str
) -> OutputRecipeDraftOut:
    run = session.get(Run, run_id)
    if run is None:
        raise OutputRecipeDraftRefused(
            404, "output-recipe-run-not-found", "This generation no longer exists."
        )
    operation = Operation(run.operation) if run.operation in set(Operation) else None
    use_case = DRAFTED_USE_CASES.get(operation) if operation is not None else None
    if operation is None or use_case is None:
        raise OutputRecipeDraftRefused(
            422,
            "output-recipe-draft-unsupported",
            "Only a picture or video made from words, or a video made from a picture, "
            "can be kept as a recipe here.",
        )
    # A generation that did not finish has not shown that its setup works.
    if run.status != RunStatus.COMPLETE.value:
        raise OutputRecipeDraftRefused(
            409,
            "output-recipe-draft-unfinished",
            "Only a finished generation can be kept as a recipe.",
        )
    revision = (
        session.get(WorkflowRevision, run.workflow_revision_id)
        if run.workflow_revision_id
        else None
    )
    definition = session.get(WorkflowDefinition, revision.workflow_id) if revision else None
    if revision is None or definition is None:
        raise _unavailable()
    try:
        seeds = seed_settings(revision)
    except ValueError:
        # A workflow whose generated settings no longer read back cannot run.
        raise _unavailable() from None
    profile = session.get(ModelProfile, run.profile_id) if run.profile_id else None
    try:
        fields = await orchestrator.engines.settings_for_role(
            operation_model_role(operation),
            engine=profile.engine if profile is not None else revision.engine,
        )
    except EngineNotConfiguredError as exc:
        raise OutputRecipeDraftRefused(409, "engine-not-configured", str(exc)) from exc
    except EngineSchemaUnavailableError as exc:
        raise OutputRecipeDraftRefused(503, "engine-schema-unavailable", str(exc)) from exc
    accepts_added_loras = revision_accepts_added_loras(revision)
    name = definition.name.strip()[:_NAME_LIMIT].strip() or "Recipe"

    def admit(settings: dict[str, Any]) -> None:
        admit_workflow_use_case_preset(
            ResolvedWorkflowUseCasePreset(
                use_case=use_case,
                mode="preset",
                scope="workspace",
                # Checked as a saved recipe is checked; the generation stands in
                # for the recipe that does not exist yet.
                preset_id=run.id,
                preset_name=name,
                settings_json=settings,
            ),
            WorkflowUseCaseInputs(operation, source_present=operation == Operation.IMAGE_TO_VIDEO),
            revision,
            definition,
            expected_revision_id=revision.id,
            fields=fields,
            accepts_added_loras=accepts_added_loras,
        )

    # A workflow that takes no recipe at all takes none of these settings.
    try:
        admit({})
    except WorkflowUseCasePresetAdmissionError:
        raise _unavailable() from None
    candidate, left_out = _recipe_candidate(
        run.settings_json, seeds, loras_chosen=loras_were_chosen(run.provenance_json)
    )
    kept, refused = admitted_recipe_settings(candidate, admit)
    return OutputRecipeDraftOut(
        run_id=run.id,
        use_case=use_case.value,
        name=name,
        settings_json=kept,
        left_out=sorted(left_out + refused, key=lambda item: item.setting),
        profile_id=profile.id if profile is not None else None,
        profile_name=profile.name if profile is not None else None,
        workflow_id=definition.id,
        workflow_family_id=definition.family_id,
        workflow_revision_id=revision.id,
        workflow_name=definition.name,
        workflow_version=revision.version,
    )


def _recipe_candidate(
    settings: object, seeds: frozenset[str], *, loras_chosen: bool
) -> tuple[dict[str, Any], list[RecipeDraftLeftOut]]:
    """The settings the generation ran with, less what each request brings for itself."""

    candidate = dict(settings) if isinstance(settings, dict) else {}
    left_out: list[RecipeDraftLeftOut] = []
    for key in sorted(PROMPT_SETTING_KEYS.intersection(candidate)):
        candidate.pop(key)
        left_out.append(recipe_left_out(key, "recipe-prompt"))
    for key in sorted(seeds.intersection(candidate)):
        candidate.pop(key)
        left_out.append(recipe_left_out(key, "recipe-seed"))
    if _OUTPUT_COUNT in candidate:
        candidate.pop(_OUTPUT_COUNT)
        left_out.append(recipe_left_out(_OUTPUT_COUNT, "recipe-output-count"))
    if "loras" in candidate and not loras_chosen:
        # A stack matched to the words, or carried by a saved prompt, would be
        # applied to every later request whatever it says, and would stop each
        # one being matched for itself.
        candidate.pop("loras")
        left_out.append(recipe_left_out("loras", "recipe-matched-loras"))
    return candidate, left_out


def seed_settings(revision: WorkflowRevision) -> frozenset[str]:
    """Every setting that carries a seed: the plain ones, and each the workflow binds to one."""

    marker = workflow_graph_settings(revision.input_schema_json or {})
    bound = {
        binding["parameter"]
        for binding in (marker["bindings"] if marker is not None else [])
        if GRAPH_SETTING_INPUT_NAMES.get(binding.get("declared_name", binding["input_name"]))
        == "seed"
    }
    return _SEEDS | bound


def loras_were_chosen(provenance: dict[str, Any]) -> bool:
    """Whether its LoRAs were chosen for this generation, not by its words or a saved prompt."""

    assets = provenance.get("auxiliary_assets")
    selection = assets.get("selection") if isinstance(assets, dict) else None
    return (
        isinstance(selection, dict)
        and selection.get("mode") == "explicit"
        and "prompt_source" not in provenance
    )


class EditRecipeDraftOut(BaseModel):
    """The words an edit was asked with, to start an Image Studio recipe from that edit."""

    run_id: str
    instruction: str


def edit_recipe_draft(session: Session, run_id: str) -> EditRecipeDraftOut:
    """The words the person typed for this edit, as Image Studio keeps a recipe's instruction.

    Not the run's standalone prompt: a follow-up in a chat carries the earlier
    picture's whole prompt there, which would ask every later edit for that
    picture again. An edit that was one step of a longer request keeps that
    step's own words. Words taken back from the chat are never offered; the
    person writes the recipe's own instead.
    """

    run = session.get(Run, run_id)
    if run is None:
        raise OutputRecipeDraftRefused(
            404, "output-recipe-run-not-found", "This generation no longer exists."
        )
    if run.operation != Operation.IMAGE_TO_IMAGE.value:
        raise OutputRecipeDraftRefused(
            422, "edit-recipe-draft-unsupported", "Only an edit of a picture is kept this way."
        )
    return EditRecipeDraftOut(
        run_id=run.id, instruction=_typed_words(session, run)[:_INSTRUCTION_LIMIT]
    )


def _typed_words(session: Session, run: Run) -> str:
    if prompt_taken_back(session, run):
        return ""
    step = run.provenance_json.get("compiled_step")
    words = step.get("prompt") if isinstance(step, dict) else None
    if isinstance(words, str) and words.strip():
        return words
    message = session.get(Message, run.user_message_id)
    parts = sorted(message.parts, key=lambda part: part.position) if message is not None else []
    return next(
        (part.text for part in parts if part.type == PartType.TEXT.value and part.text),
        "",
    )


def kept_edit_settings(
    session: Session,
    settings: dict[str, Any],
    revision_id: str | None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """An edit recipe's settings without what one request brings for itself.

    Every seed the workflow binds, and LoRAs matched to the edit's words, stay
    out, as they do for a recipe from a generation.
    """

    revision = session.get(WorkflowRevision, revision_id) if revision_id else None
    try:
        seeds = seed_settings(revision) if revision is not None else _SEEDS
    except ValueError:
        seeds = _SEEDS
    kept = {key: value for key, value in settings.items() if key not in seeds}
    if "loras" in kept and not loras_were_chosen(provenance):
        kept.pop("loras")
    return kept


def _unavailable() -> OutputRecipeDraftRefused:
    return OutputRecipeDraftRefused(
        409,
        "output-recipe-draft-unavailable",
        "The workflow this was made with is gone or takes no recipe.",
    )
