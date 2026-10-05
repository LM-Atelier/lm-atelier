"""A recipe drafted from one choice of a comparison, so a setup that won can be kept.

A recipe holds settings only: no model and no workflow. The choice's model and
workflow already exist, so the draft carries what is new, the settings the
choice ran with, and names the model and workflow they belong with. Each
setting is checked the way a turn checks a saved recipe against this exact
workflow, so the draft holds only what a recipe here can hold, and says why
anything else was left out. Nothing is written: the person reviews the draft
and saves it through the ordinary recipe editor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from . import generation_experiment_store as store
from .auxiliary_assets import revision_accepts_added_loras
from .domain import Operation, operation_model_role
from .generation_experiments_v1 import (
    RECIPE_LEFT_OUT_MESSAGES,
    GenerationExperimentRecipeDraftOut,
    RecipeDraftLeftOut,
)
from .models import GenerationExperiment, WorkflowDefinition, WorkflowRevision
from .workflow_use_case_preset_admission import (
    WorkflowUseCasePresetAdmissionError,
    admit_workflow_use_case_preset,
)
from .workflow_use_case_preset_resolution import ResolvedWorkflowUseCasePreset
from .workflow_use_case_presets_v1 import PROMPT_SETTING_KEYS
from .workflow_use_cases_v1 import WorkflowUseCase, WorkflowUseCaseInputs

if TYPE_CHECKING:
    from .orchestrator import ConversationOrchestrator

#: The comparisons whose choices a recipe can carry, by what they make from words.
USE_CASES = {
    Operation.TEXT_TO_IMAGE: WorkflowUseCase.IMAGE_GENERATION,
    Operation.TEXT_TO_VIDEO: WorkflowUseCase.VIDEO_GENERATION,
}


class RecipeDraftRefused(Exception):
    """A draft cannot be made; the code names why, from the comparison API's fixed set."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


async def recipe_draft(
    orchestrator: ConversationOrchestrator,
    session: Session,
    experiment_id: str,
    arm_ordinal: int,
) -> GenerationExperimentRecipeDraftOut:
    experiment = session.get(GenerationExperiment, experiment_id)
    if experiment is None:
        raise RecipeDraftRefused("generation-experiment-not-found")
    # Only a record that still holds what was accepted is drafted from.
    try:
        store.out(experiment)
    except store.GenerationExperimentRecordError:
        raise RecipeDraftRefused("generation-experiment-record-invalid") from None
    # Until a blind comparison's preference is said, which picture a choice
    # made is not to be known, and a draft would lead from one to the other.
    if store.blind_pending(experiment):
        raise RecipeDraftRefused("generation-experiment-blind")
    # A recipe here is one for making pictures or videos from words; a way of
    # changing a picture is kept as an edit's recipe instead.
    operation = Operation(experiment.operation)
    use_case = USE_CASES.get(operation)
    if use_case is None:
        raise RecipeDraftRefused("generation-experiment-recipe-not-for-changes")
    arm = next((arm for arm in experiment.arms if arm.ordinal == arm_ordinal), None)
    if arm is None:
        raise RecipeDraftRefused("generation-experiment-arm-not-found")
    revision = session.get(WorkflowRevision, arm.workflow_revision_id)
    definition = session.get(WorkflowDefinition, revision.workflow_id) if revision else None
    if revision is None or definition is None:
        raise RecipeDraftRefused("generation-experiment-recipe-unavailable")
    profile = arm.snapshot_json.get("profile") or {}
    engine = profile.get("engine") if isinstance(profile.get("engine"), str) else revision.engine
    fields = await orchestrator.engines.settings_for_role(
        operation_model_role(operation), engine=engine
    )
    accepts_added_loras = revision_accepts_added_loras(revision)

    def admit(settings: dict[str, Any]) -> None:
        admit_workflow_use_case_preset(
            ResolvedWorkflowUseCasePreset(
                use_case=use_case,
                mode="preset",
                scope="workspace",
                # Checked as a saved recipe is checked; the choice stands in
                # for the recipe that does not exist yet.
                preset_id=arm.id,
                preset_name=arm.label,
                settings_json=settings,
            ),
            WorkflowUseCaseInputs(operation),
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
        raise RecipeDraftRefused("generation-experiment-recipe-unavailable") from None
    candidate, left_out = _recipe_candidate(arm.effective_settings_json, arm.snapshot_json)
    kept, refused = admitted_recipe_settings(candidate, admit)
    workflow = arm.snapshot_json.get("workflow") or {}
    return GenerationExperimentRecipeDraftOut(
        experiment_id=experiment.id,
        arm_ordinal=arm.ordinal,
        use_case=use_case.value,
        name=arm.label,
        settings_json=kept,
        left_out=sorted(left_out + refused, key=lambda item: item.setting),
        profile_id=arm.profile_id,
        profile_name=profile.get("name") if isinstance(profile.get("name"), str) else None,
        workflow_id=definition.id,
        workflow_family_id=definition.family_id,
        workflow_revision_id=revision.id,
        workflow_name=definition.name,
        workflow_version=workflow.get("version")
        if isinstance(workflow.get("version"), int)
        else None,
    )


def _recipe_candidate(
    effective: dict[str, Any], snapshot: dict[str, Any]
) -> tuple[dict[str, Any], list[RecipeDraftLeftOut]]:
    """The settings the choice ran with, less what was the comparison's rather than the setup's."""

    candidate = dict(effective)
    left_out: list[RecipeDraftLeftOut] = []
    for key in sorted(PROMPT_SETTING_KEYS.intersection(candidate)):
        candidate.pop(key)
        left_out.append(recipe_left_out(key, "recipe-prompt"))
    # The comparison made one picture per choice whatever was asked; the count
    # the setup itself would make is left to each request.
    adapted = {
        item["setting"]
        for item in snapshot.get("adaptations") or []
        if isinstance(item, dict) and isinstance(item.get("setting"), str)
    }
    for key in sorted(adapted.intersection(candidate)):
        candidate.pop(key)
        left_out.append(recipe_left_out(key, "comparison-adapted"))
    return candidate, left_out


def admitted_recipe_settings(
    candidate: dict[str, Any], admit: Callable[[dict[str, Any]], None]
) -> tuple[dict[str, Any], list[RecipeDraftLeftOut]]:
    """Keep every setting a recipe here can hold, each refused one named with its reason."""

    try:
        admit(candidate)
    except WorkflowUseCasePresetAdmissionError:
        pass
    else:
        return candidate, []
    kept: dict[str, Any] = {}
    refused: list[RecipeDraftLeftOut] = []
    for key in sorted(candidate):
        try:
            admit({key: candidate[key]})
        except WorkflowUseCasePresetAdmissionError as exc:
            refused.append(recipe_left_out(key, exc.code))
        else:
            kept[key] = candidate[key]
    try:
        admit(kept)
    except WorkflowUseCasePresetAdmissionError as exc:
        # Settings that pass one by one and fail together cannot be told apart
        # here, so none of them is kept rather than a draft that cannot run.
        return {}, refused + [recipe_left_out(key, exc.code) for key in sorted(kept)]
    return kept, refused


def recipe_left_out(setting: str, reason: str) -> RecipeDraftLeftOut:
    code = reason if reason in RECIPE_LEFT_OUT_MESSAGES else "recipe-unsupported"
    return RecipeDraftLeftOut(setting=setting, reason=code, message=RECIPE_LEFT_OUT_MESSAGES[code])
