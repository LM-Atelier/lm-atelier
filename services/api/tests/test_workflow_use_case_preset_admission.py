"""The selected recipe must satisfy both input structure and exact settings."""

from dataclasses import replace

import pytest
from sqlalchemy.orm import Session
from test_workflow_selection import _family_revision
from test_workflow_selection import session as session

from local_lm.domain import Operation
from local_lm.models import WorkflowDefinition, WorkflowRevision
from local_lm.settings_registry import builtin_settings_for_role
from local_lm.workflow_selection import WorkflowFamilySelectionError
from local_lm.workflow_use_case_preset_admission import (
    WorkflowUseCasePresetAdmissionError,
    admit_workflow_use_case_preset,
    resolve_workflow_use_case_preset_family,
)
from local_lm.workflow_use_case_preset_resolution import ResolvedWorkflowUseCasePreset
from local_lm.workflow_use_cases_v1 import (
    SelectionApplication,
    WorkflowUseCase,
    WorkflowUseCaseInputs,
)


def _candidate():
    recipe = ResolvedWorkflowUseCasePreset(
        WorkflowUseCase.IMAGE_EDIT, "preset", "chat", "recipe", "Example", {"seed": 9}
    )
    definition = WorkflowDefinition(id="definition", operation="image_to_image")
    revision = WorkflowRevision(
        id="revision",
        workflow_id=definition.id,
        engine="comfyui",
        api_graph_json={"source": {"inputs": {"image": "${input_image}"}}},
        input_schema_json={},
    )
    facts = WorkflowUseCaseInputs(Operation.IMAGE_TO_IMAGE, source_present=True)
    return recipe, facts, revision, definition


def _admit(recipe, facts, revision, definition, expected_revision_id="revision"):
    return admit_workflow_use_case_preset(
        recipe,
        facts,
        revision,
        definition,
        expected_revision_id=expected_revision_id,
        fields=builtin_settings_for_role("image"),
    )


def test_admission_binds_one_detached_recipe_to_the_exact_revision() -> None:
    recipe, facts, revision, definition = _candidate()
    admitted = _admit(recipe, facts, revision, definition)
    recipe.settings_json["seed"] = 10
    assert admitted.workflow_revision_id == revision.id
    assert admitted.preset.settings_json == {"seed": 9}
    assert admitted.preset.preset_id == "recipe"
    assert admitted.upscale_kind is None


@pytest.mark.parametrize("missing", ["revision", "definition"])
def test_a_missing_revision_contract_does_not_bypass_admission(missing: str) -> None:
    recipe, facts, revision, definition = _candidate()
    with pytest.raises(WorkflowUseCasePresetAdmissionError) as raised:
        _admit(
            recipe,
            facts,
            None if missing == "revision" else revision,
            None if missing == "definition" else definition,
        )
    assert raised.value.code == "workflow-use-case-preset-revision-required"


@pytest.mark.parametrize("mutation", ["expected", "definition", "operation", "use_case"])
def test_revision_and_use_case_identity_cannot_drift(mutation: str) -> None:
    recipe, facts, revision, definition = _candidate()
    expected = "revision"
    reason = "workflow-use-case-preset-revision-mismatch"
    if mutation == "expected":
        expected = "another-revision"
    elif mutation == "definition":
        definition.id = "another-definition"
    elif mutation == "operation":
        definition.operation = "text_to_image"
        reason = "workflow-use-case-operation-mismatch"
    else:
        recipe = replace(recipe, use_case=WorkflowUseCase.IMAGE_GENERATION)
        reason = "workflow-use-case-preset-mismatch"
    with pytest.raises(WorkflowUseCasePresetAdmissionError) as raised:
        _admit(recipe, facts, revision, definition, expected)
    assert raised.value.code == reason
    assert str(raised.value) == reason


def test_valid_settings_cannot_admit_a_workflow_without_the_required_mask() -> None:
    recipe, facts, revision, definition = _candidate()
    recipe = replace(recipe, use_case=WorkflowUseCase.IMAGE_INPAINT)
    facts = replace(facts, selection=SelectionApplication.WORKFLOW)
    with pytest.raises(WorkflowUseCasePresetAdmissionError) as raised:
        _admit(recipe, facts, revision, definition)
    assert raised.value.code == "workflow-use-case-mask-unsupported"


def test_valid_input_structure_cannot_hide_an_unknown_setting() -> None:
    recipe, facts, revision, definition = _candidate()
    recipe.settings_json["unrecognized_control"] = 7
    with pytest.raises(WorkflowUseCasePresetAdmissionError) as raised:
        _admit(recipe, facts, revision, definition)
    assert raised.value.code == "workflow-use-case-preset-setting-unsupported"
    assert "unrecognized_control" not in str(raised.value)


def test_rechecking_a_changed_revision_refuses_the_previously_valid_recipe() -> None:
    recipe, facts, revision, definition = _candidate()
    _admit(recipe, facts, revision, definition)
    revision.api_graph_json = {"node": {"class_type": "TestOutput"}}
    with pytest.raises(WorkflowUseCasePresetAdmissionError) as raised:
        _admit(recipe, facts, revision, definition)
    assert raised.value.code == "workflow-use-case-source-binding-missing"


@pytest.mark.parametrize("mode", ["explicit", "default", "automatic"])
def test_family_resolution_applies_settings_before_ranking(session: Session, mode: str) -> None:
    selected, _, denied = _family_revision(session, "Selected", is_default=True)
    alternative, _, allowed = _family_revision(session, "Alternative", use_case="general")
    denied.input_schema_json = {"properties": {"seed": {"type": "integer", "readOnly": True}}}
    session.flush()
    recipe = ResolvedWorkflowUseCasePreset(
        WorkflowUseCase.IMAGE_GENERATION, "preset", "workspace", "recipe", "Example", {"seed": 9}
    )
    facts = WorkflowUseCaseInputs(Operation.TEXT_TO_IMAGE)
    arguments = dict(
        mode=mode,
        engine="comfyui",
        fields=iter(builtin_settings_for_role("image")),
        workflow_family_id=selected.id if mode == "explicit" else None,
    )
    if mode != "automatic":
        with pytest.raises(WorkflowFamilySelectionError) as raised:
            resolve_workflow_use_case_preset_family(session, recipe, facts, **arguments)
        assert raised.value.reason == "workflow-use-case-preset-setting-unavailable"
        assert raised.value.workflow_family_id == selected.id
    else:
        result = resolve_workflow_use_case_preset_family(session, recipe, facts, **arguments)
        assert result.selection.workflow_family_id == alternative.id
        assert result.selection.workflow_revision_id == allowed.id
        assert result.admission.workflow_revision_id == allowed.id
        assert result.admission.preset.settings_json == {"seed": 9}
        recipe.settings_json["seed"] = 10
        assert result.admission.preset.settings_json == {"seed": 9}


def test_an_automatic_recipe_choice_does_not_replace_an_explicit_family(session: Session) -> None:
    selected, _, _ = _family_revision(session, "Selected", operation=Operation.IMAGE_TO_IMAGE)
    _, _, alternative = _family_revision(
        session, "Alternative", operation=Operation.IMAGE_TO_IMAGE, is_default=True
    )
    alternative.api_graph_json = {"node": {"inputs": {"image": "${input_image}"}}}
    session.flush()
    recipe = ResolvedWorkflowUseCasePreset(WorkflowUseCase.IMAGE_EDIT, "automatic", "chat")
    with pytest.raises(WorkflowFamilySelectionError) as raised:
        resolve_workflow_use_case_preset_family(
            session,
            recipe,
            WorkflowUseCaseInputs(Operation.IMAGE_TO_IMAGE, source_present=True),
            mode="explicit",
            engine="comfyui",
            fields=builtin_settings_for_role("image"),
            workflow_family_id=selected.id,
        )
    assert raised.value.reason == "workflow-use-case-source-binding-missing"
    assert raised.value.workflow_family_id == selected.id
