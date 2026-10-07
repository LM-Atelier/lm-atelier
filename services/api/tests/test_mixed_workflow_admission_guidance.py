"""Mixed workflow refusals explain both instruction and recipe requirements."""

import pytest

from local_lm.domain import Operation
from local_lm.workflow_selection import WorkflowFamilySelectionError
from local_lm.workflow_use_case_errors import workflow_use_case_error


@pytest.mark.parametrize(
    "reasons",
    [
        ("workflow-instruction-edit-required", "workflow-use-case-mask-unsupported"),
        ("workflow-use-case-mask-unsupported", "workflow-instruction-edit-required"),
        (
            "workflow-use-case-upscale-unsupported",
            "workflow-instruction-edit-required",
            "unrecognized-condition",
        ),
    ],
)
def test_mixed_refusals_explain_instruction_and_recipe_requirements(
    reasons: tuple[str, ...],
) -> None:
    error = WorkflowFamilySelectionError(
        capability="image",
        operation=Operation.IMAGE_TO_IMAGE,
        reason="no_ready_workflow",
        candidate_reasons=reasons,
    )
    translated = workflow_use_case_error(error)
    assert translated is not None
    code, message = translated
    assert code == "workflow-use-case-no-compatible-workflow"
    assert "instruction-edit workflow" in message
    assert "recipe's inputs and settings" in message
    assert "Workflows" in message
    assert "Recipes for this chat" in message
    assert "unrecognized-condition" not in message


def test_recipe_only_aggregate_keeps_its_existing_guidance() -> None:
    error = WorkflowFamilySelectionError(
        capability="image",
        operation=Operation.IMAGE_TO_IMAGE,
        reason="no_ready_workflow",
        candidate_reasons=(
            "workflow-use-case-mask-unsupported",
            "workflow-use-case-upscale-unsupported",
        ),
    )
    assert workflow_use_case_error(error) == (
        "workflow-use-case-no-compatible-workflow",
        "No ready workflow supports the recipe's inputs and settings. "
        "Choose a compatible workflow or recipe, or select Automatic (no recipe) "
        "under Recipes for this chat.",
    )


def test_unrecognized_reasons_do_not_obscure_a_single_instruction_refusal() -> None:
    error = WorkflowFamilySelectionError(
        capability="image",
        operation=Operation.IMAGE_TO_IMAGE,
        reason="no_ready_workflow",
        candidate_reasons=("workflow-instruction-edit-required", "unrecognized-condition"),
    )
    translated = workflow_use_case_error(error)
    assert translated is not None
    code, message = translated
    assert code == "workflow-instruction-edit-required"
    assert "instruction-edit workflow" in message
    assert "choose a workflow explicitly" in message
    assert "unrecognized-condition" not in message
