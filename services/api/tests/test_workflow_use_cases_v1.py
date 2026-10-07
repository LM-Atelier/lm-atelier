"""Use-case requirements follow structured inputs without choosing a workflow."""

import importlib
import importlib.util
from dataclasses import FrozenInstanceError
from types import ModuleType

import pytest

from local_lm.domain import Operation


def _contract() -> ModuleType:
    name = "local_lm.workflow_use_cases_v1"
    assert importlib.util.find_spec(name) is not None, (
        "Structured use-case classification is absent."
    )
    return importlib.import_module(name)


@pytest.mark.parametrize(
    ("operation", "source", "selection", "extend", "upscale", "expected", "capability"),
    [
        (Operation.TEXT, False, "none", False, False, "chat", "chat"),
        (Operation.TEXT_TO_IMAGE, False, "none", False, False, "image_generation", "image"),
        (Operation.IMAGE_TO_IMAGE, True, "none", False, False, "image_edit", "image"),
        (Operation.IMAGE_TO_IMAGE, True, "workflow", False, False, "image_inpaint", "image"),
        (Operation.IMAGE_TO_IMAGE, True, "none", True, False, "image_outpaint", "image"),
        (Operation.IMAGE_TO_IMAGE, True, "none", False, True, "image_upscale", "image"),
        (Operation.TEXT_TO_VIDEO, False, "none", False, False, "video_generation", "video"),
        (Operation.IMAGE_TO_VIDEO, True, "none", False, False, "video_animate", "video"),
    ],
)
def test_documented_use_cases_follow_operation_and_input_facts(
    operation: Operation,
    source: bool,
    selection: str,
    extend: bool,
    upscale: bool,
    expected: str,
    capability: str,
) -> None:
    contract = _contract()
    facts = contract.WorkflowUseCaseInputs(
        operation=operation,
        source_present=source,
        selection=contract.SelectionApplication(selection),
        extend=extend,
        upscale=upscale,
    )
    result = contract.classify_workflow_use_case(facts)
    assert result.use_case.value == expected
    assert result.operation is operation and result.capability == capability
    assert result.requires_source is (
        operation in {Operation.IMAGE_TO_IMAGE, Operation.IMAGE_TO_VIDEO}
    )
    assert result.requires_mask_input is (selection == "workflow")
    assert result.requires_outpaint_input is extend
    assert result.requires_upscale_input is upscale


def test_blended_selection_does_not_require_a_workflow_mask_input() -> None:
    contract = _contract()
    result = contract.classify_workflow_use_case(
        contract.WorkflowUseCaseInputs(
            operation=Operation.IMAGE_TO_IMAGE,
            source_present=True,
            selection=contract.SelectionApplication.BLEND,
        )
    )
    assert result.use_case.value == "image_inpaint"
    assert result.requires_source and not result.requires_mask_input
    assert contract.supports_required_use_case_inputs(result, None)


@pytest.mark.parametrize("operation", [Operation.IMAGE_TO_IMAGE, Operation.IMAGE_TO_VIDEO])
def test_source_based_operations_require_a_resolved_source(operation: Operation) -> None:
    contract = _contract()
    with pytest.raises(contract.WorkflowUseCaseError) as refused:
        contract.classify_workflow_use_case(contract.WorkflowUseCaseInputs(operation=operation))
    assert refused.value.code == "workflow-use-case-source-required"


@pytest.mark.parametrize(
    "facts",
    [
        {"source_present": 1},
        {"source_present": None},
        {"extend": "yes"},
        {"extend": 1},
        {"upscale": []},
        {"selection": "workflow"},
        {"operation": "image_to_image"},
    ],
)
def test_input_facts_are_typed_and_never_coerced(facts: dict[str, object]) -> None:
    contract = _contract()
    with pytest.raises(contract.WorkflowUseCaseError) as refused:
        contract.classify_workflow_use_case(
            contract.WorkflowUseCaseInputs(**{"operation": Operation.TEXT, **facts})
        )
    assert refused.value.code == "workflow-use-case-input-invalid"


@pytest.mark.parametrize(
    ("selection", "extend", "upscale"),
    [("workflow", True, False), ("blend", False, True), ("none", True, True)],
)
def test_competing_specialized_operations_are_not_silently_ranked(
    selection: str, extend: bool, upscale: bool
) -> None:
    contract = _contract()
    with pytest.raises(contract.WorkflowUseCaseError) as refused:
        contract.classify_workflow_use_case(
            contract.WorkflowUseCaseInputs(
                operation=Operation.IMAGE_TO_IMAGE,
                source_present=True,
                selection=contract.SelectionApplication(selection),
                extend=extend,
                upscale=upscale,
            )
        )
    assert refused.value.code == "workflow-use-case-input-ambiguous"


@pytest.mark.parametrize(
    "operation",
    [Operation.TEXT, Operation.TEXT_TO_IMAGE, Operation.TEXT_TO_VIDEO, Operation.IMAGE_TO_VIDEO],
)
@pytest.mark.parametrize("specialization", ["workflow", "blend", "extend", "upscale"])
def test_image_edit_facts_do_not_reclassify_another_operation(
    operation: Operation, specialization: str
) -> None:
    contract = _contract()
    facts = contract.WorkflowUseCaseInputs(
        operation=operation,
        source_present=True,
        selection=contract.SelectionApplication(
            specialization if specialization in {"workflow", "blend"} else "none"
        ),
        extend=specialization == "extend",
        upscale=specialization == "upscale",
    )
    with pytest.raises(contract.WorkflowUseCaseError) as refused:
        contract.classify_workflow_use_case(facts)
    assert refused.value.code == "workflow-use-case-input-invalid"


@pytest.mark.parametrize(
    ("specialization", "field", "kind"),
    [
        ("workflow", "mask", "mask"),
        ("extend", "outpaint_margins", "outpaint"),
        ("upscale", "upscale_factor", "upscale"),
    ],
)
def test_specialized_input_support_uses_the_canonical_schema_declarations(
    specialization: str, field: str, kind: str
) -> None:
    contract = _contract()
    result = contract.classify_workflow_use_case(
        contract.WorkflowUseCaseInputs(
            operation=Operation.IMAGE_TO_IMAGE,
            source_present=True,
            selection=contract.SelectionApplication.WORKFLOW
            if specialization == "workflow"
            else contract.SelectionApplication.NONE,
            extend=specialization == "extend",
            upscale=specialization == "upscale",
        )
    )
    assert not contract.supports_required_use_case_inputs(result, None)
    assert not contract.supports_required_use_case_inputs(
        result, {"title": kind, "properties": {field: {"title": kind}}}
    )
    assert not contract.supports_required_use_case_inputs(
        result, {"properties": {field: {"x-lm-atelier-kind": "other"}}}
    )
    assert contract.supports_required_use_case_inputs(
        result, {"properties": {field: {"x-lm-atelier-kind": kind}}}
    )


def test_chat_with_a_picture_keeps_the_classified_text_operation() -> None:
    contract = _contract()
    facts = contract.WorkflowUseCaseInputs(operation=Operation.TEXT, source_present=True)
    result = contract.classify_workflow_use_case(facts)
    assert result.use_case.value == "chat"
    with pytest.raises(FrozenInstanceError):
        facts.source_present = False
    with pytest.raises(FrozenInstanceError):
        result.requires_source = True
