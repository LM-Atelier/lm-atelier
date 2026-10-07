"""Classify execution requirements from operation and resolved input facts."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, assert_never

from .domain import Operation, operation_model_role
from .outpaint_workflows import workflow_declares_outpaint
from .studio_masks import workflow_accepts_mask
from .upscale_workflows import workflow_declares_upscale


class WorkflowUseCase(StrEnum):
    CHAT = "chat"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"
    IMAGE_INPAINT = "image_inpaint"
    IMAGE_OUTPAINT = "image_outpaint"
    IMAGE_UPSCALE = "image_upscale"
    VIDEO_GENERATION = "video_generation"
    VIDEO_ANIMATE = "video_animate"


class SelectionApplication(StrEnum):
    NONE = "none"
    WORKFLOW = "workflow"
    BLEND = "blend"


UseCaseRefusal = Literal[
    "workflow-use-case-input-invalid",
    "workflow-use-case-input-ambiguous",
    "workflow-use-case-source-required",
]


class WorkflowUseCaseError(ValueError):
    def __init__(self, code: UseCaseRefusal) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class WorkflowUseCaseInputs:
    """Carry content-free facts after request operation and inputs are resolved.

    The caller validates mask geometry, margins and scale values through their
    existing contracts. These markers say which contracts the request uses;
    they contain no prompt, image, workflow name or artifact identifier.
    """

    operation: Operation
    source_present: bool = False
    selection: SelectionApplication = SelectionApplication.NONE
    extend: bool = False
    upscale: bool = False


@dataclass(frozen=True)
class WorkflowUseCaseRequirements:
    """Describe the operation and input declarations a candidate must support."""

    use_case: WorkflowUseCase
    operation: Operation
    capability: Literal["chat", "image", "video"]
    requires_source: bool
    selection: SelectionApplication
    requires_mask_input: bool
    requires_outpaint_input: bool
    requires_upscale_input: bool


def classify_workflow_use_case(
    facts: WorkflowUseCaseInputs,
) -> WorkflowUseCaseRequirements:
    """Resolve one use case without ranking workflows or reading request content."""
    if (
        not isinstance(facts.operation, Operation)
        or not isinstance(facts.selection, SelectionApplication)
        or any(
            type(value) is not bool for value in (facts.source_present, facts.extend, facts.upscale)
        )
    ):
        raise WorkflowUseCaseError("workflow-use-case-input-invalid")
    specialized = sum((facts.selection != SelectionApplication.NONE, facts.extend, facts.upscale))
    if specialized > 1:
        raise WorkflowUseCaseError("workflow-use-case-input-ambiguous")
    if specialized and facts.operation != Operation.IMAGE_TO_IMAGE:
        raise WorkflowUseCaseError("workflow-use-case-input-invalid")
    needs_source = facts.operation in {
        Operation.IMAGE_TO_IMAGE,
        Operation.IMAGE_TO_VIDEO,
    }
    if needs_source and not facts.source_present:
        raise WorkflowUseCaseError("workflow-use-case-source-required")

    match facts.operation:
        case Operation.TEXT:
            use_case = WorkflowUseCase.CHAT
        case Operation.TEXT_TO_IMAGE:
            use_case = WorkflowUseCase.IMAGE_GENERATION
        case Operation.TEXT_TO_VIDEO:
            use_case = WorkflowUseCase.VIDEO_GENERATION
        case Operation.IMAGE_TO_VIDEO:
            use_case = WorkflowUseCase.VIDEO_ANIMATE
        case Operation.IMAGE_TO_IMAGE:
            if facts.selection != SelectionApplication.NONE:
                use_case = WorkflowUseCase.IMAGE_INPAINT
            elif facts.extend:
                use_case = WorkflowUseCase.IMAGE_OUTPAINT
            elif facts.upscale:
                use_case = WorkflowUseCase.IMAGE_UPSCALE
            else:
                use_case = WorkflowUseCase.IMAGE_EDIT
        case _:
            assert_never(facts.operation)

    return WorkflowUseCaseRequirements(
        use_case=use_case,
        operation=facts.operation,
        capability=operation_model_role(facts.operation),
        requires_source=needs_source,
        selection=facts.selection,
        requires_mask_input=facts.selection == SelectionApplication.WORKFLOW,
        requires_outpaint_input=facts.extend,
        requires_upscale_input=facts.upscale,
    )


def supports_required_use_case_inputs(
    requirements: WorkflowUseCaseRequirements, input_schema: dict[str, Any] | None
) -> bool:
    """Check required input declarations through their canonical inspectors.

    Trust, readiness, graph bindings and setting validation remain separate
    admission requirements before a workflow can execute.
    """
    return (
        (not requirements.requires_mask_input or workflow_accepts_mask(input_schema))
        and (not requirements.requires_outpaint_input or workflow_declares_outpaint(input_schema))
        and (not requirements.requires_upscale_input or workflow_declares_upscale(input_schema))
    )
