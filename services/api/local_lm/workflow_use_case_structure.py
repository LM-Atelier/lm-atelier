"""Check necessary workflow input structure without selecting or trusting a revision."""

from dataclasses import dataclass
from typing import Any, Literal

from .domain import Operation
from .graph_placeholders import binds_parameter
from .media_references import reference_capacity
from .outpaint_workflows import source_pad_node, workflow_declares_outpaint
from .studio_masks import workflow_accepts_mask
from .upscale_workflows import upscale_capability, workflow_declares_upscale
from .workflow_use_cases_v1 import WorkflowUseCaseInputs, classify_workflow_use_case

StructureRefusal = Literal[
    "workflow-use-case-operation-mismatch",
    "workflow-use-case-mask-unsupported",
    "workflow-use-case-outpaint-unsupported",
    "workflow-use-case-upscale-unsupported",
    "workflow-use-case-graph-missing",
    "workflow-use-case-source-binding-missing",
    "workflow-use-case-mask-binding-missing",
    "workflow-use-case-outpaint-binding-missing",
    "workflow-use-case-upscale-binding-missing",
]


@dataclass(frozen=True)
class WorkflowUseCaseStructure:
    reason: StructureRefusal | None = None
    upscale_kind: Literal["model", "resample"] | None = None


def assess_workflow_use_case_structure(
    facts: WorkflowUseCaseInputs,
    *,
    operation: str,
    engine: str,
    api_graph: dict[str, Any] | None,
    input_schema: dict[str, Any] | None,
) -> WorkflowUseCaseStructure:
    """Check declarations and the selected adapter's known structural contracts.

    A result without a reason proves only these necessary input conditions.
    Trust, readiness, model fit, settings and output influence need their own
    admission checks. Other adapters retain their own graph contracts.
    """
    requirements = classify_workflow_use_case(facts)
    if operation != requirements.operation.value:
        return WorkflowUseCaseStructure("workflow-use-case-operation-mismatch")
    if requirements.requires_mask_input and not workflow_accepts_mask(input_schema):
        return WorkflowUseCaseStructure("workflow-use-case-mask-unsupported")
    if requirements.requires_outpaint_input and not workflow_declares_outpaint(input_schema):
        return WorkflowUseCaseStructure("workflow-use-case-outpaint-unsupported")
    if requirements.requires_upscale_input and not workflow_declares_upscale(input_schema):
        return WorkflowUseCaseStructure("workflow-use-case-upscale-unsupported")
    if engine != "comfyui" or requirements.operation == Operation.TEXT:
        return WorkflowUseCaseStructure()
    if not api_graph:
        return WorkflowUseCaseStructure("workflow-use-case-graph-missing")
    if requirements.requires_source and reference_capacity(api_graph) == 0:
        return WorkflowUseCaseStructure("workflow-use-case-source-binding-missing")
    if requirements.requires_mask_input and not binds_parameter(api_graph, "mask"):
        return WorkflowUseCaseStructure("workflow-use-case-mask-binding-missing")
    if requirements.requires_outpaint_input and source_pad_node(api_graph) is None:
        return WorkflowUseCaseStructure("workflow-use-case-outpaint-binding-missing")
    if requirements.requires_upscale_input:
        kind = upscale_capability(api_graph)
        if kind == "model":
            return WorkflowUseCaseStructure(upscale_kind="model")
        if kind == "resample":
            return WorkflowUseCaseStructure(upscale_kind="resample")
        return WorkflowUseCaseStructure("workflow-use-case-upscale-binding-missing")
    return WorkflowUseCaseStructure()
