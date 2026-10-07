"""Explain workflow admission failures without exposing stored values or identifiers."""

from .workflow_selection import WorkflowFamilySelectionError
from .workflow_use_case_preset_admission import WorkflowUseCasePresetAdmissionError
from .workflow_use_case_preset_resolution import WorkflowUseCasePresetResolutionError
from .workflow_use_case_preset_settings import WorkflowUseCasePresetSettingsError
from .workflow_use_cases_v1 import WorkflowUseCaseError

_CHOOSE = (
    "Choose a compatible workflow or recipe, or select Automatic (no recipe) "
    "under Recipes for this chat."
)
_MESSAGES = {
    "workflow-instruction-edit-required": (
        "Automatic localized edits need a ready instruction-edit workflow. "
        "Enable one in Workflows, or choose a workflow explicitly to use a strength edit."
    ),
    "workflow-use-case-preset-not-found": (
        "The selected recipe no longer exists. Choose another recipe or Automatic "
        "(no recipe) under Recipes for this chat."
    ),
    "workflow-use-case-preset-disabled": (
        "The selected recipe is disabled. Enable it in Manage recipes, or choose another "
        "recipe or Automatic (no recipe) under Recipes for this chat."
    ),
    "workflow-use-case-preset-revision-required": (
        "This recipe needs a workflow with a saved revision. Choose one that supports "
        "this request, or select Automatic (no recipe) under Recipes for this chat."
    ),
    "workflow-use-case-preset-setting-unsupported": (
        "The selected workflow does not support every setting in this recipe. "
        "Remove unsupported settings in Manage recipes. " + _CHOOSE
    ),
    "workflow-use-case-preset-setting-unavailable": (
        "A recipe setting is read-only, unavailable, or changes how a model loads. "
        "Remove it in Manage recipes. " + _CHOOSE
    ),
    "workflow-use-case-preset-settings-invalid": (
        "A recipe setting has an invalid value. Correct it in Manage recipes, or "
        "select Automatic (no recipe) under Recipes for this chat."
    ),
    "workflow-use-case-preset-invalid": (
        "The saved recipe cannot be read. Recreate it in Manage recipes, or select "
        "Automatic (no recipe) under Recipes for this chat."
    ),
    "workflow-use-case-preset-mismatch": (
        "The accepted recipe is for a different use case. Start a new request with "
        "a matching recipe or Automatic (no recipe)."
    ),
    "workflow-use-case-preset-revision-mismatch": (
        "The recipe was checked against a different workflow revision. Refresh the "
        "workflow choice and send a new request."
    ),
    "workflow-use-case-preset-schema-invalid": (
        "The workflow's setting controls cannot be read. Repair its input schema or "
        "choose another workflow before applying this recipe."
    ),
    "workflow-use-case-preset-prompt-setting-forbidden": (
        "Recipes cannot contain prompt settings. Remove those settings in Manage "
        "recipes and enter the request in chat."
    ),
    "workflow-use-case-operation-mismatch": (
        "The selected workflow cannot run this recipe's request type. " + _CHOOSE
    ),
    "workflow-use-case-mask-unsupported": (
        "This recipe needs a workflow with a declared mask input. Choose a workflow "
        "that supports masked inpainting."
    ),
    "workflow-use-case-outpaint-unsupported": (
        "This recipe needs a workflow with declared outpainting controls. Choose a "
        "workflow that supports extending a picture."
    ),
    "workflow-use-case-upscale-unsupported": (
        "This recipe needs a workflow with a declared scale control. Choose a "
        "workflow that supports enlarging a picture."
    ),
    "workflow-use-case-graph-missing": (
        "The workflow has no executable graph for this recipe. Restore its graph "
        "or choose another workflow."
    ),
    "workflow-use-case-source-binding-missing": (
        "The workflow does not bind the source picture required by this recipe. "
        "Choose a workflow that accepts a source picture."
    ),
    "workflow-use-case-mask-binding-missing": (
        "The workflow does not bind the mask required by this recipe. Choose a "
        "workflow that accepts a mask."
    ),
    "workflow-use-case-outpaint-binding-missing": (
        "The workflow does not have a supported source-padding connection. Choose "
        "a workflow built for outpainting."
    ),
    "workflow-use-case-upscale-binding-missing": (
        "The workflow does not have a supported enlargement step. Choose an "
        "upscaling or resampling workflow."
    ),
    "workflow-use-case-chat-not-found": (
        "This chat no longer exists. Open an existing chat before choosing a recipe."
    ),
    "workflow-use-case-project-not-found": (
        "This project no longer exists. Refresh the chat's project before choosing a recipe."
    ),
    "workflow-use-case-project-mismatch": (
        "The recipe scope does not match this chat's project. Refresh the chat "
        "and choose its recipe again."
    ),
    "workflow-use-case-input-invalid": (
        "These inputs do not match the request type. Check the source picture and "
        "selected editing operation, then send again."
    ),
    "workflow-use-case-input-ambiguous": (
        "This request combines more than one specialized image operation. Send "
        "masked editing, outpainting, and upscaling as separate requests."
    ),
    "workflow-use-case-source-required": (
        "This request needs a source picture. Attach one before sending."
    ),
}


def workflow_use_case_error(exc: ValueError) -> tuple[str, str] | None:
    """Recognize typed workflow failures while leaving unrelated errors unchanged."""
    if isinstance(exc, WorkflowFamilySelectionError):
        code = exc.reason
        if code == "no_ready_workflow":
            reasons = {reason for reason in exc.candidate_reasons if reason in _MESSAGES}
            if len(reasons) == 1:
                code = reasons.pop()
            elif reasons:
                if "workflow-instruction-edit-required" in reasons:
                    aggregate_message = (
                        "No ready instruction-edit workflow supports the recipe's inputs and "
                        "settings. Enable one in Workflows, or adjust the recipe under "
                        "Recipes for this chat."
                    )
                else:
                    aggregate_message = (
                        "No ready workflow supports the recipe's inputs and settings. " + _CHOOSE
                    )
                return ("workflow-use-case-no-compatible-workflow", aggregate_message)
    elif isinstance(
        exc,
        (
            WorkflowUseCaseError,
            WorkflowUseCasePresetAdmissionError,
            WorkflowUseCasePresetResolutionError,
            WorkflowUseCasePresetSettingsError,
        ),
    ):
        code = exc.code
    else:
        return None
    message = _MESSAGES.get(code)
    return (code, message) if message is not None else None
