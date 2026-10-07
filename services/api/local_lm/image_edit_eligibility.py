"""Require instruction wiring when Auto selects a localized image edit."""

from __future__ import annotations

from .domain import Operation
from .image_edit_kind import image_edit_kind
from .image_edit_strength import EditScope, estimate_image_edit_strength
from .models import WorkflowRevision
from .workflow_selection import RevisionEligibility


def automatic_image_edit_eligibility(
    operation: Operation,
    prompt: str,
    existing: RevisionEligibility | None = None,
) -> RevisionEligibility | None:
    """Keep prior restrictions and require source-conditioned instruction wiring.

    Recognized localized edits need structural instruction evidence even if a
    model declares that ability; an unknown provider declaration does not veto
    that evidence. Global or ambiguous requests keep their existing eligibility.
    """

    if operation != Operation.IMAGE_TO_IMAGE:
        return existing
    if estimate_image_edit_strength(prompt).scope not in {
        EditScope.MINIMAL,
        EditScope.LOCALIZED,
        EditScope.REPLACEMENT,
    }:
        return existing

    def eligible(revision: WorkflowRevision | None) -> str | None:
        if existing is not None:
            reason = existing(revision)
            if reason is not None:
                return reason
        if (
            revision is not None
            and image_edit_kind(
                operation.value,
                revision.ui_graph_json,
                revision.api_graph_json,
                revision.input_schema_json,
            )
            == "instruction"
        ):
            return None
        return "workflow-instruction-edit-required"

    return eligible
