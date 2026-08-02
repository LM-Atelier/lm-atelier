"""Check the checker.

The image-edit verifier is a model answering a yes/no question about two
pictures, and a model that is not suited to the task does not fail loudly -
it answers confidently and wrongly. That is exactly what happened in the
field: an unchanged image was reported as a completed edit at 0.95
confidence, and every downstream decision inherited that confidence.

A canary is the cheapest possible falsification: hand the verifier the *same
image twice* and ask its ordinary question. Any answer other than "nothing
changed" proves it is not reading the pictures, and the result is not a
judgement call - it is a fact about that verifier, on this machine, with this
model.

Deliberately one-directional. Failing a canary is conclusive evidence the
verifier cannot be trusted; passing one proves only that it noticed identical
inputs, which is the floor rather than the standard. Passing therefore grants
no extra confidence anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .image_edit_verification import ImageEditVerificationAssessment

CANARY_PROMPT_SUBJECT = "an unchanged image"


class CanaryOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CanaryResult:
    outcome: CanaryOutcome
    detail: str

    @property
    def verifier_is_trustworthy(self) -> bool:
        """False only on a definite failure.

        An unavailable canary - no vision model, an interrupted assessment -
        must not be read as a failure: refusing to verify because a check
        could not run would punish the user for an unrelated fault.
        """

        return self.outcome != CanaryOutcome.FAILED


def evaluate_canary(assessment: ImageEditVerificationAssessment | None) -> CanaryResult:
    """Judge a verifier's answer about two identical images."""

    if assessment is None:
        return CanaryResult(
            CanaryOutcome.UNAVAILABLE,
            "The verifier could not be checked, so its answers are used unchanged.",
        )
    if assessment.requested_change_visible:
        return CanaryResult(
            CanaryOutcome.FAILED,
            "The verifier reported a visible change between two identical images, "
            "so its answers about real edits cannot be relied on.",
        )
    if not assessment.unrelated_content_preserved:
        return CanaryResult(
            CanaryOutcome.FAILED,
            "The verifier reported that unrelated content changed between two "
            "identical images, so its answers about real edits cannot be relied on.",
        )
    return CanaryResult(
        CanaryOutcome.PASSED,
        "The verifier correctly reported no change between two identical images.",
    )
