"""Checking the checker: canary answers about two identical images."""

from __future__ import annotations

from local_lm.image_edit_verification import (
    ImageEditVerificationAssessment,
    VerificationDirection,
)
from local_lm.verifier_trust import CanaryOutcome, evaluate_canary


def _assessment(
    *,
    visible: bool,
    preserved: bool = True,
    confidence: float = 0.95,
) -> ImageEditVerificationAssessment:
    return ImageEditVerificationAssessment(
        requested_change_visible=visible,
        unrelated_content_preserved=preserved,
        retry_recommended=False,
        direction=VerificationDirection.NONE,
        confidence=confidence,
    )


def test_a_verifier_that_sees_a_change_in_identical_images_is_not_trusted() -> None:
    """The field failure: an unchanged image reported as an edit at 0.95."""
    result = evaluate_canary(_assessment(visible=True))

    assert result.outcome == CanaryOutcome.FAILED
    assert not result.verifier_is_trustworthy
    assert "identical images" in result.detail


def test_high_confidence_does_not_rescue_a_failed_canary() -> None:
    """Confidence is the verifier's own claim; the canary is the evidence."""
    assert not evaluate_canary(_assessment(visible=True, confidence=1.0)).verifier_is_trustworthy


def test_claiming_unrelated_content_changed_also_fails() -> None:
    result = evaluate_canary(_assessment(visible=False, preserved=False))

    assert result.outcome == CanaryOutcome.FAILED
    assert not result.verifier_is_trustworthy


def test_a_correct_answer_passes_without_granting_extra_confidence() -> None:
    result = evaluate_canary(_assessment(visible=False))

    assert result.outcome == CanaryOutcome.PASSED
    assert result.verifier_is_trustworthy


def test_an_unavailable_canary_is_not_read_as_a_failure() -> None:
    """No vision model or an interrupted check must not punish the user."""
    result = evaluate_canary(None)

    assert result.outcome == CanaryOutcome.UNAVAILABLE
    assert result.verifier_is_trustworthy
    assert "used unchanged" in result.detail
