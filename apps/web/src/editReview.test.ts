import { describe, expect, it } from "vitest";
import { editReviewSummary } from "./editReview";

const assessment = {
  requested_change_visible: true,
  unrelated_content_preserved: true,
  retry_recommended: false,
  direction: "none",
  confidence: 0.9,
};

describe("editReviewSummary", () => {
  it("explains the second image on the retry that produced it", () => {
    expect(editReviewSummary({
      image_edit_verification_retry: {
        source_run_id: "run_1",
        strength_before: 0.35,
        strength_after: 0.47,
      },
    })).toBe("Automatic retry after edit review at a higher strength");
  });

  it("names the retry without a direction when the strengths are unusable", () => {
    expect(editReviewSummary({
      image_edit_verification_retry: { source_run_id: "run_1" },
    })).toBe("Automatic retry after edit review");
  });

  it("says the source result was retried, and which way the strength moved", () => {
    expect(editReviewSummary({
      image_edit_verification: {
        status: "complete",
        automatic_retry_executed: true,
        assessment: { ...assessment, requested_change_visible: false },
        strength_adjustment: { parameter: "denoise", before: 0.6, after: 0.48 },
      },
    })).toBe("Edit review retried this once at a lower strength");
  });

  it("distinguishes the three reasons a suggested retry did not run", () => {
    const base = { status: "complete", automatic_retry_executed: false, assessment };
    expect(editReviewSummary({
      image_edit_verification: { ...base, retry_execution_reason: "manual_strength_preserved" },
    })).toBe("Edit review suggested another attempt · your strength setting was kept");
    expect(editReviewSummary({
      image_edit_verification: { ...base, retry_execution_reason: "unavailable" },
    })).toBe("Edit review suggested another attempt · it could not be started");
    expect(editReviewSummary({
      image_edit_verification: { ...base, retry_execution_reason: "bound_not_started" },
    })).toBe("Edit review suggested another attempt · it has not started yet");
  });

  it("confirms a check that ran and found the change", () => {
    expect(editReviewSummary({
      image_edit_verification: { status: "complete", automatic_retry_executed: false, assessment },
    })).toBe("Edit review found your change");
  });

  it("reports a missing change and collateral change separately", () => {
    expect(editReviewSummary({
      image_edit_verification: {
        status: "complete",
        automatic_retry_executed: false,
        assessment: { ...assessment, requested_change_visible: false },
      },
    })).toBe("Edit review did not find the change you asked for");
    expect(editReviewSummary({
      image_edit_verification: {
        status: "complete",
        automatic_retry_executed: false,
        assessment: { ...assessment, unrelated_content_preserved: false },
      },
    })).toBe("Edit review found changes beyond the one you asked for");
  });

  it("stays silent for the skips that describe an ordinary message", () => {
    for (const reason of ["not_image_edit", "disabled", "eligible", "cancelled"]) {
      expect(editReviewSummary({
        image_edit_verification: { status: "skipped", reason, automatic_retry_executed: false },
      })).toBeNull();
    }
  });

  it("says a check the user enabled did not run", () => {
    expect(editReviewSummary({
      image_edit_verification: {
        status: "skipped",
        reason: "vision_profile_unavailable",
        automatic_retry_executed: false,
      },
    })).toBe("Edit review did not run");
  });

  it("invents nothing from an absent or malformed record", () => {
    expect(editReviewSummary(undefined)).toBeNull();
    expect(editReviewSummary({})).toBeNull();
    expect(editReviewSummary({ image_edit_verification: "complete" })).toBeNull();
    expect(editReviewSummary({ image_edit_verification: [] })).toBeNull();
    expect(editReviewSummary({
      image_edit_verification: { status: "running", automatic_retry_executed: false },
    })).toBeNull();
    expect(editReviewSummary({
      image_edit_verification: { status: "complete", automatic_retry_executed: false },
    })).toBeNull();
  });
});
