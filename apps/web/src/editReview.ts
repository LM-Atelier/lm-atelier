/**
 * One short line saying what "Review image edits" did to a result.
 *
 * The check already runs, already retries once, and already records why it did
 * or did not - all of it in run provenance that nothing has ever displayed. So a
 * person who turned the setting on sees a second image appear with no
 * explanation, and a person whose check found nothing sees no difference from a
 * person who never enabled it. This turns that record into one span.
 *
 * Two provenance keys are read, because a retried edit produces two results and
 * the person is usually looking at the second one: the source result carries
 * `image_edit_verification`, and the retry it started carries
 * `image_edit_verification_retry`.
 */

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/**
 * "at a higher strength", "at a lower strength", or nothing when the numbers
 * are absent, unusable or equal. The exact values are deliberately not shown:
 * the strength was chosen automatically, so its number is not one the person
 * set or can act on.
 */
function strengthChange(before: unknown, after: unknown): string {
  const from = Number(before);
  const to = Number(after);
  if (!Number.isFinite(from) || !Number.isFinite(to) || from === to) return "";
  return to > from ? " at a higher strength" : " at a lower strength";
}

/**
 * Skips that are not worth a line. `not_image_edit` and `disabled` are the
 * ordinary state of almost every message, and `eligible` means the check was
 * about to run rather than that it stopped.
 */
const SILENT_SKIPS = new Set(["not_image_edit", "disabled", "eligible", "cancelled"]);

export function editReviewSummary(
  provenance: Record<string, unknown> | undefined,
): string | null {
  const retry = record(provenance?.image_edit_verification_retry);
  if (retry) {
    return `Automatic retry after edit review${strengthChange(
      retry.strength_before,
      retry.strength_after,
    )}`;
  }

  const review = record(provenance?.image_edit_verification);
  if (!review) return null;

  if (review.status === "skipped") {
    const reason = review.reason;
    return typeof reason === "string" && SILENT_SKIPS.has(reason)
      ? null
      : "Edit review did not run";
  }
  if (review.status !== "complete") return null;

  if (review.automatic_retry_executed === true) {
    const adjustment = record(review.strength_adjustment);
    return `Edit review retried this once${strengthChange(
      adjustment?.before,
      adjustment?.after,
    )}`;
  }

  switch (review.retry_execution_reason) {
    case "manual_strength_preserved":
      return "Edit review suggested another attempt · your strength setting was kept";
    case "unavailable":
      return "Edit review suggested another attempt · it could not be started";
    case "bound_not_started":
      return "Edit review suggested another attempt · it has not started yet";
  }

  const assessment = record(review.assessment);
  if (!assessment) return null;
  if (assessment.requested_change_visible === false) {
    return "Edit review did not find the change you asked for";
  }
  if (assessment.unrelated_content_preserved === false) {
    return "Edit review found changes beyond the one you asked for";
  }
  if (assessment.requested_change_visible === true) return "Edit review found your change";
  return null;
}
