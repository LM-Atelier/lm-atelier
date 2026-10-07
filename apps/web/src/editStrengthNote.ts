/**
 * One short line saying plainly that an edit may not show, and why.
 *
 * A strength edit encodes the picture and redraws it, and how much changes is
 * the strength. When that strength was chosen automatically for a specific
 * change, a small request can come back looking untouched, and nothing on the
 * result says so. Auto prefers a workflow that reads the words beside the
 * picture for such requests, so this appears when none was ready or when a
 * redraw workflow was chosen. It names the control that lets the person decide.
 *
 * A change to the whole picture is what a redraw does well and gets no line,
 * and neither does a result the edit review already found the change in. An
 * edit that reads the words records no strength at all, so it never gets one.
 */

export const EDIT_STRENGTH_NOTE =
  "Redrawn at an automatic strength, so a small change may not show · set Change strength to Manual to change more";

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function editStrengthNote(
  provenance: Record<string, unknown> | undefined,
): string | null {
  const strength = record(record(provenance?.image_edit)?.strength);
  if (strength?.mode !== "auto" || strength.scope === "global") return null;
  const review = record(provenance?.image_edit_verification);
  const found =
    review?.status === "complete" &&
    review.reason !== "no_measurable_change" &&
    record(review.assessment)?.requested_change_visible === true;
  return found ? null : EDIT_STRENGTH_NOTE;
}
