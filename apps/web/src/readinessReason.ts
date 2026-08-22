import type { WorkflowFamilyVariant } from "./types";

/** What each readiness reason means to the person reading it.
 *
 * `readiness_reason` is a stable machine slug - `model_unavailable`,
 * `activation_not_ready` - and all three workflow selectors were rendering it
 * straight into the page. The fallback sentences beside it were good and
 * unreachable, because the server sets a reason on every non-ready path, so the
 * `??` never fired and somebody blocked from running a workflow read
 * `model_unavailable` in a `role="status"` region.
 *
 * Slugs are the right thing to send. A stable code survives rewording and can
 * be matched on; a sentence cannot. The translation belongs here.
 */
const REASON_TEXT: Record<string, string> = {
  engine_mismatch: "This workflow was built for a different engine.",
  revision_not_executable: "This version has no runnable graph yet.",
  revision_untrusted: "Needs review before it can run.",
  activation_not_ready: "Its files are not prepared yet.",
  model_unavailable: "The model it needs is not installed.",
  operation_unavailable: "It cannot do this kind of work.",
  current_revision_missing: "It has no current version.",
  family_archived: "This workflow is archived.",
  family_disabled: "This workflow is turned off.",
  dependency_contract_drift: "Its recorded dependencies no longer match itself.",
};

/** The sentence to show, or `null` when there is nothing to say.
 *
 * An unrecognised slug falls back to wording chosen by readiness rather than
 * being rendered raw. A reason the browser has never heard of is still a real
 * refusal, and the person needs to know the shape of it even when this build
 * cannot name the cause.
 */
export function readinessReason(variant: WorkflowFamilyVariant): string | null {
  if (variant.readiness === "ready") return null;
  const known = variant.readiness_reason ? REASON_TEXT[variant.readiness_reason] : undefined;
  if (known) return known;
  if (variant.readiness === "setup_required") {
    return "Needs files or nodes installed before it can run.";
  }
  if (variant.readiness === "review_required") {
    // Not a machine problem, and saying so sent people to change settings that
    // were never wrong. What it needs is somebody to look at it and say so.
    return "Needs review before it can run.";
  }
  return "Cannot run on this machine as configured.";
}

/** The slugs this build can name, for the contract test to difference. */
export const KNOWN_READINESS_REASONS = Object.keys(REASON_TEXT);
