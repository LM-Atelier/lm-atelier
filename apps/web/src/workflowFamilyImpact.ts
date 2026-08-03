import type { WorkflowFamilyRemovalImpact } from "./types";

/** What survives archiving, in the order a person would worry about it.
 *
 * Archiving is not deletion here: immutable revisions, exact project pins,
 * queued steps, run history, and shared model files all outlive it. Listing
 * what is kept is the honest way to ask, because the word "archive" invites
 * people to assume the opposite.
 */
export function survivingWork(impact: WorkflowFamilyRemovalImpact): string[] {
  const kept: string[] = [];
  if (impact.queued_step_count > 0) {
    kept.push(
      `${impact.queued_step_count} queued step${impact.queued_step_count === 1 ? "" : "s"} still runs`,
    );
  }
  if (impact.active_run_count > 0) {
    kept.push(
      `${impact.active_run_count} run${impact.active_run_count === 1 ? "" : "s"} in progress finishes`,
    );
  }
  if (impact.historical_run_count > 0) {
    kept.push(`${impact.historical_run_count} past runs stay in their chats`);
  }
  if (impact.project_revision_pin_count > 0) {
    kept.push(
      `${impact.project_revision_pin_count} project${impact.project_revision_pin_count === 1 ? "" : "s"} pinned to an exact revision keeps working`,
    );
  }
  if (impact.revision_count > 0) {
    kept.push(`${impact.revision_count} saved revisions are kept`);
  }
  return kept;
}

/** What the user has to deal with afterwards, which is the part worth reading. */
export function needsAttention(impact: WorkflowFamilyRemovalImpact): string[] {
  const consequences: string[] = [];
  const following = impact.chat_selection_count + impact.project_selection_count;
  if (following > 0) {
    consequences.push(
      `${following} chat${following === 1 ? "" : "s"} and projects choosing it fall back to automatic`,
    );
  }
  if (impact.default_for.length > 0) {
    consequences.push(`It is the default for ${impact.default_for.join(", ")}`);
  }
  const exclusive = impact.dependencies.filter((dependency) => !dependency.shared);
  if (exclusive.length > 0) {
    // Shared files are the reassuring case and are deliberately not listed:
    // nothing is being deleted, so what matters is which downloads nothing
    // else is using afterwards.
    consequences.push(
      `${exclusive.length} model file${exclusive.length === 1 ? "" : "s"} would then be used by nothing else`,
    );
  }
  return consequences;
}
