import type { Workflow, WorkflowFamily, WorkflowSelection } from "./types";

export type TurnOperation =
  | "text_to_image"
  | "image_to_image"
  | "text_to_video"
  | "image_to_video";

export function operationForTurn(
  mode: "image" | "video",
  hasAttachments: boolean,
): TurnOperation {
  if (mode === "image") return hasAttachments ? "image_to_image" : "text_to_image";
  return hasAttachments ? "image_to_video" : "text_to_video";
}

/** The revision a turn will actually run, in the order the server resolves it.
 *
 * The settings panel and the executor have to agree about this. They did
 * not: the panel read a project's legacy pinned revision while a family
 * selection took precedence at run time, so the controls on screen could
 * belong to one workflow while a different one produced the picture. That
 * is worse than showing nothing, because the settings look authoritative.
 *
 * So the order here is the resolution order, not a convenience: an explicit
 * revision pin, then a chosen family's variant for this exact operation,
 * then the legacy project field, then whatever the workspace offers.
 */
export function revisionForTurn(
  workflows: Workflow[],
  families: WorkflowFamily[],
  selection: WorkflowSelection | undefined,
  legacyRevisionId: string | null | undefined,
  operation: TurnOperation,
): string | undefined {
  if (selection?.mode === "revision" && selection.workflow_revision_id) {
    return selection.workflow_revision_id;
  }
  if (selection?.mode === "family" && selection.workflow_family_id) {
    const family = families.find((one) => one.id === selection.workflow_family_id);
    const variant = family?.variants.find((one) => one.operation === operation);
    // A family with no variant for this operation cannot answer this turn,
    // so fall through rather than pinning something that does not fit.
    if (variant?.current_revision_id) return variant.current_revision_id;
  }
  if (selection?.mode === "automatic") return undefined;
  return legacyRevisionId ?? undefined;
}

/** The settings schema for whatever that revision turns out to be. */
export function schemaForRevision(
  workflows: Workflow[],
  revisionId: string | undefined,
  operation: TurnOperation,
): Record<string, unknown> | undefined {
  if (revisionId) {
    for (const workflow of workflows) {
      const revision = workflow.revisions.find((one) => one.id === revisionId);
      if (revision && workflow.operation === operation) return revision.input_schema_json;
    }
  }
  const workflow = workflows.find((one) => one.operation === operation);
  return workflow?.revisions.find((one) => one.id === workflow.current_revision_id)
    ?.input_schema_json;
}
