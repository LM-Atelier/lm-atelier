import type {
  RoutingMode,
  WorkflowFamilyVariant,
  WorkflowSelection,
} from "./types";

export type ComposerWorkflowCapability = "chat" | "image" | "video";

export type WorkflowChoiceKind =
  | "default"
  | "automatic"
  | "explicit"
  | "compatibility";

/** The workflow capability that an explicit composer mode will ask for.
 *
 * Auto deliberately has no client-side answer. The server classifies the
 * request before selecting a workflow, so guessing a capability here would
 * make the control disagree with the workflow that eventually runs.
 */
export function activeWorkflowCapability(
  mode: RoutingMode,
): ComposerWorkflowCapability | null {
  if (mode === "text") return "chat";
  if (mode === "image") return "image";
  if (mode === "video") return "video";
  return null;
}

/** Reduce persisted compatibility modes to the four choices the composer can
 * present without silently replacing an existing setup. */
export function workflowChoiceKind(
  selection: WorkflowSelection | null | undefined,
): WorkflowChoiceKind {
  if (selection?.mode === "automatic") return "automatic";
  if (selection?.mode === "family") return "explicit";
  if (selection?.mode === "legacy" || selection?.mode === "revision") {
    return "compatibility";
  }
  return "default";
}

/** Whether one family variant can answer the capability visible in the
 * composer. Operation is the durable fallback for older revisions whose
 * capability list predates workflow-family selection. */
export function variantServesComposerCapability(
  variant: WorkflowFamilyVariant,
  capability: ComposerWorkflowCapability,
): boolean {
  if (variant.capabilities.includes(capability)) return true;
  if (capability === "chat") return variant.operation === "text";
  if (capability === "image") {
    return variant.operation === "text_to_image" || variant.operation === "image_to_image";
  }
  return variant.operation === "text_to_video" || variant.operation === "image_to_video";
}
