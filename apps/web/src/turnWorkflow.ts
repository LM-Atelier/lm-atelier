import type {
  Workflow,
  WorkflowFamily,
  WorkflowSelection,
  WorkflowSelectorCapability,
} from "./types";

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

function familyRevision(
  families: WorkflowFamily[],
  familyId: string | null,
  operation: TurnOperation,
): string | null {
  if (!familyId) return null;
  const variants = families
    .find((family) => family.id === familyId)
    ?.variants.filter((variant) =>
      variant.operation === operation
      && variant.readiness === "ready"
      && Boolean(variant.current_revision_id)
    ) ?? [];
  return variants.length === 1 ? variants[0].current_revision_id : null;
}

function workspaceDefaultRevision(
  families: WorkflowFamily[],
  capability: WorkflowSelectorCapability,
  operation: TurnOperation,
): string | null {
  const family = families.find((candidate) => candidate.preferences.some(
    (preference) => preference.selector_capability === capability
      && preference.enabled
      && preference.is_default,
  ));
  return familyRevision(families, family?.id ?? null, operation);
}

/** Resolve only choices that are deterministic before the prompt is sent.
 *
 * `undefined` means a required selection response has not loaded, while
 * `null` means that scope does not exist (for example, a chat without a
 * project). Automatic and legacy-profile choices are deliberately unresolved:
 * the server needs the prompt or compatibility bridge to choose them, so
 * showing some other workflow's controls would be false authority.
 */
export function revisionForTurn(
  families: WorkflowFamily[],
  capability: "image" | "video",
  chatSelection: WorkflowSelection | null | undefined,
  projectSelection: WorkflowSelection | null | undefined,
  operation: TurnOperation,
): string | null {
  if (chatSelection === undefined) return null;
  if (chatSelection?.mode === "revision") return chatSelection.workflow_revision_id;
  if (chatSelection?.mode === "family") {
    return familyRevision(families, chatSelection.workflow_family_id, operation);
  }
  if (chatSelection?.mode === "automatic" || chatSelection?.mode === "legacy") return null;

  // A default chat follows its project. A null project selection means this
  // chat has no project and therefore reaches the workspace default directly.
  if (projectSelection === undefined) return null;
  if (projectSelection?.mode === "revision") return projectSelection.workflow_revision_id;
  if (projectSelection?.mode === "family") {
    return familyRevision(families, projectSelection.workflow_family_id, operation);
  }
  if (projectSelection?.mode === "automatic" || projectSelection?.mode === "legacy") return null;
  return workspaceDefaultRevision(families, capability, operation);
}

/** The settings schema for whatever that revision turns out to be. */
export function schemaForRevision(
  workflows: Workflow[],
  revisionId: string | null,
  operation: TurnOperation,
): Record<string, unknown> | undefined {
  if (!revisionId) return undefined;
  for (const workflow of workflows) {
    const revision = workflow.revisions.find((one) => one.id === revisionId);
    if (revision && workflow.operation === operation) return revision.input_schema_json;
  }
  return undefined;
}
