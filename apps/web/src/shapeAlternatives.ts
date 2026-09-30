import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { operationForTurn, revisionForTurn, type TurnOperation } from "./turnWorkflow";
import { orderFamilies, servesCapability } from "./workflowFamilies";
import type { WorkflowFamily } from "./types";

/** Another installed workflow that could answer this turn, with the revision it would run. */
export interface ShapeAlternative {
  familyId: string;
  name: string;
  revisionId: string;
}

/** What the shape row needs to offer the workflows that take a shape. */
export interface ShapeAlternatives {
  candidates: ShapeAlternative[];
  onChoose: (familyId: string) => void;
  choosing: boolean;
  error: string | null;
}

/** The kinds of request whose output size a workflow can prove it takes.
 * An edit's shape comes from its source picture, so an edit is not among them. */
const SHAPED: ReadonlySet<TurnOperation> = new Set(["text_to_image", "text_to_video", "image_to_video"]);

/** The families other than the one in use that could answer this turn, in the
 * picker's order, each with the one ready revision it would run.
 *
 * A family counts only where choosing it settles the revision before anything
 * is sent, the same test the composer applies to the family already chosen,
 * so what is offered here is exactly what choosing it would run.
 */
export function shapeAlternativeCandidates(
  families: WorkflowFamily[],
  capability: "image" | "video",
  hasAttachments: boolean,
  currentRevisionId: string | null,
): ShapeAlternative[] {
  const operation = operationForTurn(capability, hasAttachments);
  if (!SHAPED.has(operation)) return [];
  return orderFamilies(families, capability).flatMap((family) => {
    if (!family.enabled || family.archived || family.compatibility || !servesCapability(family, capability)) {
      return [];
    }
    const revisionId = revisionForTurn(families, capability, {
      selector_capability: capability,
      mode: "family",
      workflow_family_id: family.id,
      workflow_revision_id: null,
      legacy_profile_id: null,
    }, null, operation);
    return revisionId && revisionId !== currentRevisionId
      ? [{ familyId: family.id, name: family.name, revisionId }]
      : [];
  });
}

/** Offering this chat another workflow for one kind of request, chosen as the workflow picker chooses one.
 *
 * Undefined where the settings do not describe the chat's own choice: a
 * version being edited on its own, or a turn whose workflow is fixed by its
 * caller, has no chat choice to change.
 */
export function useShapeAlternatives({
  chatId,
  capability,
  hasAttachments,
  families,
  currentRevisionId,
  enabled,
}: {
  chatId: string | null;
  capability: "image" | "video" | null;
  hasAttachments: boolean;
  families: WorkflowFamily[];
  currentRevisionId: string | null;
  enabled: boolean;
}): ShapeAlternatives | undefined {
  const client = useQueryClient();
  const choose = useMutation({
    mutationFn: ({ chat, kind, familyId }: { chat: string; kind: "image" | "video"; familyId: string }) =>
      api.setChatWorkflowSelection(chat, kind, { mode: "family", workflow_family_id: familyId }),
    onSuccess: (_selection, { chat }) =>
      void client.invalidateQueries({ queryKey: ["chat", chat, "workflow-selections"] }),
  });
  if (!enabled || !chatId || !capability) return undefined;
  return {
    candidates: shapeAlternativeCandidates(families, capability, hasAttachments, currentRevisionId),
    onChoose: (familyId) => choose.mutate({ chat: chatId, kind: capability, familyId }),
    choosing: choose.isPending,
    error: choose.error ? (choose.error as Error).message : null,
  };
}
