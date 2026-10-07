import { useQueries } from "@tanstack/react-query";
import { api } from "./api";
import type { RoutingMode, WorkflowFamily, WorkflowSelection } from "./types";
import { operationForTurn, type TurnOperation } from "./turnWorkflow";

export interface WorkflowFamilyTarget {
  capability: "image" | "video";
  operation: TurnOperation;
  familyId: string | null;
}

export function workflowSelectionForCapability(
  selections: WorkflowSelection[] | undefined, capability: RoutingMode,
) {
  return selections === undefined ? undefined
    : selections.find(selection => selection.selector_capability === capability) ?? null;
}

export function workflowFamilyTarget(
  mode: RoutingMode, hasAttachments: boolean,
  selection: WorkflowSelection | null | undefined,
  projectSelection: WorkflowSelection | null | undefined,
): WorkflowFamilyTarget | null {
  if ((mode !== "image" && mode !== "video") || selection === undefined) return null;
  const chosen = !selection || (selection.mode === "default" || selection.mode === "inherit") ? projectSelection : selection;
  if (chosen === undefined || (chosen && chosen.mode !== "family" && chosen.mode !== "default" && chosen.mode !== "inherit")) return null;
  if (chosen?.mode === "family" && !chosen.workflow_family_id) return null;
  return { capability: mode, operation: operationForTurn(mode, hasAttachments),
    familyId: chosen?.mode === "family" ? chosen.workflow_family_id : null };
}

/** Read enough ready variants to distinguish a unique revision from an ambiguous family. */
export function useWorkflowResolutionFamilies(targets: (WorkflowFamilyTarget | null)[]) {
  const distinct = [...new Map(targets.filter(target => target !== null)
    .map(target => [JSON.stringify(target), target])).values()];
  const results = useQueries({ queries: distinct.map(target => ({
    queryKey: ["workflow-families", "resolution", target],
    queryFn: ({ signal }: { signal: AbortSignal }) => api.workflowFamilies(target.capability, false, false, {
      limit: 1, variantLimit: 2, operation: target.operation, readiness: "ready",
      ...(target.familyId ? { familyIds: [target.familyId] } : { defaultsOnly: true, enabledOnly: true }),
    }, signal),
  })) });
  const merged = new Map<string, WorkflowFamily>();
  results.forEach((result, index) => {
    if (!result.isSuccess) return;
    const target = distinct[index];
    const family = result.data.find(row => target.familyId ? row.id === target.familyId
      : row.preferences.some(preference => preference.selector_capability === target.capability
        && preference.enabled && preference.is_default));
    if (!family || family.archived || !family.enabled) return;
    const previous = merged.get(family.id);
    const variants = family.variants.filter(variant => variant.operation === target.operation
      && variant.readiness === "ready");
    merged.set(family.id, { ...family, variants: [...new Map([
      ...(previous?.variants ?? []), ...variants,
    ].map(variant => [variant.id, variant])).values()] });
  });
  return { families: [...merged.values()], error: results.find(result => result.error)?.error ?? null,
    isPending: results.some(result => result.isPending),
    refetch: () => Promise.all(results.map(result => result.refetch())) };
}
