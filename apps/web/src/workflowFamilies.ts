import type { WorkflowFamily, WorkflowSelectorCapability } from "./types";

/** Whether this family can serve the capability being chosen for. */
export function servesCapability(
  family: WorkflowFamily,
  capability: WorkflowSelectorCapability,
): boolean {
  return family.preferences.some(
    (preference) => preference.selector_capability === capability && preference.enabled,
  );
}

/** Families in the order the server asked for, then by name.
 *
 * Sort order is a property of the preference rather than of the family,
 * because the same family can sit anywhere in two different capabilities'
 * lists.
 */
export function orderFamilies(
  families: WorkflowFamily[],
  capability: WorkflowSelectorCapability,
): WorkflowFamily[] {
  const rank = (family: WorkflowFamily) =>
    family.preferences.find((preference) => preference.selector_capability === capability)
      ?.sort_order ?? Number.MAX_SAFE_INTEGER;
  return [...families].sort(
    (left, right) => rank(left) - rank(right) || left.name.localeCompare(right.name),
  );
}

