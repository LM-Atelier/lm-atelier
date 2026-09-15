import type {
  WorkflowActivationChoice,
  WorkflowActivationPreparation,
  WorkflowActivationSelection,
  WorkflowActivationSlotChoices,
} from "./types";

export function activationChoiceKey(selection: WorkflowActivationSelection): string {
  return JSON.stringify([
    selection.slot_name, selection.requirement_key, selection.local_kind,
    selection.local_id, selection.recorded_resource_identity_sha256, selection.mount,
  ]);
}

export function activationChoiceGroups(slot: WorkflowActivationSlotChoices) {
  const label = slot.name.replaceAll("_", " ");
  if (slot.satisfaction === "any_of") {
    return [{ key: JSON.stringify([slot.name, null]), label, choices: slot.choices }];
  }
  return slot.requirement_keys.map(requirement => ({
    key: JSON.stringify([slot.name, requirement]),
    label: slot.requirement_keys.length === 1 ? label : label + " / " + requirement.replaceAll("_", " "),
    choices: slot.choices.filter(choice => choice.selection.requirement_key === requirement),
  }));
}

export function initialActivationChoices(preparation: WorkflowActivationPreparation): Record<string, string> {
  return Object.fromEntries(preparation.slots.flatMap(slot => activationChoiceGroups(slot).map(group => [
    group.key, group.choices.length === 1 ? activationChoiceKey(group.choices[0].selection) : "",
  ])));
}

export function selectedActivationDependencies(
  preparation: WorkflowActivationPreparation,
  choices: Record<string, string>,
  optional: Record<string, boolean>,
): WorkflowActivationSelection[] | null {
  const selected: WorkflowActivationSelection[] = [];
  for (const slot of preparation.slots) {
    if (!slot.required && !optional[slot.name]) continue;
    const groups = activationChoiceGroups(slot);
    if (groups.length === 0) return null;
    for (const group of groups) {
      const choice = group.choices.find(item => activationChoiceKey(item.selection) === choices[group.key]);
      if (!choice || choice.selection.slot_name !== slot.name
          || choice.selection.local_kind !== slot.resource_kind
          || !/^[a-f0-9]{64}$/.test(choice.selection.recorded_resource_identity_sha256 ?? "")) return null;
      selected.push(choice.selection);
    }
  }
  return selected;
}

export function freshActivationSelections(
  selections: WorkflowActivationSelection[],
  fresh: WorkflowActivationPreparation,
): WorkflowActivationSelection[] | null {
  const available: WorkflowActivationChoice[] = fresh.slots.flatMap(slot => slot.choices);
  const result: WorkflowActivationSelection[] = [];
  for (const selection of selections) {
    const match = available.find(choice => activationChoiceKey(choice.selection) === activationChoiceKey(selection));
    if (!match) return null;
    result.push(match.selection);
  }
  return result;
}
