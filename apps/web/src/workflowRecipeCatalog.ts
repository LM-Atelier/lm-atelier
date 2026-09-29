import { api } from "./api";
import type { WorkflowUseCasePreset } from "./workflowUseCaseTypes";

export async function workflowRecipeCatalog(signal: AbortSignal): Promise<WorkflowUseCasePreset[]> {
  const recipes: WorkflowUseCasePreset[] = [];
  const seen = new Set<string>();
  for (let offset = 0; ; offset += 200) {
    signal.throwIfAborted();
    const page = await api.workflowUseCasePresets(undefined, offset, signal);
    for (const recipe of page) {
      if (seen.has(recipe.id)) throw new Error("The recipe list changed while loading. Try again.");
      seen.add(recipe.id);
      recipes.push(recipe);
    }
    if (page.length < 200) return recipes;
  }
}
