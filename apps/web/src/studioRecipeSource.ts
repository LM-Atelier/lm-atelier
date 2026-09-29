import type { ChatDetail } from "./types";
import type { StudioStep } from "./useStudioSession";

/** What a recipe saved from a Studio result is made of: the run that made it, and the words it was given. */
export type StudioRecipeSource = { runId: string; instruction: string };

/** The recipe a Studio result can be saved as, when a model made it.
 *
 * A model's result carries its run beside the record of what that run did, and
 * the server reads the recipe from the run rather than from whatever is set up
 * now. An exact edit made without a model records no run, and the original
 * picture was made by nothing here, so neither offers a recipe.
 */
export function studioRecipeSource(
  session: ChatDetail | null | undefined,
  step: StudioStep | null,
): StudioRecipeSource | null {
  if (!step || step.isSource) return null;
  const answer = session?.messages.find((message) => message.id === step.messageId);
  const runId = answer?.parts.find((part) => part.type === "generation_metadata")?.metadata_json.run_id;
  if (typeof runId !== "string" || !runId) return null;
  // A recipe is its words as much as its run; a result recorded without them has none to keep.
  const instruction: unknown = step.instruction;
  return typeof instruction === "string" && instruction.trim() ? { runId, instruction } : null;
}
