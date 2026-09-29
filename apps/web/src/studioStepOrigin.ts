/** Where a Studio result came from, when the history strip alone would not say.
 *
 * The strip lists results in the order they were made, so a result made from
 * the step before it needs nothing said. One made from an earlier step, after
 * the reader went back and applied from there, starts a branch: listed last, it
 * would read as the next change to the latest picture unless it names the
 * picture it was made from.
 */

import type { StudioStep } from "./useStudioSession";

/** "From the original", "From step N", or nothing when the step before it is where it came from. */
export function studioStepOrigin(steps: readonly StudioStep[], index: number): string | null {
  const made = steps[index]?.beforeArtifactId;
  if (!made) return null;
  // The nearest earlier picture it was made from; a picture can appear twice
  // when an edit gave back exactly what it was given.
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    if (steps[cursor].artifactId !== made) continue;
    if (cursor === index - 1) return null;
    return steps[cursor].isSource ? "From the original" : `From step ${cursor}`;
  }
  return null;
}
