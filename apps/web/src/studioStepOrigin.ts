/** Where a Studio result came from, when the history strip alone would not say.
 *
 * The strip lists results in the order they were made, so a result made from
 * the step before it needs nothing said. One made from an earlier step, after
 * the reader went back and applied from there, starts a branch: listed last, it
 * would read as the next change to the latest picture unless it names the
 * picture it was made from.
 */

import type { StudioStep } from "./useStudioSession";

/** The step a result was made from: the nearest earlier appearance of the picture it changed.
 *
 * A picture can appear twice when an edit gave back exactly what it was given,
 * and the nearest one is the one it was made from. Nothing when that picture
 * is not in the strip.
 */
export function studioStepParent(steps: readonly StudioStep[], index: number): number | null {
  const made = steps[index]?.beforeArtifactId;
  if (!made) return null;
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    if (steps[cursor].artifactId === made) return cursor;
  }
  return null;
}

/** "From the original", "From step N", or nothing when the step before it is where it came from. */
export function studioStepOrigin(steps: readonly StudioStep[], index: number): string | null {
  const parent = studioStepParent(steps, index);
  if (parent === null || parent === index - 1) return null;
  return steps[parent].isSource ? "From the original" : `From step ${parent}`;
}

/** Every step a result was made from, back to the original: the picture it changed, what that was made from, and so on. */
export function studioStepAncestors(steps: readonly StudioStep[], index: number): Set<number> {
  const ancestors = new Set<number>();
  let current = studioStepParent(steps, index);
  // A result is always made from a step earlier in the strip, so the walk ends.
  while (current !== null) {
    ancestors.add(current);
    current = studioStepParent(steps, current);
  }
  return ancestors;
}
