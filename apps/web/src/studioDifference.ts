/** What an edit changed, pixel by pixel, as an overlay for its result.
 *
 * Only two pictures of exactly one size are compared this way. Resampling one
 * to the other's size would mark every edge in the picture as changed, which
 * is the opposite of what the overlay is for. Pure, so the arithmetic is
 * provable without a canvas.
 */

/** A change this small or smaller counts as none: saving a picture again can leave it. */
export const NOISE_LEVELS = 4;

/** How much one pixel changed: the largest difference in any channel, its opacity included. */
function change(before: Uint8ClampedArray, after: Uint8ClampedArray, at: number): number {
  return Math.max(
    Math.abs(after[at] - before[at]),
    Math.abs(after[at + 1] - before[at + 1]),
    Math.abs(after[at + 2] - before[at + 2]),
    Math.abs(after[at + 3] - before[at + 3]),
  );
}

/** The overlay, and how many pixels changed by more than noise.
 *
 * Clear where nothing changed, so the result shows through. Where a pixel
 * changed it is tinted from amber, for the least change, to red, for the
 * most, and the more it changed the less of the result shows through it.
 */
export function differenceOverlay(
  before: Uint8ClampedArray,
  after: Uint8ClampedArray,
): { pixels: Uint8ClampedArray; changed: number } {
  const pixels = new Uint8ClampedArray(after.length);
  let changed = 0;
  for (let at = 0; at < after.length; at += 4) {
    const amount = change(before, after, at);
    if (amount <= NOISE_LEVELS) continue;
    changed += 1;
    const strength = amount / 255;
    pixels[at] = 255;
    pixels[at + 1] = Math.round(191 * (1 - strength));
    pixels[at + 3] = Math.round(96 + 159 * strength);
  }
  return { pixels, changed };
}

/** How much of the picture changed, in words. */
export function changedWords(changed: number, total: number): string {
  if (changed === 0 || total === 0) return "Nothing changed beyond noise.";
  const percent = (100 * changed) / total;
  // A change too small to round to a whole percent is still a change.
  const shown = percent < 1 ? "Less than 1%" : `${Math.round(percent)}%`;
  return `${shown} of the picture changed.`;
}
