import type { OutputRatioPresetId } from "./types";

/** The vocabulary of shapes, and how to tell which one a pair of pixels is.
 *
 * Its own file because the component file exports a component, and because
 * neither of these needs a rendered tree to be true. Which of these shapes a
 * given workflow can actually produce is the server's answer, never this one's.
 */

export const RATIO_LABELS: Record<OutputRatioPresetId, string> = {
  "1:1": "Square",
  "3:4": "Portrait",
  "2:3": "Portrait tall",
  "9:16": "Tall",
  "4:3": "Landscape",
  "3:2": "Landscape wide",
  "16:9": "Wide",
};

/** Which offered ratio these pixels already are, if any.
 *
 * Classifying a pair, not computing one: the id IS the reduced ratio, so a
 * stored 1024 by 576 is the wide preset whether it was chosen here or typed
 * into the number boxes. Anything that is not a pair of positive integers
 * classifies as nothing rather than reducing to a fraction of NaN.
 */
export function ratioOf(
  width: unknown,
  height: unknown,
  offered: readonly OutputRatioPresetId[],
): OutputRatioPresetId | null {
  if (!Number.isInteger(width) || !Number.isInteger(height)) return null;
  const across = width as number;
  const down = height as number;
  if (across <= 0 || down <= 0) return null;
  return offered.find((preset) => {
    const [numerator, denominator] = preset.split(":").map(Number);
    // Cross-multiplication rather than division: exact in integers, and it
    // cannot produce a value the settings filter would have to catch.
    return across * denominator === down * numerator;
  }) ?? null;
}
