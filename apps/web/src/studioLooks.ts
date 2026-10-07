import { NEUTRAL_ADJUSTMENTS } from "./studioAdjustments";
import type { StudioColorAdjustments } from "./types";

/** Looks: named settings of the ordinary light and color sliders.
 *
 * A look is a starting point, not an effect of its own. Choosing one sets the
 * sliders to it from their middle, so what it did is there to read on the
 * sliders, each can still be moved, and the server makes the picture with the
 * arithmetic it always uses. A look leaves every slider it does not name at
 * zero, so choosing another one never adds to the last.
 */
export const LOOKS: ReadonlyArray<{ name: string; adjustments: Partial<StudioColorAdjustments> }> = [
  { name: "Vivid", adjustments: { contrast: 15, saturation: 10, vibrance: 30 } },
  { name: "Soft", adjustments: { contrast: -20, highlights: -15, shadows: 15, sharpness: -10 } },
  { name: "Warm", adjustments: { warmth: 30, vibrance: 10 } },
  { name: "Cool", adjustments: { warmth: -30, tint: -5 } },
  { name: "Mono", adjustments: { saturation: -100, contrast: 10 } },
  { name: "Faded", adjustments: { contrast: -30, shadows: 25, saturation: -25 } },
  { name: "Dramatic", adjustments: { contrast: 35, highlights: -25, shadows: -10, vignette: 35 } },
  { name: "Film", adjustments: { contrast: 10, saturation: -15, warmth: 10, grain: 35 } },
];

/** Every slider as the look sets it. */
export function lookAdjustments(look: (typeof LOOKS)[number]): StudioColorAdjustments {
  return { ...NEUTRAL_ADJUSTMENTS, ...look.adjustments };
}

/** The look the sliders stand at exactly, with the tone curve as the look left it, if any. */
export function currentLook(adjustments: StudioColorAdjustments): string | null {
  const keys = Object.keys(NEUTRAL_ADJUSTMENTS) as Array<keyof StudioColorAdjustments>;
  const look = LOOKS.find((candidate) => {
    const set = lookAdjustments(candidate);
    // Compared by value: a curve is a list of points, and two lists alike are two lists.
    return keys.every((key) => JSON.stringify(set[key]) === JSON.stringify(adjustments[key]));
  });
  return look ? look.name : null;
}
