import { useEffect, useState } from "react";
import { adjustPixels, isNeutral } from "./studioAdjustments";
import type { StudioColorAdjustments } from "./types";

/** The picture as the light and color sliders would leave it, for the canvas.
 *
 * Null when there is nothing to show but the picture itself: no adjustment,
 * no pixels to adjust, or a browser that gives no canvas. The work runs on the
 * next frame, so a slider dragged quickly redraws once per frame, not once per
 * step, and a result made for an earlier picture is never shown on a later one.
 */
export function useAdjustedPreview(
  bitmap: ImageBitmap | null,
  pixels: Uint8ClampedArray | null,
  adjustments: StudioColorAdjustments | null,
): HTMLCanvasElement | null {
  const [shown, setShown] = useState<{ pixels: Uint8ClampedArray; canvas: HTMLCanvasElement } | null>(null);
  const active = Boolean(bitmap && pixels && adjustments && !isNeutral(adjustments));

  useEffect(() => {
    if (!active || !bitmap || !pixels || !adjustments) return;
    const frame = requestAnimationFrame(() => {
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext("2d");
      if (!context) return;
      const adjusted = adjustPixels(pixels, bitmap.width, adjustments);
      context.putImageData(new ImageData(adjusted, bitmap.width, bitmap.height), 0, 0);
      setShown({ pixels, canvas });
    });
    return () => cancelAnimationFrame(frame);
  }, [active, bitmap, pixels, adjustments]);

  return active && shown && shown.pixels === pixels ? shown.canvas : null;
}
