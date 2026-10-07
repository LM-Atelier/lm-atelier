import { useMemo } from "react";
import { drawStraightened } from "./studioStraighten";

/** The picture as straightening by `degrees` would keep it, for the canvas.
 *
 * Null when there is no angle to show or no canvas to draw on. The browser's
 * own resampling draws it, so it can differ from the kept picture by a level
 * at an edge, but what is in the box and how it is turned are the same.
 */
export function useStraightenPreview(bitmap: ImageBitmap | null, degrees: number | null): HTMLCanvasElement | null {
  return useMemo(
    () => (bitmap && degrees ? drawStraightened(bitmap, degrees) : null),
    [bitmap, degrees],
  );
}
