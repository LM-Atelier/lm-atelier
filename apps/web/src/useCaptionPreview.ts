import { useEffect, useState } from "react";
import { drawCaption, type StudioCaption } from "./studioCaption";

/** The picture with the words over it, for the canvas while they are written.
 *
 * Null when there are no words to show or no canvas to draw them on. Drawing
 * waits for the shipped typefaces, so a result that arrives after the words
 * changed again is dropped, and one made for an earlier picture is never shown
 * on a later one.
 */
export function useCaptionPreview(bitmap: ImageBitmap | null, caption: StudioCaption | null): HTMLCanvasElement | null {
  const [shown, setShown] = useState<{ bitmap: ImageBitmap; canvas: HTMLCanvasElement } | null>(null);
  const active = Boolean(bitmap && caption && caption.text.trim());
  const key = caption ? JSON.stringify(caption) : "";

  useEffect(() => {
    if (!active || !bitmap || !caption) return;
    let current = true;
    void drawCaption(bitmap.width, bitmap.height, caption).then((words) => {
      if (!current || !words) return;
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.drawImage(bitmap, 0, 0);
      context.drawImage(words, 0, 0);
      setShown({ bitmap, canvas });
    });
    return () => {
      current = false;
    };
    // The caption is compared by its key: a new object with the same words
    // must not redraw.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, bitmap, key]);

  return active && shown && shown.bitmap === bitmap ? shown.canvas : null;
}
