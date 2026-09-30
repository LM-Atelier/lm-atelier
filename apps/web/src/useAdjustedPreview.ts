import { useEffect, useRef, useState } from "react";
import { adjustPixels, isNeutral } from "./studioAdjustments";
import { AdjustmentWork, type AdjustWorkAnswer } from "./studioAdjustWork";
import type { StudioColorAdjustments } from "./types";

type Shown = { pixels: Uint8ClampedArray; canvas: HTMLCanvasElement };
type Held = { id: number; pixels: Uint8ClampedArray; width: number; height: number };
type OffPage = { worker: Worker; work: AdjustmentWork };

/** A canvas holding `adjusted` at the picture's size, or null where the browser gives no canvas. */
function canvasOf(adjusted: Uint8ClampedArray, width: number, height: number): HTMLCanvasElement | null {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.putImageData(new ImageData(adjusted, width, height), 0, 0);
  return canvas;
}

/** A worker for the preview's arithmetic, or null where the browser has none to give. */
function startWorker(): Worker | null {
  if (typeof Worker === "undefined") return null;
  try {
    return new Worker(new URL("./studioAdjustWorker.ts", import.meta.url), { type: "module" });
  } catch {
    return null;
  }
}

/** The picture as the light and color sliders would leave it, for the canvas.
 *
 * Null when there is nothing to show but the picture itself: no adjustment,
 * no pixels to adjust, or a browser that gives no canvas. Where the browser
 * has workers, the arithmetic runs in one beside the page, so the page keeps
 * answering the slider while a large picture is worked out. Otherwise it runs
 * on the page on the next frame, so a slider dragged quickly redraws once per
 * frame, not once per step. Either way the result is the same arithmetic,
 * and one made for an earlier picture is never shown on a later one.
 */
export function useAdjustedPreview(
  bitmap: ImageBitmap | null,
  pixels: Uint8ClampedArray | null,
  adjustments: StudioColorAdjustments | null,
): HTMLCanvasElement | null {
  const [shown, setShown] = useState<Shown | null>(null);
  // Once a worker fails, the page does the work for as long as it is open.
  const [pageOnly, setPageOnly] = useState(false);
  const active = Boolean(bitmap && pixels && adjustments && !isNeutral(adjustments));
  // Undefined until first wanted; null where there is no worker to be had.
  const offPage = useRef<OffPage | null | undefined>(undefined);
  const held = useRef<Held | null>(null);

  useEffect(() => () => {
    offPage.current?.worker.terminate();
    offPage.current = undefined;
    held.current = null;
  }, []);

  useEffect(() => {
    if (!active || !bitmap || !pixels || !adjustments) {
      offPage.current?.work.forget();
      return;
    }
    const connect = (worker: Worker): OffPage => {
      const work = new AdjustmentWork(
        (message) => worker.postMessage(message),
        (picture, adjusted) => {
          const current = held.current;
          if (!current || current.id !== picture) return;
          const canvas = canvasOf(adjusted, current.width, current.height);
          if (canvas) setShown({ pixels: current.pixels, canvas });
        },
      );
      worker.onmessage = (event: MessageEvent<AdjustWorkAnswer>) => work.answered(event.data);
      worker.onerror = () => {
        worker.terminate();
        offPage.current = null;
        held.current = null;
        setPageOnly(true);
      };
      return { worker, work };
    };
    if (offPage.current === undefined && !pageOnly) {
      const worker = startWorker();
      offPage.current = worker ? connect(worker) : null;
    }
    const helper = pageOnly ? null : offPage.current;
    if (helper) {
      if (held.current?.pixels !== pixels) {
        const id = helper.work.hold(pixels, bitmap.width);
        held.current = { id, pixels, width: bitmap.width, height: bitmap.height };
      }
      helper.work.adjust(adjustments);
      return;
    }
    const frame = requestAnimationFrame(() => {
      const canvas = canvasOf(adjustPixels(pixels, bitmap.width, adjustments), bitmap.width, bitmap.height);
      if (canvas) setShown({ pixels, canvas });
    });
    return () => cancelAnimationFrame(frame);
  }, [active, bitmap, pixels, adjustments, pageOnly]);

  return active && shown && shown.pixels === pixels ? shown.canvas : null;
}
