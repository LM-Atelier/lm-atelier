/** The light and color preview worked out in a worker beside the page, and on the page without one. */

import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { adjustPixels, NEUTRAL_ADJUSTMENTS } from "./studioAdjustments";
import { adjustmentAnswerer, type AdjustWorkAnswer, type AdjustWorkMessage } from "./studioAdjustWork";
import { useAdjustedPreview } from "./useAdjustedPreview";

/** A worker that answers in the test's own thread, when the test lets it. */
class StandInWorker {
  static made: StandInWorker[] = [];
  static failing = false;
  onmessage: ((event: MessageEvent<AdjustWorkAnswer>) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  terminated = false;
  held: AdjustWorkAnswer[] = [];
  private readonly answer = adjustmentAnswerer();

  constructor(readonly url: URL | string, readonly options?: WorkerOptions) {
    StandInWorker.made.push(this);
  }

  postMessage(message: AdjustWorkMessage): void {
    if (StandInWorker.failing) {
      queueMicrotask(() => this.onerror?.(new Event("error")));
      return;
    }
    const reply = this.answer(message);
    if (reply) this.held.push(reply);
  }

  /** Send back every answer made so far. */
  release(): void {
    for (const reply of this.held.splice(0)) this.onmessage?.({ data: reply } as MessageEvent<AdjustWorkAnswer>);
  }

  terminate(): void {
    this.terminated = true;
  }
}

const drawn: ImageData[] = [];
let frames = 0;

beforeEach(() => {
  drawn.length = 0;
  frames = 0;
  StandInWorker.made = [];
  StandInWorker.failing = false;
  vi.stubGlobal("Worker", StandInWorker);
  vi.stubGlobal("ImageData", class {
    constructor(readonly data: Uint8ClampedArray, readonly width: number, readonly height: number) {}
  });
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    frames += 1;
    callback(0);
    return frames;
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    putImageData: (image: ImageData) => drawn.push(image),
  } as never);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const bitmap = { width: 2, height: 1 } as ImageBitmap;
const FIRST = new Uint8ClampedArray([200, 100, 50, 255, 10, 240, 128, 255]);
const SECOND = new Uint8ClampedArray([37, 91, 203, 255, 128, 128, 128, 255]);
const GREY = { ...NEUTRAL_ADJUSTMENTS, saturation: -100 };

describe("the preview beside the page", () => {
  it("is worked out in a worker, by the page's own arithmetic, and the worker ends with the view", async () => {
    const { result, unmount } = renderHook(() => useAdjustedPreview(bitmap, FIRST, GREY));
    const [worker] = StandInWorker.made;
    expect(String(worker.url)).toContain("studioAdjustWorker");
    expect(worker.options?.type).toBe("module");

    act(() => worker.release());

    await waitFor(() => expect(result.current).toBeInstanceOf(HTMLCanvasElement));
    expect(Array.from(drawn[0].data)).toEqual(Array.from(adjustPixels(FIRST, 2, GREY)));
    // Nothing was worked out on the page itself.
    expect(frames).toBe(0);
    unmount();
    expect(worker.terminated).toBe(true);
  });

  it("never shows an answer made for an earlier picture on a later one", async () => {
    const { result, rerender } = renderHook(({ pixels }) => useAdjustedPreview(bitmap, pixels, GREY), {
      initialProps: { pixels: FIRST },
    });
    const [worker] = StandInWorker.made;
    rerender({ pixels: SECOND });

    // Both answers come back only now: the first is for a picture no longer on screen.
    act(() => worker.release());

    await waitFor(() => expect(drawn).toHaveLength(1));
    expect(Array.from(drawn[0].data)).toEqual(Array.from(adjustPixels(SECOND, 2, GREY)));
    expect(result.current).toBeInstanceOf(HTMLCanvasElement);
  });

  it("falls back to the page when the worker fails", async () => {
    StandInWorker.failing = true;

    const { result } = renderHook(() => useAdjustedPreview(bitmap, FIRST, GREY));

    await waitFor(() => expect(result.current).toBeInstanceOf(HTMLCanvasElement));
    expect(StandInWorker.made[0].terminated).toBe(true);
    expect(frames).toBeGreaterThan(0);
    expect(Array.from(drawn[drawn.length - 1].data)).toEqual(Array.from(adjustPixels(FIRST, 2, GREY)));
  });
});
