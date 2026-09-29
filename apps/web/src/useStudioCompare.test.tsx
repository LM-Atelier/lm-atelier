/** What changed: an overlay made once per result, shown in place of the earlier picture, and always released. */

import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { readSourcePixels } from "./studioSourcePixels";
import { useStudioCompare } from "./useStudioCompare";
import { useStudioImage } from "./useStudioImage";
import type { StudioStep } from "./useStudioSession";

vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./studioSourcePixels", () => ({ readSourcePixels: vi.fn() }));

function bitmap(width: number, height: number): ImageBitmap {
  return { width, height, close: vi.fn() } as unknown as ImageBitmap;
}

function result(artifactId: string, beforeArtifactId: string): StudioStep {
  return { messageId: `message-${artifactId}`, artifactId, instruction: "", beforeArtifactId, isSource: false, generationIdentity: null };
}

// Four pixels, the last two of which the edit changed.
const EARLIER = new Uint8ClampedArray([9, 9, 9, 255, 9, 9, 9, 255, 9, 9, 9, 255, 9, 9, 9, 255]);
const LATER = new Uint8ClampedArray([9, 9, 9, 255, 9, 9, 9, 255, 200, 9, 9, 255, 9, 200, 9, 255]);

let overlays: ImageBitmap[];
let before: ImageBitmap;
let shown: ImageBitmap;

beforeEach(() => {
  overlays = [];
  before = bitmap(4, 1);
  shown = bitmap(4, 1);
  vi.mocked(useStudioImage).mockImplementation((artifactId: string | null) => ({
    bitmap: artifactId ? before : null,
    error: null,
    reload: vi.fn(),
  }));
  vi.mocked(readSourcePixels).mockImplementation((image: ImageBitmap) => (image === before ? EARLIER : LATER));
  vi.stubGlobal("ImageData", class {
    constructor(readonly data: Uint8ClampedArray, readonly width: number, readonly height: number) {}
  });
  vi.stubGlobal("createImageBitmap", vi.fn(async () => {
    const made = bitmap(4, 1);
    overlays.push(made);
    return made;
  }));
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("what changed", () => {
  it("tints what changed in place of the earlier picture once asked, and says how much", async () => {
    const { result: hook } = renderHook(() => useStudioCompare(result("art-2", "art-1"), shown, false));
    expect(hook.current.layer).toEqual({ image: before, reveal: 0 });
    expect(hook.current.controls.canDiffer).toBe(true);

    act(() => hook.current.controls.onDifference(true));

    await waitFor(() => expect(hook.current.layer?.image).toBe(overlays[0]));
    expect(hook.current.layer?.reveal).toBe(1);
    expect(hook.current.controls.difference).toBe(true);
    expect(hook.current.controls.changed).toBe("50% of the picture changed.");
    // Holding still shows the earlier picture itself.
    act(() => hook.current.controls.onHold(true));
    expect(hook.current.layer).toEqual({ image: before, reveal: 1 });
  });

  it("gives way to a split, and a split to it", async () => {
    const { result: hook } = renderHook(() => useStudioCompare(result("art-2", "art-1"), shown, false));

    act(() => hook.current.controls.onDifference(true));
    await waitFor(() => expect(hook.current.layer?.image).toBe(overlays[0]));
    act(() => hook.current.controls.onSplit(0.5));
    expect(hook.current.controls.difference).toBe(false);
    expect(hook.current.layer).toEqual({ image: before, reveal: 0.5 });

    act(() => hook.current.controls.onDifference(true));
    expect(hook.current.controls.split).toBeNull();
  });

  it("is not offered between pictures of different sizes, even of one shape", () => {
    // A workflow's rounding: the same picture to the eye, and still split, but
    // its pixels no longer line up one to one.
    before = bitmap(1000, 667);
    shown = bitmap(1000, 672);
    const { result: hook } = renderHook(() => useStudioCompare(result("art-2", "art-1"), shown, false));

    expect(hook.current.controls.canSplit).toBe(true);
    expect(hook.current.controls.canDiffer).toBe(false);
    act(() => hook.current.controls.onDifference(true));
    expect(hook.current.controls.difference).toBe(false);
    expect(createImageBitmap).not.toHaveBeenCalled();
  });

  it("releases the overlay when it is turned off, when the result changes, and when the studio closes", async () => {
    const { result: hook, rerender, unmount } = renderHook(
      ({ step }) => useStudioCompare(step, shown, false),
      { initialProps: { step: result("art-2", "art-1") } },
    );

    act(() => hook.current.controls.onDifference(true));
    await waitFor(() => expect(hook.current.layer?.image).toBe(overlays[0]));
    act(() => hook.current.controls.onDifference(false));
    expect(overlays[0].close).toHaveBeenCalled();
    expect(hook.current.layer).toEqual({ image: before, reveal: 0 });

    act(() => hook.current.controls.onDifference(true));
    await waitFor(() => expect(hook.current.layer?.image).toBe(overlays[1]));
    // Another result has its own comparison, off until asked for.
    shown = bitmap(4, 1);
    rerender({ step: result("art-3", "art-1") });
    expect(overlays[1].close).toHaveBeenCalled();
    expect(hook.current.controls.difference).toBe(false);

    act(() => hook.current.controls.onDifference(true));
    await waitFor(() => expect(hook.current.layer?.image).toBe(overlays[2]));
    unmount();
    expect(overlays[2].close).toHaveBeenCalled();
  });

  it("closes an overlay that finishes after it was no longer wanted", async () => {
    let finish: (made: ImageBitmap) => void = () => undefined;
    vi.mocked(createImageBitmap).mockImplementation(() => new Promise<ImageBitmap>((resolve) => (finish = resolve)));
    const { result: hook } = renderHook(() => useStudioCompare(result("art-2", "art-1"), shown, false));

    act(() => hook.current.controls.onDifference(true));
    await waitFor(() => expect(createImageBitmap).toHaveBeenCalled());
    act(() => hook.current.controls.onDifference(false));
    const late = bitmap(4, 1);
    await act(async () => finish(late));

    expect(late.close).toHaveBeenCalled();
    expect(hook.current.layer).toEqual({ image: before, reveal: 0 });
  });
});
