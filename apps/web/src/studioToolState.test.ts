import { describe, expect, it } from "vitest";
import { coverage, isEmpty, fillRect } from "./studioMasks";
import { BrushTool, ClickSelectTool, RectTool } from "./studioTools";
import {
  initialToolState,
  studioToolReducer,
  snapshotBeforeGesture,
  toolFor,
  toolUsesMask,
  type StudioToolState,
} from "./studioToolState";

function withImage(width = 64, height = 64): StudioToolState {
  return studioToolReducer(initialToolState(), {
    type: "image-changed",
    width,
    height,
  });
}

describe("studio tool state", () => {
  it("starts in instruct mode with no mask and no pointer tool", () => {
    const state = initialToolState();
    expect(state.kind).toBe("instruct");
    expect(state.mask).toBeNull();
    expect(toolFor(state)).toBeNull();
  });

  it("gives each selection tool its own pointer behavior", () => {
    const base = withImage();
    expect(toolFor({ ...base, kind: "brush" })).toBeInstanceOf(BrushTool);
    expect(toolFor({ ...base, kind: "eraser" })).toBeInstanceOf(BrushTool);
    expect(toolFor({ ...base, kind: "rect" })).toBeInstanceOf(RectTool);
    expect(toolFor({ ...base, kind: "instruct" })).toBeNull();
  });

  it("gives the paint bucket its click, and the wand one only with the picture's pixels", () => {
    const base = withImage(4, 4);
    const pixels = new Uint8ClampedArray(4 * 4 * 4);
    expect(toolFor({ ...base, kind: "bucket" })).toBeInstanceOf(ClickSelectTool);
    expect(toolFor({ ...base, kind: "wand" }, pixels)).toBeInstanceOf(ClickSelectTool);
    expect(toolFor({ ...base, kind: "wand" }, null)).toBeNull();
    // Both draw the mask the apply sends, like every other selection.
    expect(toolUsesMask("bucket")).toBe(true);
    expect(toolUsesMask("wand")).toBe(true);
  });

  it("adds with the bucket and the wand by default, and takes away when asked", () => {
    let state = withImage(4, 4);
    const pixels = new Uint8ClampedArray(4 * 4 * 4);
    const wand = toolFor({ ...state, kind: "wand" }, pixels)!;
    wand.down({ x: 1, y: 1 });
    wand.up({ x: 1, y: 1 });
    expect(coverage(state.mask!)).toBe(1);

    state = studioToolReducer(state, { type: "set-selection-mode", mode: "remove" });
    const bucket = toolFor({ ...state, kind: "bucket" })!;
    bucket.down({ x: 2, y: 2 });
    bucket.up({ x: 2, y: 2 });
    expect(isEmpty(state.mask!)).toBe(true);
  });

  it("keeps the color tolerance within a byte", () => {
    let state = initialToolState();
    expect(state.colorTolerance).toBe(32);
    state = studioToolReducer(state, { type: "set-color-tolerance", tolerance: 400 });
    expect(state.colorTolerance).toBe(255);
    state = studioToolReducer(state, { type: "set-color-tolerance", tolerance: -3 });
    expect(state.colorTolerance).toBe(0);
  });

  it("undoes to the pre-gesture mask, not the painted one", () => {
    // The ordering that matters: snapshot as the gesture starts, mutate, then
    // stroke-end. Snapshotting at stroke end would store the painted raster
    // and leave the first Undo doing nothing at all.
    let state = withImage();
    const tool = toolFor({ ...state, kind: "brush" })!;

    snapshotBeforeGesture(state);
    tool.down({ x: 20, y: 20 });
    tool.up({ x: 40, y: 20 });
    state = studioToolReducer(state, { type: "stroke-end" });
    expect(isEmpty(state.mask!)).toBe(false);

    state = studioToolReducer(state, { type: "undo" });
    expect(isEmpty(state.mask!)).toBe(true);

    state = studioToolReducer(state, { type: "redo" });
    expect(isEmpty(state.mask!)).toBe(false);
  });

  it("treats invert, feather, and clear as undoable steps", () => {
    let state = withImage(32, 32);
    fillRect(state.mask!, 0, 0, 16, 32);
    const half = coverage(state.mask!);

    state = studioToolReducer(state, { type: "invert" });
    expect(coverage(state.mask!)).toBeCloseTo(1 - half);
    state = studioToolReducer(state, { type: "undo" });
    expect(coverage(state.mask!)).toBeCloseTo(half);

    state = studioToolReducer(state, { type: "clear" });
    expect(isEmpty(state.mask!)).toBe(true);
    state = studioToolReducer(state, { type: "undo" });
    expect(coverage(state.mask!)).toBeCloseTo(half);
  });

  it("drops the mask when the image changes", () => {
    let state = withImage(32, 32);
    fillRect(state.mask!, 0, 0, 32, 32);
    state = studioToolReducer(state, { type: "image-changed", width: 40, height: 20 });

    // Carrying a selection onto different pixels would silently mask the
    // wrong region, so a new image starts clean.
    expect(state.mask!.width).toBe(40);
    expect(isEmpty(state.mask!)).toBe(true);
    expect(state.history.canUndo).toBe(false);
  });

  it("bumps the repaint version only when the raster actually changes", () => {
    let state = withImage();
    const before = state.maskVersion;
    state = studioToolReducer(state, { type: "select-tool", kind: "brush" });
    state = studioToolReducer(state, { type: "set-brush-radius", radius: 40 });
    expect(state.maskVersion).toBe(before);
    expect(state.brushRadius).toBe(40);

    state = studioToolReducer(state, { type: "stroke-end" });
    expect(state.maskVersion).toBe(before + 1);
  });

  it("clamps brush and feather to usable ranges", () => {
    let state = withImage();
    state = studioToolReducer(state, { type: "set-brush-radius", radius: 0 });
    expect(state.brushRadius).toBe(1);
    state = studioToolReducer(state, { type: "set-brush-radius", radius: 9_999 });
    expect(state.brushRadius).toBe(512);
    state = studioToolReducer(state, { type: "set-feather", px: -5 });
    expect(state.featherPx).toBe(0);
  });

  it("ignores mask operations before an image is loaded", () => {
    const state = initialToolState();
    for (const action of ["undo", "redo", "invert", "clear"] as const) {
      expect(studioToolReducer(state, { type: action })).toBe(state);
    }
    snapshotBeforeGesture(state);
    expect(state.history.canUndo).toBe(false);
  });
});
