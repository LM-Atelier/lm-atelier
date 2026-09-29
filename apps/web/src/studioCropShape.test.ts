import { describe, expect, it } from "vitest";
import { constrainedCorner, cropRatio, fittedBox } from "./studioCropShape";
import { createMask, maskBounds } from "./studioMasks";
import { RectTool } from "./studioTools";
import { initialToolState, studioToolReducer, toolFor } from "./studioToolState";

const PICTURE = { width: 400, height: 200 };

describe("a crop shape's ratio", () => {
  it("holds a named shape and a small picture shape to whole multiples", () => {
    expect(cropRatio("free", PICTURE)).toBeNull();
    expect(cropRatio("16:9", PICTURE)).toEqual({ across: 16, down: 9, exact: true });
    expect(cropRatio("picture", PICTURE)).toEqual({ across: 2, down: 1, exact: true });
  });

  it("holds a picture shape with large terms to within a pixel instead", () => {
    expect(cropRatio("picture", { width: 1023, height: 767 })).toEqual({ across: 1023, down: 767, exact: false });
  });
});

describe("a corner drawn to a shape", () => {
  const square = { across: 1, down: 1, exact: true };
  const wide = { across: 16, down: 9, exact: true };

  it("grows the box by whichever way the pointer moved further", () => {
    expect(constrainedCorner({ x: 100, y: 50 }, { x: 140, y: 60 }, square, PICTURE)).toEqual({ x: 140, y: 90 });
    expect(constrainedCorner({ x: 100, y: 50 }, { x: 90, y: 20 }, square, PICTURE)).toEqual({ x: 70, y: 20 });
  });

  it("keeps whole multiples of the shape", () => {
    // Forty across is two and a half sixteens, so the box is two: 32 by 18.
    expect(constrainedCorner({ x: 10, y: 11 }, { x: 50, y: 20 }, wide, PICTURE)).toEqual({ x: 42, y: 29 });
  });

  it("stops at the picture's edge and keeps the shape there", () => {
    // Only 20 pixels remain below the origin, so the square stops at 20.
    expect(constrainedCorner({ x: 100, y: 180 }, { x: 300, y: 300 }, square, PICTURE)).toEqual({ x: 120, y: 200 });
  });

  it("fits a picture shape with large terms within a pixel", () => {
    const shape = cropRatio("picture", { width: 1023, height: 767 });
    if (!shape) throw new Error("no shape");
    const corner = constrainedCorner({ x: 0, y: 0 }, { x: 500, y: 10 }, shape, { width: 1023, height: 767 });
    expect(corner.x).toBe(500);
    expect(Math.abs(corner.y - (500 * 767) / 1023)).toBeLessThan(1);
  });
});

describe("a box refitted to a shape", () => {
  it("is the largest box of the shape inside the old one, centered in it", () => {
    expect(fittedBox({ left: 10, top: 20, width: 100, height: 40 }, { across: 1, down: 1, exact: true }))
      .toEqual({ left: 40, top: 20, width: 40, height: 40 });
  });

  it("is nothing when no box of the shape fits", () => {
    expect(fittedBox({ left: 0, top: 0, width: 10, height: 5 }, { across: 16, down: 9, exact: true })).toBeNull();
  });
});

describe("the crop box on the canvas", () => {
  it("counts from a whole pixel and keeps the chosen shape", () => {
    const mask = createMask(400, 200);
    const box = new RectTool(mask, true, { across: 16, down: 9, exact: true });
    box.down({ x: 10.4, y: 10.6 });
    box.move({ x: 50, y: 20 });

    expect(box.preview()).toEqual({ kind: "rect", from: { x: 10, y: 11 }, to: { x: 42, y: 29 } });
    expect(box.up({ x: 50, y: 20 })).toBe(true);
    expect(maskBounds(mask)).toEqual({ left: 10, top: 11, width: 32, height: 18 });
  });

  it("takes a newly chosen shape at once, and Undo gives the old box back", () => {
    let state = studioToolReducer(initialToolState(), { type: "image-changed", width: 400, height: 200 });
    state = studioToolReducer(state, { type: "select-tool", kind: "crop" });
    const drawing = toolFor(state);
    drawing?.down({ x: 10, y: 20 });
    drawing?.up({ x: 110, y: 60 });
    state = studioToolReducer(state, { type: "stroke-end" });

    state = studioToolReducer(state, { type: "set-crop-shape", shape: "1:1" });
    expect(state.cropShape).toBe("1:1");
    expect(state.mask && maskBounds(state.mask)).toEqual({ left: 40, top: 20, width: 40, height: 40 });

    state = studioToolReducer(state, { type: "undo" });
    expect(state.mask && maskBounds(state.mask)).toEqual({ left: 10, top: 20, width: 100, height: 40 });
  });

  it("changes nothing drawn when the shape is set free again", () => {
    let state = studioToolReducer(initialToolState(), { type: "image-changed", width: 400, height: 200 });
    state = studioToolReducer(state, { type: "select-tool", kind: "crop" });
    const drawing = toolFor(state);
    drawing?.down({ x: 10, y: 20 });
    drawing?.up({ x: 110, y: 60 });
    state = studioToolReducer(state, { type: "stroke-end" });

    const free = studioToolReducer(state, { type: "set-crop-shape", shape: "free" });

    expect(free.maskVersion).toBe(state.maskVersion);
    expect(free.mask && maskBounds(free.mask)).toEqual({ left: 10, top: 20, width: 100, height: 40 });
  });
});
