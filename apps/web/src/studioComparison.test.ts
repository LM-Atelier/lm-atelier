import { describe, expect, it } from "vitest";
import { compareFit, sameShape } from "./studioComparison";

describe("sameShape", () => {
  it("counts a size a workflow rounded as the same shape, and an extended one as another", () => {
    expect(sameShape({ width: 640, height: 400 }, { width: 1280, height: 800 })).toBe(true);
    // Rounded to a multiple of 16: under one percent apart.
    expect(sameShape({ width: 1000, height: 667 }, { width: 1000, height: 672 })).toBe(true);
    // Extended by an eighth to one side.
    expect(sameShape({ width: 640, height: 400 }, { width: 720, height: 400 })).toBe(false);
  });

  it("gives no shape to a picture with no size", () => {
    expect(sameShape({ width: 0, height: 400 }, { width: 640, height: 400 })).toBe(false);
    expect(sameShape({ width: 640, height: 400 }, { width: 640, height: 0 })).toBe(false);
  });
});

describe("compareFit", () => {
  it("fills the frame with a picture of the same shape, edge for edge", () => {
    expect(compareFit({ width: 640, height: 400 }, { width: 1280, height: 800 })).toEqual({
      x: 0,
      y: 0,
      width: 1280,
      height: 800,
    });
  });

  it("fits a picture of another shape whole and centred, never stretched", () => {
    // The picture before it was extended to both sides.
    expect(compareFit({ width: 400, height: 400 }, { width: 800, height: 400 })).toEqual({
      x: 200,
      y: 0,
      width: 400,
      height: 400,
    });
    // Wider than the frame: bars above and below.
    expect(compareFit({ width: 800, height: 200 }, { width: 400, height: 400 })).toEqual({
      x: 0,
      y: 150,
      width: 400,
      height: 100,
    });
  });

  it("draws nothing for a picture with no size, rather than NaN", () => {
    expect(compareFit({ width: 0, height: 0 }, { width: 400, height: 400 })).toEqual({
      x: 0,
      y: 0,
      width: 0,
      height: 0,
    });
  });
});
