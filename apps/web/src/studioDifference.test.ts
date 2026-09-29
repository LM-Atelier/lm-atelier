import { describe, expect, it } from "vitest";
import { changedWords, differenceOverlay, NOISE_LEVELS } from "./studioDifference";

function rgba(...pixels: number[][]): Uint8ClampedArray {
  return new Uint8ClampedArray(pixels.flat());
}

describe("what an edit changed", () => {
  it("leaves clear a pixel that did not change, or changed by no more than noise", () => {
    const before = rgba([10, 20, 30, 255], [200, 100, 50, 255]);
    const after = rgba([10, 20, 30, 255], [200 + NOISE_LEVELS, 100 - NOISE_LEVELS, 50, 255]);

    const { pixels, changed } = differenceOverlay(before, after);

    expect(Array.from(pixels)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    expect(changed).toBe(0);
  });

  it("tints a change from amber to red, less see-through the more it changed", () => {
    const before = rgba([0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255]);
    const after = rgba([NOISE_LEVELS + 1, 0, 0, 255], [0, 128, 0, 255], [255, 255, 255, 255]);

    const { pixels, changed } = differenceOverlay(before, after);

    expect(changed).toBe(3);
    // A change of 5 of 255 keeps green at 191 * 250/255 = 187 and opacity at
    // 96 + 159 * 5/255 = 99; 128 halves both ways; 255 is red and opaque.
    expect(Array.from(pixels.slice(0, 4))).toEqual([255, 187, 0, 99]);
    expect(Array.from(pixels.slice(4, 8))).toEqual([255, 95, 0, 176]);
    expect(Array.from(pixels.slice(8, 12))).toEqual([255, 0, 0, 255]);
  });

  it("counts a change in opacity alone", () => {
    expect(differenceOverlay(rgba([50, 50, 50, 255]), rgba([50, 50, 50, 0])).changed).toBe(1);
  });
});

describe("how much changed, in words", () => {
  it("says nothing changed, less than a percent, or the nearest percent", () => {
    expect(changedWords(0, 100)).toBe("Nothing changed beyond noise.");
    expect(changedWords(1, 1000)).toBe("Less than 1% of the picture changed.");
    // 0.7% would round to a whole 1%, more than did change.
    expect(changedWords(7, 1000)).toBe("Less than 1% of the picture changed.");
    expect(changedWords(505, 1000)).toBe("51% of the picture changed.");
    expect(changedWords(1000, 1000)).toBe("100% of the picture changed.");
  });
});
