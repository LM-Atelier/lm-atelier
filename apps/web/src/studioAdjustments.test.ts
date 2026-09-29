/** The preview's arithmetic, checked against the pixels the server makes. */

import { describe, expect, it } from "vitest";
import { adjustPixels, channelTables, isNeutral, NEUTRAL_ADJUSTMENTS } from "./studioAdjustments";
import type { StudioColorAdjustments } from "./types";

// The same pixels and results test_studio_adjustments.py checks the server
// against. If either side's arithmetic moves, one of the two suites fails. The
// last three pixels sit where rounding the saturation mix to single precision
// changes the answer, so they fail if it is ever done in double.
const PIXELS = [
  [200, 100, 50], [10, 240, 128], [128, 128, 128], [255, 0, 255], [0, 0, 0],
  [255, 255, 255], [37, 91, 203], [0, 170, 0], [210, 253, 126], [169, 45, 156],
];
const CASES: Array<[string, Partial<StudioColorAdjustments>, number[][]]> = [
  ["brighter", { brightness: 40 }, [
    [255, 132, 66], [13, 255, 169], [169, 169, 169], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [49, 120, 255], [0, 224, 0], [255, 255, 166], [223, 59, 206]]],
  ["flatter", { contrast: -30 }, [
    [186, 105, 65], [32, 219, 128], [128, 128, 128], [231, 24, 231], [24, 24, 24],
    [231, 231, 231], [54, 98, 189], [24, 162, 24], [195, 229, 126], [161, 60, 151]]],
  ["grey", { saturation: -100 }, [
    [124, 124, 124], [158, 158, 158], [128, 128, 128], [105, 105, 105], [0, 0, 0],
    [255, 255, 255], [88, 88, 88], [100, 100, 100], [226, 226, 226], [95, 95, 95]]],
  ["richer", { saturation: 60 }, [
    [245, 85, 5], [0, 255, 110], [128, 128, 128], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [6, 92, 255], [0, 212, 0], [200, 255, 66], [213, 15, 192]]],
  ["paler", { saturation: -41 }, [
    [168, 109, 80], [70, 206, 140], [128, 128, 128], [193, 43, 193], [0, 0, 0],
    [255, 255, 255], [57, 89, 155], [41, 141, 41], [216, 241, 167], [138, 65, 130]]],
  ["livelier", { saturation: 8 }, [
    [206, 98, 44], [0, 246, 125], [128, 128, 128], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [32, 91, 212], [0, 175, 0], [208, 255, 117], [174, 40, 160]]],
  ["warmer", { warmth: 50 }, [
    [215, 99, 46], [11, 237, 118], [138, 126, 118], [255, 0, 235], [0, 0, 0],
    [255, 251, 235], [40, 90, 187], [0, 168, 0], [226, 249, 116], [182, 44, 144]]],
  ["magenta", { tint: 60 }, [
    [227, 95, 57], [11, 227, 145], [145, 121, 145], [255, 0, 255], [0, 0, 0],
    [255, 241, 255], [42, 86, 230], [0, 161, 0], [238, 240, 143], [192, 43, 177]]],
  ["greener", { tint: -100 }, [
    [160, 108, 40], [8, 255, 102], [102, 138, 102], [204, 0, 204], [0, 0, 0],
    [204, 255, 204], [30, 98, 162], [0, 184, 0], [168, 255, 101], [135, 49, 125]]],
  ["tinted and warmed", { tint: 35, warmth: -45, brightness: 15 }, [
    [232, 107, 66], [12, 255, 169], [148, 137, 169], [255, 0, 255], [0, 0, 0],
    [255, 255, 255], [43, 98, 255], [0, 183, 0], [243, 255, 167], [196, 48, 207]]],
  ["everything", { brightness: -20, contrast: 35, saturation: 25, warmth: -40 }, [
    [200, 69, 7], [0, 251, 114], [102, 107, 124], [255, 0, 255], [0, 0, 0],
    [238, 247, 255], [0, 66, 250], [0, 168, 0], [186, 252, 95], [166, 1, 177]]],
];

function rgba(pixels: number[][], alpha = 255): Uint8ClampedArray {
  return new Uint8ClampedArray(pixels.flatMap(([r, g, b]) => [r, g, b, alpha]));
}

describe("light and color adjustments", () => {
  it.each(CASES)("makes the server's pixels when %s", (_name, sliders, expected) => {
    const adjusted = adjustPixels(rgba(PIXELS), { ...NEUTRAL_ADJUSTMENTS, ...sliders });

    expect(Array.from(adjusted)).toEqual(Array.from(rgba(expected)));
  });

  it("changes nothing with every slider at zero", () => {
    expect(isNeutral(NEUTRAL_ADJUSTMENTS)).toBe(true);
    for (const table of channelTables(NEUTRAL_ADJUSTMENTS)) {
      expect(Array.from(table)).toEqual(Array.from({ length: 256 }, (_, value) => value));
    }
  });

  it("leaves transparency as it was", () => {
    const pixels = new Uint8ClampedArray([200, 100, 50, 0, 200, 100, 50, 77]);

    const adjusted = adjustPixels(pixels, { ...NEUTRAL_ADJUSTMENTS, brightness: 60, saturation: -50 });

    expect([adjusted[3], adjusted[7]]).toEqual([0, 77]);
  });
});
