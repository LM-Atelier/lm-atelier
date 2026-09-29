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

// A picture with edges in it, for sharpness, which reads each pixel's
// neighbors: the same pixels and results test_studio_adjustments.py checks the
// server against, opaque and as a cutout. In the cutout some pixels are partly
// transparent, and some wholly so with a color hidden under them.
const SHARP_WIDTH = 6;
const SHARP_COLORS = [
  [12, 40, 200], [60, 60, 60], [200, 30, 30], [250, 250, 250], [0, 0, 0], [90, 180, 40],
  [30, 30, 30], [220, 200, 10], [128, 128, 128], [15, 90, 210], [240, 120, 60], [33, 66, 99],
  [255, 255, 255], [10, 10, 10], [180, 40, 160], [70, 200, 120], [0, 255, 0], [128, 0, 255],
  [45, 45, 200], [150, 150, 20], [5, 5, 5], [250, 10, 120], [100, 100, 100], [200, 200, 200],
  [77, 22, 11], [0, 128, 255], [255, 128, 0], [60, 30, 90], [210, 210, 40], [18, 180, 180],
];
const SHARP_ALPHAS = [
  255, 255, 255, 255, 0, 255,
  255, 128, 255, 64, 255, 255,
  0, 255, 200, 255, 17, 255,
  255, 255, 255, 255, 128, 0,
  255, 255, 0, 255, 255, 255,
];
const SHARP_CASES: Array<[string, Partial<StudioColorAdjustments>, boolean, number[][]]> = [
  ["crisper", { sharpness: 60 }, false, [
    [12, 40, 200, 255], [60, 60, 60, 255], [200, 30, 30, 255], [250, 250, 250, 255], [0, 0, 0, 255], [90, 180, 40, 255],
    [30, 30, 30, 255], [255, 255, 0, 255], [125, 138, 138, 255], [0, 66, 255, 255], [255, 119, 39, 255], [33, 66, 99, 255],
    [255, 255, 255, 255], [0, 0, 0, 255], [221, 15, 199, 255], [50, 247, 126, 255], [0, 255, 0, 255], [128, 0, 255, 255],
    [45, 45, 200, 255], [184, 186, 0, 255], [0, 0, 0, 255], [255, 0, 142, 255], [85, 79, 94, 255], [200, 200, 200, 255],
    [77, 22, 11, 255], [0, 128, 255, 255], [255, 128, 0, 255], [60, 30, 90, 255], [210, 210, 40, 255], [18, 180, 180, 255]]],
  ["softer", { sharpness: -45 }, false, [
    [12, 40, 200, 255], [60, 60, 60, 255], [200, 30, 30, 255], [250, 250, 250, 255], [0, 0, 0, 255], [90, 180, 40, 255],
    [30, 30, 30, 255], [176, 155, 37, 255], [130, 119, 120, 255], [59, 108, 175, 255], [177, 120, 75, 255], [33, 66, 99, 255],
    [255, 255, 255, 255], [57, 48, 41, 255], [148, 58, 130, 255], [84, 164, 115, 255], [44, 202, 47, 255], [128, 0, 255, 255],
    [45, 45, 200, 255], [124, 122, 51, 255], [54, 32, 33, 255], [196, 43, 102, 255], [111, 115, 104, 255], [200, 200, 200, 255],
    [77, 22, 11, 255], [0, 128, 255, 255], [255, 128, 0, 255], [60, 30, 90, 255], [210, 210, 40, 255], [18, 180, 180, 255]]],
  ["crisper, warmer and paler", { sharpness: 100, warmth: 30, saturation: -20 }, false, [
    [20, 41, 162, 255], [62, 60, 57, 255], [183, 40, 39, 255], [253, 248, 241, 255], [0, 0, 0, 255], [102, 170, 58, 255],
    [30, 30, 29, 255], [255, 255, 15, 255], [130, 144, 140, 255], [0, 46, 225, 255], [255, 132, 55, 255], [39, 63, 87, 255],
    [254, 253, 245, 255], [0, 0, 0, 255], [227, 20, 191, 255], [68, 255, 138, 255], [0, 255, 0, 255], [120, 13, 208, 255],
    [50, 48, 165, 255], [210, 204, 0, 255], [0, 0, 0, 255], [255, 0, 140, 255], [77, 65, 82, 255], [206, 198, 192, 255],
    [71, 25, 16, 255], [20, 122, 215, 255], [234, 131, 30, 255], [59, 33, 78, 255], [213, 204, 68, 255], [41, 169, 163, 255]]],
  ["a crisper cutout", { sharpness: 60 }, true, [
    [12, 40, 200, 255], [60, 60, 60, 255], [200, 30, 30, 255], [250, 250, 250, 255], [0, 0, 0, 0], [90, 180, 40, 255],
    [30, 30, 30, 255], [255, 255, 0, 128], [122, 140, 141, 255], [0, 57, 253, 64], [255, 117, 27, 255], [33, 66, 99, 255],
    [255, 255, 255, 0], [0, 0, 0, 255], [223, 17, 203, 200], [36, 255, 125, 255], [0, 255, 0, 17], [128, 0, 255, 255],
    [45, 45, 200, 255], [199, 195, 0, 255], [0, 0, 0, 255], [255, 0, 138, 255], [78, 95, 91, 128], [255, 255, 255, 0],
    [77, 22, 11, 255], [0, 128, 255, 255], [255, 204, 0, 0], [60, 30, 90, 255], [210, 210, 40, 255], [18, 180, 180, 255]]],
  ["the softest cutout", { sharpness: -100 }, true, [
    [12, 40, 200, 255], [60, 60, 60, 255], [200, 30, 30, 255], [250, 250, 250, 255], [0, 0, 0, 0], [90, 180, 40, 255],
    [30, 30, 30, 255], [97, 75, 65, 128], [138, 107, 105, 255], [159, 144, 137, 64], [147, 124, 114, 255], [33, 66, 99, 255],
    [0, 0, 0, 0], [84, 65, 56, 255], [108, 78, 88, 200], [126, 107, 111, 255], [132, 95, 130, 17], [128, 0, 255, 255],
    [45, 45, 200, 255], [68, 74, 83, 255], [92, 56, 75, 255], [134, 67, 90, 255], [136, 107, 114, 128], [0, 0, 0, 0],
    [77, 22, 11, 255], [0, 128, 255, 255], [0, 0, 0, 0], [60, 30, 90, 255], [210, 210, 40, 255], [18, 180, 180, 255]]],
];

function rgba(pixels: number[][], alpha = 255): Uint8ClampedArray {
  return new Uint8ClampedArray(pixels.flatMap(([r, g, b]) => [r, g, b, alpha]));
}

describe("light and color adjustments", () => {
  it.each(CASES)("makes the server's pixels when %s", (_name, sliders, expected) => {
    const adjusted = adjustPixels(rgba(PIXELS), PIXELS.length, { ...NEUTRAL_ADJUSTMENTS, ...sliders });

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

    const adjusted = adjustPixels(pixels, 2, { ...NEUTRAL_ADJUSTMENTS, brightness: 60, saturation: -50 });

    expect([adjusted[3], adjusted[7]]).toEqual([0, 77]);
  });
});

describe("sharpness", () => {
  it.each(SHARP_CASES)("makes the server's pixels for %s", (_name, sliders, cutout, expected) => {
    const pixels = new Uint8ClampedArray(
      SHARP_COLORS.flatMap(([r, g, b], index) => [r, g, b, cutout ? SHARP_ALPHAS[index] : 255]),
    );

    const adjusted = adjustPixels(pixels, SHARP_WIDTH, { ...NEUTRAL_ADJUSTMENTS, ...sliders });

    expect(Array.from(adjusted)).toEqual(expected.flat());
  });

  it("counts as a change on its own", () => {
    expect(isNeutral({ ...NEUTRAL_ADJUSTMENTS, sharpness: -1 })).toBe(false);
  });
});
