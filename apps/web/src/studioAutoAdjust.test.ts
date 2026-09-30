/** Auto: light and color worked out from the picture, within a moderate reach. */

import { describe, expect, it } from "vitest";
import { channelTables, NEUTRAL_ADJUSTMENTS } from "./studioAdjustments";
import { autoAdjustments } from "./studioAutoAdjust";
import type { StudioColorAdjustments } from "./types";

/** Every level from `low` to `high`, as grey pixels with each channel scaled by `cast`. */
function ramp(low: number, high: number, cast: [number, number, number] = [1, 1, 1], alpha = 255): Uint8ClampedArray {
  const pixels = new Uint8ClampedArray(256 * 4);
  for (let index = 0; index < 256; index += 1) {
    const level = low + ((high - low) * index) / 255;
    pixels.set([level * cast[0], level * cast[1], level * cast[2], alpha], index * 4);
  }
  return pixels;
}

/** Each channel's average after the sliders' own tables. */
function averages(pixels: Uint8ClampedArray, adjustments: StudioColorAdjustments): [number, number, number] {
  const tables = channelTables(adjustments);
  const sums = [0, 0, 0];
  for (let at = 0; at < pixels.length; at += 4) {
    for (let channel = 0; channel < 3; channel += 1) sums[channel] += tables[channel][pixels[at + channel]];
  }
  const count = pixels.length / 4;
  return [sums[0] / count, sums[1] / count, sums[2] / count];
}

/** The brightness levels below which 2%, 50% and 98% of the pixels lie, after the sliders. */
function levels(pixels: Uint8ClampedArray, adjustments: StudioColorAdjustments): [number, number, number] {
  const [r, g, b] = channelTables(adjustments);
  const sorted: number[] = [];
  for (let at = 0; at < pixels.length; at += 4) {
    sorted.push(Math.round(0.2126 * r[pixels[at]] + 0.7152 * g[pixels[at + 1]] + 0.0722 * b[pixels[at + 2]]));
  }
  sorted.sort((left, right) => left - right);
  const at = (share: number) => sorted[Math.max(0, Math.ceil(share * sorted.length) - 1)];
  return [at(0.02), at(0.5), at(0.98)];
}

describe("Auto", () => {
  it("leaves a balanced picture as it was", () => {
    expect(autoAdjustments(ramp(10, 240), NEUTRAL_ADJUSTMENTS)).toEqual(NEUTRAL_ADJUSTMENTS);
  });

  it("brings a dark picture's middle up to middle grey", () => {
    const dark = ramp(10, 180);
    const set = autoAdjustments(dark, NEUTRAL_ADJUSTMENTS);

    expect(levels(dark, NEUTRAL_ADJUSTMENTS)[1]).toBeLessThan(100);
    expect(set.brightness).toBeGreaterThan(0);
    expect(Math.abs(levels(dark, set)[1] - 118)).toBeLessThanOrEqual(5);
  });

  it("takes a blue cast out of a grey picture", () => {
    const cold = ramp(40, 200, [0.95, 1, 1.08]);
    const set = autoAdjustments(cold, NEUTRAL_ADJUSTMENTS);
    const apart = ([r, g, b]: [number, number, number]) => Math.abs(r - g) + Math.abs(b - g);

    // Warmer is a positive warmth.
    expect(set.warmth).toBeGreaterThan(0);
    expect(apart(averages(cold, set))).toBeLessThan(apart(averages(cold, NEUTRAL_ADJUSTMENTS)) / 3);
  });

  it("measures a cast only where the picture is near grey, so a red picture stays red", () => {
    const red = new Uint8ClampedArray(64 * 4);
    for (let at = 0; at < red.length; at += 4) red.set([220, 40, 40, 255], at);

    const set = autoAdjustments(red, NEUTRAL_ADJUSTMENTS);

    expect([set.warmth, set.tint]).toEqual([0, 0]);
  });

  it("widens a picture that uses only part of the range, and no more than it needs", () => {
    const flat = ramp(40, 200);
    const set = autoAdjustments(flat, NEUTRAL_ADJUSTMENTS);
    const spread = (adjustments: StudioColorAdjustments) => {
      const [low, , high] = levels(flat, adjustments);
      return high - low;
    };

    expect(spread(NEUTRAL_ADJUSTMENTS)).toBeLessThan(170);
    expect(set.contrast).toBeGreaterThan(0);
    expect(spread(set)).toBeGreaterThanOrEqual(200);
    expect(spread({ ...set, contrast: set.contrast - 1 })).toBeLessThan(200);
  });

  it("goes no further than its reach, however far the picture is off", () => {
    const set = autoAdjustments(ramp(2, 40), NEUTRAL_ADJUSTMENTS);

    // As far as helps, and no further: the last steps can leave the median where it was.
    expect(set.brightness).toBeGreaterThanOrEqual(55);
    expect(set.brightness).toBeLessThanOrEqual(60);
    expect(Math.abs(set.warmth)).toBeLessThanOrEqual(50);
    expect(set.contrast).toBeLessThanOrEqual(50);
  });

  it("reads only what can be seen, and keeps every slider it does not set", () => {
    const seen = ramp(10, 180);
    const both = new Uint8ClampedArray(seen.length * 2);
    both.set(seen);
    // Hidden pixels, strongly tinted, after the seen ones.
    both.set(ramp(10, 180, [0.5, 1, 1.5], 0), seen.length);
    const current = { ...NEUTRAL_ADJUSTMENTS, vignette: 30, sharpness: 20, saturation: -10, warmth: 40 };

    const set = autoAdjustments(both, current);

    expect(set).toEqual({ ...autoAdjustments(seen, current) });
    expect([set.vignette, set.sharpness, set.saturation]).toEqual([30, 20, -10]);
    expect(autoAdjustments(new Uint8ClampedArray(0), current)).toBe(current);
  });
});
