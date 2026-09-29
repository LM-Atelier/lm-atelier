import type { StudioColorAdjustments } from "./types";

/** Light and color adjustments, computed exactly as the server applies them.
 *
 * The canvas shows an adjustment while its sliders move, and an apply asks the
 * server to make the same picture. So this is the server's arithmetic step for
 * step (studio_adjustments.py): one lookup table per channel for warmth, tint,
 * brightness and contrast, then saturation as a mix toward each pixel's grey,
 * with the same roundings. Both copies are checked against the same pixels.
 */

export const NEUTRAL_ADJUSTMENTS: StudioColorAdjustments = {
  brightness: 0,
  contrast: 0,
  saturation: 0,
  warmth: 0,
  tint: 0,
};

/** Each slider runs from -100 to 100, with 0 changing nothing. */
export const ADJUSTMENT_LIMIT = 100;
const NEUTRAL_KELVIN = 6500;
const KELVIN_PER_WARMTH_STEP = 25;
const TINT_REACH = 0.15;

export function isNeutral(adjustments: StudioColorAdjustments): boolean {
  return !(
    adjustments.brightness || adjustments.contrast || adjustments.saturation || adjustments.warmth
    || adjustments.tint
  );
}

function channel(value: number): number {
  return Math.min(255, Math.max(1, value));
}

/** The color of a blackbody at `kelvin`, from Tanner Helland's fit, 0-255. */
function blackbody(kelvin: number): [number, number, number] {
  const t = kelvin / 100;
  const red = t <= 66 ? 255 : 329.698727446 * Math.pow(t - 60, -0.1332047592);
  const green = t <= 66
    ? 99.4708025861 * Math.log(t) - 161.1195681661
    : 288.1221695283 * Math.pow(t - 60, -0.0755148492);
  const blue = t >= 66 ? 255 : t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  return [channel(red), channel(green), channel(blue)];
}

/** The red, green and blue gains that white-balance toward `kelvin`, keeping brightness. */
export function warmthGains(kelvin: number): [number, number, number] {
  const target = blackbody(kelvin);
  const neutral = blackbody(NEUTRAL_KELVIN);
  const gains = [0, 1, 2].map((index) => target[index] / neutral[index]);
  const luma = 0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2];
  return [gains[0] / luma, gains[1] / luma, gains[2] / luma];
}

/** The red, green and blue gains that move the picture toward magenta or green, keeping brightness. */
export function tintGains(tint: number): [number, number, number] {
  const shift = TINT_REACH * tint / ADJUSTMENT_LIMIT;
  const gains = [1 + shift, 1 - shift, 1 + shift];
  const luma = 0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2];
  return [gains[0] / luma, gains[1] / luma, gains[2] / luma];
}

function rounded(value: number): number {
  return Math.min(255, Math.max(0, Math.floor(value + 0.5)));
}

/** The red, green and blue lookup tables for warmth, tint, brightness and contrast. */
export function channelTables(adjustments: StudioColorAdjustments): [Uint8Array, Uint8Array, Uint8Array] {
  const warmth = warmthGains(NEUTRAL_KELVIN - KELVIN_PER_WARMTH_STEP * adjustments.warmth);
  const tint = tintGains(adjustments.tint);
  const gains = [0, 1, 2].map((index) => warmth[index] * tint[index]);
  const brightness = Math.pow(2, adjustments.brightness / ADJUSTMENT_LIMIT);
  const contrast = Math.pow(2, adjustments.contrast / ADJUSTMENT_LIMIT);
  const tables = gains.map((gain) => {
    const table = new Uint8Array(256);
    for (let value = 0; value < 256; value += 1) {
      const lit = value * gain * brightness;
      table[value] = rounded((lit - 127.5) * contrast + 127.5);
    }
    return table;
  });
  return [tables[0], tables[1], tables[2]];
}

/** One channel moved from its pixel's grey by `keep`, as the server's blend does it.
 *
 * The blend takes the factor as a single-precision float and works in single
 * precision, product and sum each rounded to it, then truncates. Both are
 * exact in double precision first, so rounding each to single matches it.
 */
function mixed(grey: number, value: number, keep: number): number {
  const factor = Math.fround(keep);
  const moved = Math.fround(grey + Math.fround(factor * (value - grey)));
  if (factor >= 0 && factor <= 1) return Math.trunc(moved);
  return moved <= 0 ? 0 : moved >= 255 ? 255 : Math.trunc(moved);
}

/** The adjusted copy of RGBA pixels; transparency is kept as it was. */
export function adjustPixels(pixels: Uint8ClampedArray, adjustments: StudioColorAdjustments): Uint8ClampedArray {
  const [red, green, blue] = channelTables(adjustments);
  const keep = 1 + adjustments.saturation / ADJUSTMENT_LIMIT;
  const out = new Uint8ClampedArray(pixels.length);
  for (let index = 0; index < pixels.length; index += 4) {
    const r = red[pixels[index]];
    const g = green[pixels[index + 1]];
    const b = blue[pixels[index + 2]];
    if (adjustments.saturation === 0) {
      out[index] = r;
      out[index + 1] = g;
      out[index + 2] = b;
    } else {
      // The server's grey: integer luma with weights summing to 65536.
      const grey = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
      out[index] = mixed(grey, r, keep);
      out[index + 1] = mixed(grey, g, keep);
      out[index + 2] = mixed(grey, b, keep);
    }
    out[index + 3] = pixels[index + 3];
  }
  return out;
}
