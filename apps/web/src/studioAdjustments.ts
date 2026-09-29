import type { StudioColorAdjustments } from "./types";

/** Light and color adjustments, computed exactly as the server applies them.
 *
 * The canvas shows an adjustment while its sliders move, and an apply asks the
 * server to make the same picture. So this is the server's arithmetic step for
 * step (studio_adjustments.py): one lookup table per channel for shadows,
 * highlights, warmth, tint, brightness and contrast, then saturation as a mix
 * toward each pixel's grey, then sharpness as a mix away from a softened copy
 * of the picture, with the same roundings. Both copies are checked against the
 * same pixels.
 */

export const NEUTRAL_ADJUSTMENTS: StudioColorAdjustments = {
  brightness: 0,
  contrast: 0,
  highlights: 0,
  shadows: 0,
  saturation: 0,
  warmth: 0,
  tint: 0,
  sharpness: 0,
};

/** Each slider runs from -100 to 100, with 0 changing nothing. */
export const ADJUSTMENT_LIMIT = 100;
const NEUTRAL_KELVIN = 6500;
const KELVIN_PER_WARMTH_STEP = 25;
const TINT_REACH = 0.15;

export function isNeutral(adjustments: StudioColorAdjustments): boolean {
  return !(
    adjustments.brightness || adjustments.contrast || adjustments.highlights || adjustments.shadows
    || adjustments.saturation || adjustments.warmth || adjustments.tint || adjustments.sharpness
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

/** A level moved by the shadows and highlights sliders, each from -1 to 1.
 *
 * Taking the level as a share of white, shadows moves it by the share times
 * the square of what is left above it, and highlights by the square of the
 * share times what is left. Black and white stay where they are, and no two
 * levels swap places. Written as the server writes it, so the two agree to the
 * last bit.
 */
export function toned(value: number, shadows: number, highlights: number): number {
  const share = value / 255;
  const left = 1 - share;
  return value + 255 * (shadows * share * left * left + highlights * share * share * left);
}

/** The red, green and blue lookup tables for tone, color, brightness and contrast. */
export function channelTables(adjustments: StudioColorAdjustments): [Uint8Array, Uint8Array, Uint8Array] {
  const warmth = warmthGains(NEUTRAL_KELVIN - KELVIN_PER_WARMTH_STEP * adjustments.warmth);
  const tint = tintGains(adjustments.tint);
  const gains = [0, 1, 2].map((index) => warmth[index] * tint[index]);
  const brightness = Math.pow(2, adjustments.brightness / ADJUSTMENT_LIMIT);
  const contrast = Math.pow(2, adjustments.contrast / ADJUSTMENT_LIMIT);
  const shadows = adjustments.shadows / ADJUSTMENT_LIMIT;
  const highlights = adjustments.highlights / ADJUSTMENT_LIMIT;
  const tables = gains.map((gain) => {
    const table = new Uint8Array(256);
    for (let value = 0; value < 256; value += 1) {
      const lit = toned(value, shadows, highlights) * gain * brightness;
      table[value] = rounded((lit - 127.5) * contrast + 127.5);
    }
    return table;
  });
  return [tables[0], tables[1], tables[2]];
}

/** One channel moved from `base` by `keep`, as the server's blend does it.
 *
 * The base is the pixel's grey for saturation and the softened copy for
 * sharpness. The blend takes the factor as a single-precision float and works
 * in single precision, product and sum each rounded to it, then truncates.
 * Both are exact in double precision first, so rounding each to single
 * matches it.
 */
function mixed(base: number, value: number, keep: number): number {
  const factor = Math.fround(keep);
  const moved = Math.fround(base + Math.fround(factor * (value - base)));
  if (factor >= 0 && factor <= 1) return Math.trunc(moved);
  return moved <= 0 ? 0 : moved >= 255 ? 255 : Math.trunc(moved);
}

/** A color scaled by its opacity, rounded as the server's conversion rounds it. */
function weighted(value: number, alpha: number): number {
  const product = value * alpha + 128;
  return ((product >> 8) + product) >> 8;
}

/** Opacity-weighted pixels softened as the server's kernel softens them.
 *
 * Each channel becomes its neighborhood of nine, weighted 1-2-1 each way,
 * over 16 and rounded half up. The server sums in single precision, but with
 * weights over 16 every sum is exact, so these integers give its answer.
 * The kernel leaves the picture's own edge, and any picture narrower or
 * shorter than three pixels, as it was.
 */
function softened(pixels: Uint8ClampedArray, width: number, height: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(pixels);
  if (width < 3 || height < 3) return out;
  const row = width * 4;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      for (let at = y * row + x * 4, end = at + 4; at < end; at += 1) {
        const above = at - row;
        const below = at + row;
        const sum = pixels[above - 4] + 2 * pixels[above] + pixels[above + 4]
          + 2 * pixels[at - 4] + 4 * pixels[at] + 2 * pixels[at + 4]
          + pixels[below - 4] + 2 * pixels[below] + pixels[below + 4];
        out[at] = (sum + 8) >> 4;
      }
    }
  }
  return out;
}

/** Each channel moved toward or away from a softened copy by the sharpness slider.
 *
 * The copy is made from each color weighted by its opacity and divided back
 * out, as the server does, so a color hidden under a transparent pixel does
 * not bleed into the visible ones beside it. Transparency is kept as it was.
 */
function sharpened(pixels: Uint8ClampedArray, width: number, sharpness: number): Uint8ClampedArray {
  const premultiplied = new Uint8ClampedArray(pixels.length);
  for (let index = 0; index < pixels.length; index += 4) {
    const alpha = pixels[index + 3];
    premultiplied[index] = weighted(pixels[index], alpha);
    premultiplied[index + 1] = weighted(pixels[index + 1], alpha);
    premultiplied[index + 2] = weighted(pixels[index + 2], alpha);
    premultiplied[index + 3] = alpha;
  }
  const soft = softened(premultiplied, width, pixels.length / 4 / width);
  const keep = 1 + sharpness / ADJUSTMENT_LIMIT;
  const out = new Uint8ClampedArray(pixels.length);
  for (let index = 0; index < pixels.length; index += 4) {
    const alpha = soft[index + 3];
    for (let channel = index; channel < index + 3; channel += 1) {
      // Divided back out as the server's conversion does, in whole numbers.
      const base = alpha === 0 || alpha === 255
        ? soft[channel]
        : Math.min(255, Math.floor((255 * soft[channel]) / alpha));
      out[channel] = mixed(base, pixels[channel], keep);
    }
    out[index + 3] = pixels[index + 3];
  }
  return out;
}

/** The adjusted copy of RGBA pixels, `width` to a row; transparency is kept as it was. */
export function adjustPixels(
  pixels: Uint8ClampedArray,
  width: number,
  adjustments: StudioColorAdjustments,
): Uint8ClampedArray {
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
  return adjustments.sharpness === 0 ? out : sharpened(out, width, adjustments.sharpness);
}
