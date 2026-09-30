import type { StudioColorAdjustments } from "./types";

/** Light and color adjustments, computed exactly as the server applies them.
 *
 * The canvas shows an adjustment while its sliders move, and an apply asks the
 * server to make the same picture. So this is the server's arithmetic step for
 * step (studio_adjustments.py): one lookup table per channel for shadows,
 * highlights, warmth, tint, brightness and contrast, then saturation as a mix
 * toward each pixel's grey, then vibrance as the same mix kept in proportion to
 * how muted each pixel is, then sharpness as a mix away from a softened copy of
 * the picture, then the vignette, a mix toward black or white that grows toward
 * the corners, and last grain, the same small step up or down on all three
 * channels of each pixel, taken from a fixed tile of noise, with the same
 * roundings. Both copies are checked against the same pixels.
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
  vibrance: 0,
  vignette: 0,
  grain: 0,
};

/** Each slider runs from -100 to 100, with 0 changing nothing. */
export const ADJUSTMENT_LIMIT = 100;
const NEUTRAL_KELVIN = 6500;
const KELVIN_PER_WARMTH_STEP = 25;
const TINT_REACH = 0.15;
/** How far toward black, or toward white, the vignette takes the corners at either end. */
const VIGNETTE_REACH = 0.8;
/** Where the vignette begins and where it is whole, as the squared distance
 * from the middle, on which the middle of each edge is 1 and each corner 2. */
const VIGNETTE_START = 0.25;
const VIGNETTE_FULL = 1.75;
/** How finely that squared distance is counted: this many steps to the middle of an edge. */
const VIGNETTE_STEPS = 4096;
/** How far the grain slider at its top moves a pixel up or down, as a share of the grain's own spread. */
const GRAIN_REACH = 0.25;
/** The grain repeats every this many pixels across and down. */
const GRAIN_TILE = 256;

export function isNeutral(adjustments: StudioColorAdjustments): boolean {
  return !(
    adjustments.brightness || adjustments.contrast || adjustments.highlights || adjustments.shadows
    || adjustments.saturation || adjustments.warmth || adjustments.tint || adjustments.sharpness
    || adjustments.vibrance || adjustments.vignette || adjustments.grain
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

/** The red, green and blue gains the warmth and tint sliders make together. */
export function colorGains(warmth: number, tint: number): [number, number, number] {
  const warm = warmthGains(NEUTRAL_KELVIN - KELVIN_PER_WARMTH_STEP * warmth);
  const tinted = tintGains(tint);
  return [warm[0] * tinted[0], warm[1] * tinted[1], warm[2] * tinted[2]];
}

/** The red, green and blue lookup tables for tone, color, brightness and contrast. */
export function channelTables(adjustments: StudioColorAdjustments): [Uint8Array, Uint8Array, Uint8Array] {
  const gains = colorGains(adjustments.warmth, adjustments.tint);
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

/** `toward` laid over `value` by `weight` out of 255, rounded as the server's composite rounds it. */
function composited(value: number, toward: number, weight: number): number {
  const sum = value * (255 - weight) + toward * weight + 128;
  return ((sum >> 8) + sum) >> 8;
}

/** The server's grey: integer luma with weights summing to 65536. */
function greyOf(r: number, g: number, b: number): number {
  return (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
}

/** Each column's or row's squared distance from the middle, in whole steps.
 *
 * Measured from the centre of each pixel as a share of half the width or
 * height, squared and floored, as the server counts it. Worked in exact
 * integers, since a long enough side would take the product past what a
 * float holds exactly.
 */
function distanceSteps(size: number): number[] {
  const area = BigInt(size) * BigInt(size);
  return Array.from({ length: size }, (_, index) => {
    const offset = BigInt(2 * index + 1 - size);
    return Number((BigInt(VIGNETTE_STEPS) * offset * offset) / area);
  });
}

/** How much of the vignette each squared distance takes, from 0 to 255.
 *
 * None up to the start, all of it from the full distance, and a smooth step
 * between; the same sums, products and divisions as the server, in the same
 * order, so every weight comes out the same.
 */
const VIGNETTE_WEIGHTS: Uint8Array = (() => {
  const weights = new Uint8Array(2 * VIGNETTE_STEPS);
  for (let step = 0; step < weights.length; step += 1) {
    let share = (step / VIGNETTE_STEPS - VIGNETTE_START) / (VIGNETTE_FULL - VIGNETTE_START);
    share = Math.min(1, Math.max(0, share));
    weights[step] = Math.floor(255 * share * share * (3 - 2 * share) + 0.5);
  }
  return weights;
})();

/** The edges darkened above zero, or lightened below, most at the corners. */
function vignetted(pixels: Uint8ClampedArray, width: number, vignette: number): Uint8ClampedArray {
  const reach = Math.abs(vignette) / ADJUSTMENT_LIMIT * VIGNETTE_REACH;
  const toward = new Uint8Array(256);
  for (let value = 0; value < 256; value += 1) {
    toward[value] = vignette > 0 ? rounded(value * (1 - reach)) : rounded(value + (255 - value) * reach);
  }
  const height = pixels.length / 4 / width;
  const across = distanceSteps(width);
  const down = distanceSteps(height);
  const out = new Uint8ClampedArray(pixels);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const weight = VIGNETTE_WEIGHTS[down[y] + across[x]];
      const at = (y * width + x) * 4;
      for (let channel = at; channel < at + 3; channel += 1) {
        out[channel] = composited(pixels[channel], toward[pixels[channel]], weight);
      }
    }
  }
  return out;
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

/** The grain at one place of its tile, from 0 to 255, as the server works it out.
 *
 * The same whole-number hash in 32 bits: Math.imul and unsigned shifts give
 * exactly the products and shifts the server takes modulo 2 to the 32. The
 * mean of two of its bytes makes middling levels commoner than extremes.
 */
export function grainLevel(x: number, y: number): number {
  let mixed = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + 0x9e3779b9) >>> 0;
  mixed = Math.imul(mixed ^ (mixed >>> 13), 1274126177) >>> 0;
  mixed = (mixed ^ (mixed >>> 16)) >>> 0;
  return ((mixed & 255) + ((mixed >>> 8) & 255)) >> 1;
}

let grainTileLevels: Uint8Array | null = null;

/** The tile of grain levels, made once, row by row. */
function grainTile(): Uint8Array {
  if (!grainTileLevels) {
    const levels = new Uint8Array(GRAIN_TILE * GRAIN_TILE);
    for (let y = 0; y < GRAIN_TILE; y += 1) {
      for (let x = 0; x < GRAIN_TILE; x += 1) levels[y * GRAIN_TILE + x] = grainLevel(x, y);
    }
    grainTileLevels = levels;
  }
  return grainTileLevels;
}

/** How far each grain level moves a pixel at this slider value, plus 128. */
export function grainOffsets(grain: number): Uint8Array {
  const reach = grain / ADJUSTMENT_LIMIT * GRAIN_REACH;
  return Uint8Array.from({ length: 256 }, (_, level) => rounded(128 + (level - 128) * reach));
}

/** Grain laid over the picture: each pixel moved the same way on all three channels. */
function grained(pixels: Uint8ClampedArray, width: number, grain: number): Uint8ClampedArray {
  const tile = grainTile();
  const offsets = grainOffsets(grain);
  const out = new Uint8ClampedArray(pixels);
  for (let index = 0; index < pixels.length; index += 4) {
    const pixel = index / 4;
    const x = (pixel % width) % GRAIN_TILE;
    const y = Math.floor(pixel / width) % GRAIN_TILE;
    // A clipped whole-number add, as the server's is: the offset carries 128.
    const step = offsets[tile[y * GRAIN_TILE + x]] - 128;
    for (let channel = index; channel < index + 3; channel += 1) {
      out[channel] = Math.min(255, Math.max(0, pixels[channel] + step));
    }
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
  const vivid = 1 + adjustments.vibrance / ADJUSTMENT_LIMIT;
  const out = new Uint8ClampedArray(pixels.length);
  for (let index = 0; index < pixels.length; index += 4) {
    let r = red[pixels[index]];
    let g = green[pixels[index + 1]];
    let b = blue[pixels[index + 2]];
    if (adjustments.saturation !== 0) {
      const grey = greyOf(r, g, b);
      r = mixed(grey, r, keep);
      g = mixed(grey, g, keep);
      b = mixed(grey, b, keep);
    }
    if (adjustments.vibrance !== 0) {
      // The saturation mix again, from this pixel's own grey, kept in
      // proportion to how muted the pixel is: its spread taken from white.
      const grey = greyOf(r, g, b);
      const muted = 255 - (Math.max(r, g, b) - Math.min(r, g, b));
      r = composited(r, mixed(grey, r, vivid), muted);
      g = composited(g, mixed(grey, g, vivid), muted);
      b = composited(b, mixed(grey, b, vivid), muted);
    }
    out[index] = r;
    out[index + 1] = g;
    out[index + 2] = b;
    out[index + 3] = pixels[index + 3];
  }
  const sharp = adjustments.sharpness === 0 ? out : sharpened(out, width, adjustments.sharpness);
  const edged = adjustments.vignette === 0 ? sharp : vignetted(sharp, width, adjustments.vignette);
  return adjustments.grain <= 0 ? edged : grained(edged, width, adjustments.grain);
}
