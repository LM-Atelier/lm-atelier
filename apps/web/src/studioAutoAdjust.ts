import { channelTables, colorGains } from "./studioAdjustments";
import type { StudioColorAdjustments } from "./types";

/** Light and color set in one press, from the picture itself.
 *
 * Auto moves the sliders a person would move, so what it does is shown on the
 * picture, can be changed before Apply, and is made by the server's own
 * arithmetic. It corrects three things and leaves every other slider as it
 * was: a color cast, measured on the picture's near-grey middle tones and
 * taken away with warmth and tint; exposure, bringing the middle of the
 * picture's brightness toward a middle grey; and contrast, widening a picture
 * that uses only part of the range. Each is judged through the sliders' own
 * tables and held to a moderate reach, so a balanced picture is left as it
 * was, and a picture that is colored on purpose keeps most of its color.
 */

/** Pixels read, at most: a large picture is sampled evenly, which is plenty for a histogram. */
const MAX_SAMPLES = 65_536;
/** How far Auto moves warmth and tint, exposure, and contrast, out of 100. */
const CAST_REACH = 50;
const EXPOSURE_REACH = 60;
const CONTRAST_REACH = 50;
/** The middle grey the median brightness is brought toward, and how near counts as there already. */
const MIDDLE = 118;
const MIDDLE_TOLERANCE = 10;
/** The brightness spread, second to ninety-eighth percentile, that counts as narrow, and the one it is widened to. */
const NARROW = 170;
const SPREAD = 200;
/** A pixel counts toward the cast when its channels lie this close together, away from black and white. */
const NEAR_GREY = 64;
/** The share of the picture that must be near grey before a cast is measured at all. */
const ENOUGH_GREY = 0.02;

type Samples = { red: Uint8Array; green: Uint8Array; blue: Uint8Array };

/** The picture's own light and color, set by Auto; every other slider as it was. */
export function autoAdjustments(pixels: Uint8ClampedArray, current: StudioColorAdjustments): StudioColorAdjustments {
  const samples = sampled(pixels);
  if (samples.red.length === 0) return current;
  const [warmth, tint] = castCorrection(samples);
  const tone = { ...current, warmth, tint, brightness: 0, contrast: 0 };
  const brightness = exposure(samples, tone);
  const contrast = widening(samples, { ...tone, brightness });
  return { ...current, warmth, tint, brightness, contrast };
}

/** The opaque pixels, evenly spaced, at most MAX_SAMPLES of them. */
function sampled(pixels: Uint8ClampedArray): Samples {
  const total = Math.floor(pixels.length / 4);
  const stride = Math.max(1, Math.ceil(total / MAX_SAMPLES));
  const red: number[] = [];
  const green: number[] = [];
  const blue: number[] = [];
  for (let pixel = 0; pixel < total; pixel += stride) {
    const at = pixel * 4;
    // A mostly transparent pixel is not part of what anybody sees.
    if (pixels[at + 3] < 128) continue;
    red.push(pixels[at]);
    green.push(pixels[at + 1]);
    blue.push(pixels[at + 2]);
  }
  return { red: Uint8Array.from(red), green: Uint8Array.from(green), blue: Uint8Array.from(blue) };
}

/** The warmth and tint that bring the near-grey middle tones back to grey, within reach.
 *
 * Grey things show a cast plainly and colored things hide it, so only pixels
 * whose channels are close together, and neither near black nor near white,
 * are measured. Too few of them and there is nothing to measure. The gains
 * scale each channel, so a candidate is judged by how far apart the scaled
 * averages still are; the smaller change wins a tie.
 */
function castCorrection({ red, green, blue }: Samples): [number, number] {
  let count = 0;
  const sums = [0, 0, 0];
  for (let index = 0; index < red.length; index += 1) {
    const [r, g, b] = [red[index], green[index], blue[index]];
    const high = Math.max(r, g, b);
    const low = Math.min(r, g, b);
    if (high - low > NEAR_GREY || high > 235 || low < 20) continue;
    count += 1;
    sums[0] += r;
    sums[1] += g;
    sums[2] += b;
  }
  if (count === 0 || count < ENOUGH_GREY * red.length) return [0, 0];
  let best: [number, number] = [0, 0];
  let bestScore = Infinity;
  for (let warmth = -CAST_REACH; warmth <= CAST_REACH; warmth += 1) {
    for (let tint = -CAST_REACH; tint <= CAST_REACH; tint += 1) {
      const gains = colorGains(warmth, tint);
      const [r, g, b] = [0, 1, 2].map((channel) => gains[channel] * sums[channel] / count);
      const score = (r - g) ** 2 + (b - g) ** 2;
      const nearer = score < bestScore - 1e-9
        || (Math.abs(score - bestScore) <= 1e-9 && Math.abs(warmth) + Math.abs(tint) < Math.abs(best[0]) + Math.abs(best[1]));
      if (nearer) {
        best = [warmth, tint];
        bestScore = score;
      }
    }
  }
  return best;
}

/** The brightness of each sample under these settings, as a histogram of 256 levels. */
function brightnessHistogram({ red, green, blue }: Samples, adjustments: StudioColorAdjustments): Uint32Array {
  const [r, g, b] = channelTables(adjustments);
  const histogram = new Uint32Array(256);
  for (let index = 0; index < red.length; index += 1) {
    const level = 0.2126 * r[red[index]] + 0.7152 * g[green[index]] + 0.0722 * b[blue[index]];
    histogram[Math.min(255, Math.round(level))] += 1;
  }
  return histogram;
}

/** The level below which this share of the samples lie. */
function percentile(histogram: Uint32Array, share: number): number {
  const total = histogram.reduce((sum, count) => sum + count, 0);
  const wanted = share * total;
  let seen = 0;
  for (let level = 0; level < 256; level += 1) {
    seen += histogram[level];
    if (seen >= wanted) return level;
  }
  return 255;
}

/** The brightness that brings the median to middle grey, or none when it is near there already. */
function exposure(samples: Samples, tone: StudioColorAdjustments): number {
  const median = percentile(brightnessHistogram(samples, tone), 0.5);
  if (Math.abs(median - MIDDLE) <= MIDDLE_TOLERANCE) return 0;
  let best = 0;
  let bestDistance = Math.abs(median - MIDDLE);
  const step = median < MIDDLE ? 1 : -1;
  for (let brightness = step; Math.abs(brightness) <= EXPOSURE_REACH; brightness += step) {
    const distance = Math.abs(percentile(brightnessHistogram(samples, { ...tone, brightness }), 0.5) - MIDDLE);
    if (distance < bestDistance) {
      best = brightness;
      bestDistance = distance;
    }
    if (distance <= MIDDLE_TOLERANCE / 2) break;
  }
  return best;
}

/** The least contrast that widens a narrow picture to SPREAD, within reach; none for a wide one. */
function widening(samples: Samples, lit: StudioColorAdjustments): number {
  const spread = (contrast: number) => {
    const histogram = brightnessHistogram(samples, { ...lit, contrast });
    return percentile(histogram, 0.98) - percentile(histogram, 0.02);
  };
  if (spread(0) >= NARROW) return 0;
  for (let contrast = 1; contrast <= CONTRAST_REACH; contrast += 1) {
    if (spread(contrast) >= SPREAD) return contrast;
  }
  return CONTRAST_REACH;
}
