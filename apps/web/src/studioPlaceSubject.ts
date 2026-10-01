/** Replacing a subject by placing the new one where the old one stood.
 *
 * A redraw from two pictures does not put the new subject in the old one's
 * place: it draws the second picture's subject where that picture has it,
 * usually the middle, and can leave the old one standing. So the old subject is
 * removed, and the new one, cut out of its own picture, is placed here instead:
 * as large as fits the old subject's box, centred across it and standing on its
 * bottom edge, with its own soft edge as the only blend. Everything here is pure
 * except the two functions that decode and encode through a canvas.
 */

import { subjectMask } from "./studioBackground";
import { maskBounds, type MaskRaster } from "./studioMasks";
import { readSourcePixels } from "./studioSourcePixels";

export type PictureBox = { left: number; top: number; width: number; height: number };
export type RgbaPixels = { width: number; height: number; data: Uint8ClampedArray };

/** Coverage from here up counts as the subject; the faint halo a cutout leaves does not. */
export const SOLID_COVERAGE = 128;

/** The box around the solid part of a subject's coverage, or null when nothing is solid. */
export function solidBox(mask: MaskRaster): PictureBox | null {
  return maskBounds(mask, SOLID_COVERAGE);
}

/** Where a subject this size goes in the old one's box: as large as fits, centred across, standing on its bottom. */
export function placementIn(box: PictureBox, width: number, height: number): PictureBox {
  const scale = Math.min(box.width / width, box.height / height);
  const placedWidth = Math.max(1, Math.round(width * scale));
  const placedHeight = Math.max(1, Math.round(height * scale));
  return {
    left: box.left + Math.floor((box.width - placedWidth) / 2),
    top: box.top + box.height - placedHeight,
    width: placedWidth,
    height: placedHeight,
  };
}

/** The new subject drawn alone at the picture's size, transparent everywhere else,
 * or null when its cutout holds no solid subject. */
export function placedSubject(
  cutout: RgbaPixels,
  width: number,
  height: number,
  box: PictureBox,
): Uint8ClampedArray | null {
  const own = solidBox(subjectMask(cutout.data, cutout.width, cutout.height));
  if (!own) return null;
  const target = placementIn(box, own.width, own.height);
  const placed = new Uint8ClampedArray(width * height * 4);
  const stepX = own.width / target.width;
  const stepY = own.height / target.height;
  const sample = stepX > 1 || stepY > 1 ? averaged : interpolated;
  for (let y = Math.max(0, target.top); y < Math.min(height, target.top + target.height); y += 1) {
    for (let x = Math.max(0, target.left); x < Math.min(width, target.left + target.width); x += 1) {
      sample(
        cutout,
        own.left + (x - target.left) * stepX,
        own.top + (y - target.top) * stepY,
        stepX,
        stepY,
        placed,
        (y * width + x) * 4,
      );
    }
  }
  return placed;
}

type Sample = (
  source: RgbaPixels,
  x: number,
  y: number,
  stepX: number,
  stepY: number,
  out: Uint8ClampedArray,
  at: number,
) => void;

// Colour is weighted by its own coverage throughout, so a transparent pixel's
// colour, which means nothing, never darkens the subject's soft edge.
function write(out: Uint8ClampedArray, at: number, red: number, green: number, blue: number, alpha: number, weight: number) {
  if (weight <= 0 || alpha <= 0) return;
  out[at] = red / alpha;
  out[at + 1] = green / alpha;
  out[at + 2] = blue / alpha;
  out[at + 3] = alpha / weight;
}

/** Shrinking: every source pixel under the target pixel counts by how much of it is covered. */
const averaged: Sample = (source, x0, y0, stepX, stepY, out, at) => {
  const x1 = x0 + stepX;
  const y1 = y0 + stepY;
  let red = 0;
  let green = 0;
  let blue = 0;
  let alpha = 0;
  let weight = 0;
  for (let sy = Math.floor(y0); sy < Math.min(source.height, Math.ceil(y1)); sy += 1) {
    const coverY = Math.min(y1, sy + 1) - Math.max(y0, sy);
    for (let sx = Math.floor(x0); sx < Math.min(source.width, Math.ceil(x1)); sx += 1) {
      const share = coverY * (Math.min(x1, sx + 1) - Math.max(x0, sx));
      const index = (sy * source.width + sx) * 4;
      const covered = source.data[index + 3] * share;
      red += source.data[index] * covered;
      green += source.data[index + 1] * covered;
      blue += source.data[index + 2] * covered;
      alpha += covered;
      weight += share;
    }
  }
  write(out, at, red, green, blue, alpha, weight);
};

/** Enlarging: the four source pixels around the target pixel's centre, by nearness. */
const interpolated: Sample = (source, x0, y0, stepX, stepY, out, at) => {
  const cx = Math.min(source.width - 1, Math.max(0, x0 + stepX / 2 - 0.5));
  const cy = Math.min(source.height - 1, Math.max(0, y0 + stepY / 2 - 0.5));
  const left = Math.floor(cx);
  const top = Math.floor(cy);
  const right = Math.min(source.width - 1, left + 1);
  const bottom = Math.min(source.height - 1, top + 1);
  const fx = cx - left;
  const fy = cy - top;
  let red = 0;
  let green = 0;
  let blue = 0;
  let alpha = 0;
  for (const [sx, sy, share] of [
    [left, top, (1 - fx) * (1 - fy)],
    [right, top, fx * (1 - fy)],
    [left, bottom, (1 - fx) * fy],
    [right, bottom, fx * fy],
  ] as const) {
    const index = (sy * source.width + sx) * 4;
    const covered = source.data[index + 3] * share;
    red += source.data[index] * covered;
    green += source.data[index + 1] * covered;
    blue += source.data[index + 2] * covered;
    alpha += covered;
  }
  write(out, at, red, green, blue, alpha, 1);
};

/** A cutout's own pixels at its own size, or null when they cannot be read. */
export async function readCutoutPixels(artifactId: string): Promise<RgbaPixels | null> {
  const response = await fetch(`/api/artifacts/${encodeURIComponent(artifactId)}/content`);
  if (!response.ok) return null;
  const bitmap = await createImageBitmap(await response.blob());
  try {
    const data = readSourcePixels(bitmap);
    return data ? { width: bitmap.width, height: bitmap.height, data } : null;
  } finally {
    bitmap.close();
  }
}

/** Encode pixels as a PNG that keeps their transparency, or null when the browser cannot. */
export async function encodeRgbaPng(width: number, height: number, data: Uint8ClampedArray): Promise<Blob | null> {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.putImageData(new ImageData(new Uint8ClampedArray(data), width, height), 0, 0);
  return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), "image/png"));
}
