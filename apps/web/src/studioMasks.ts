/** The studio's mask model: pure typed-array rasters, no canvas required.
 *
 * A mask is per-pixel coverage 0..255 at image resolution. Keeping it as a
 * `Uint8Array` instead of an OffscreenCanvas makes every operation exact,
 * synchronous, and fully testable; canvas enters only at the edges - the
 * tint overlay reads `toAlphaImageData`, and the upload path encodes the
 * same bytes to PNG. Tools stamp into the raster through these functions
 * exclusively, which is what keeps undo trivial and behavior provable.
 */

export type MaskRaster = {
  readonly width: number;
  readonly height: number;
  /** Row-major coverage, 0 = untouched, 255 = fully selected. */
  readonly data: Uint8Array;
};

const MAX_DIMENSION = 8192;

export function createMask(width: number, height: number): MaskRaster {
  if (
    !Number.isInteger(width)
    || !Number.isInteger(height)
    || width < 1
    || height < 1
    || width > MAX_DIMENSION
    || height > MAX_DIMENSION
  ) {
    throw new Error("mask dimensions must be integers within the supported canvas size");
  }
  return { width, height, data: new Uint8Array(width * height) };
}

export function cloneMask(mask: MaskRaster): MaskRaster {
  return { width: mask.width, height: mask.height, data: new Uint8Array(mask.data) };
}

export function isEmpty(mask: MaskRaster): boolean {
  return mask.data.every((value) => value === 0);
}

/** Fraction of pixels with any coverage; the panel warns below 0.5%. */
export function coverage(mask: MaskRaster): number {
  let touched = 0;
  for (const value of mask.data) if (value > 0) touched += 1;
  return touched / mask.data.length;
}

/** Stamp one hard-edged filled circle; the brush's atom. */
export function stampCircle(
  mask: MaskRaster,
  cx: number,
  cy: number,
  radius: number,
  value = 255,
): void {
  const r = Math.max(0.5, radius);
  const x0 = Math.max(0, Math.floor(cx - r));
  const x1 = Math.min(mask.width - 1, Math.ceil(cx + r));
  const y0 = Math.max(0, Math.floor(cy - r));
  const y1 = Math.min(mask.height - 1, Math.ceil(cy + r));
  const rSquared = r * r;
  for (let y = y0; y <= y1; y += 1) {
    const dy = y + 0.5 - cy;
    for (let x = x0; x <= x1; x += 1) {
      const dx = x + 0.5 - cx;
      if (dx * dx + dy * dy <= rSquared) {
        mask.data[y * mask.width + x] = value;
      }
    }
  }
}

/** Stamp along a segment at radius/2 spacing - the standard brush stroke. */
export function strokeSegment(
  mask: MaskRaster,
  from: { x: number; y: number },
  to: { x: number; y: number },
  radius: number,
  value = 255,
): void {
  const distance = Math.hypot(to.x - from.x, to.y - from.y);
  const spacing = Math.max(0.5, radius / 2);
  const steps = Math.max(1, Math.ceil(distance / spacing));
  for (let step = 0; step <= steps; step += 1) {
    const t = step / steps;
    stampCircle(mask, from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t, radius, value);
  }
}

export function fillRect(
  mask: MaskRaster,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  value = 255,
): void {
  const left = Math.max(0, Math.floor(Math.min(x0, x1)));
  const right = Math.min(mask.width - 1, Math.ceil(Math.max(x0, x1)) - 1);
  const top = Math.max(0, Math.floor(Math.min(y0, y1)));
  const bottom = Math.min(mask.height - 1, Math.ceil(Math.max(y0, y1)) - 1);
  for (let y = top; y <= bottom; y += 1) {
    mask.data.fill(value, y * mask.width + left, y * mask.width + right + 1);
  }
}

/** Scanline fill for the lasso's closed polygon (even-odd rule). */
export function fillPolygon(
  mask: MaskRaster,
  points: ReadonlyArray<{ x: number; y: number }>,
  value = 255,
): void {
  if (points.length < 3) return;
  const top = Math.max(0, Math.floor(Math.min(...points.map((p) => p.y))));
  const bottom = Math.min(mask.height - 1, Math.ceil(Math.max(...points.map((p) => p.y))));
  for (let y = top; y <= bottom; y += 1) {
    const scanY = y + 0.5;
    const crossings: number[] = [];
    for (let index = 0; index < points.length; index += 1) {
      const a = points[index];
      const b = points[(index + 1) % points.length];
      if (a.y <= scanY === b.y <= scanY) continue;
      crossings.push(a.x + ((scanY - a.y) / (b.y - a.y)) * (b.x - a.x));
    }
    crossings.sort((left, right) => left - right);
    for (let pair = 0; pair + 1 < crossings.length; pair += 2) {
      const left = Math.max(0, Math.round(crossings[pair]));
      const right = Math.min(mask.width - 1, Math.round(crossings[pair + 1]) - 1);
      if (right >= left) mask.data.fill(value, y * mask.width + left, y * mask.width + right + 1);
    }
  }
}

export function invert(mask: MaskRaster): void {
  for (let index = 0; index < mask.data.length; index += 1) {
    mask.data[index] = 255 - mask.data[index];
  }
}

/** Two box passes approximate a gaussian; enough for selection feathering. */
export function feather(mask: MaskRaster, radiusPx: number): void {
  const radius = Math.floor(radiusPx);
  if (radius < 1) return;
  boxBlur(mask, radius);
  boxBlur(mask, radius);
}

function boxBlur(mask: MaskRaster, radius: number): void {
  const { width, height, data } = mask;
  const window = radius * 2 + 1;
  const row = new Uint8Array(width);
  for (let y = 0; y < height; y += 1) {
    let sum = 0;
    for (let x = -radius; x <= radius; x += 1) {
      sum += data[y * width + clampIndex(x, width)];
    }
    for (let x = 0; x < width; x += 1) {
      row[x] = Math.round(sum / window);
      sum -= data[y * width + clampIndex(x - radius, width)];
      sum += data[y * width + clampIndex(x + radius + 1, width)];
    }
    data.set(row, y * width);
  }
  const column = new Uint8Array(height);
  for (let x = 0; x < width; x += 1) {
    let sum = 0;
    for (let y = -radius; y <= radius; y += 1) {
      sum += data[clampIndex(y, height) * width + x];
    }
    for (let y = 0; y < height; y += 1) {
      column[y] = Math.round(sum / window);
      sum -= data[clampIndex(y - radius, height) * width + x];
      sum += data[clampIndex(y + radius + 1, height) * width + x];
    }
    for (let y = 0; y < height; y += 1) data[y * width + x] = column[y];
  }
}

function clampIndex(value: number, size: number): number {
  return value < 0 ? 0 : value >= size ? size - 1 : value;
}

/** RGBA bytes with the mask as alpha, for the tint overlay or PNG export. */
export function toAlphaImageData(
  mask: MaskRaster,
  rgb: readonly [number, number, number] = [255, 255, 255],
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask.data.length * 4);
  for (let index = 0; index < mask.data.length; index += 1) {
    out[index * 4] = rgb[0];
    out[index * 4 + 1] = rgb[1];
    out[index * 4 + 2] = rgb[2];
    out[index * 4 + 3] = mask.data[index];
  }
  return out;
}

/** A bounded undo ring; tools push on stroke end, never mid-gesture. */
export class MaskHistory {
  private past: MaskRaster[] = [];
  private future: MaskRaster[] = [];

  constructor(private readonly depth = 16) {}

  push(mask: MaskRaster): void {
    this.past.push(cloneMask(mask));
    if (this.past.length > this.depth) this.past.shift();
    this.future = [];
  }

  undo(current: MaskRaster): MaskRaster | null {
    const previous = this.past.pop();
    if (!previous) return null;
    this.future.push(cloneMask(current));
    return previous;
  }

  redo(current: MaskRaster): MaskRaster | null {
    const next = this.future.pop();
    if (!next) return null;
    this.past.push(cloneMask(current));
    return next;
  }

  get canUndo(): boolean {
    return this.past.length > 0;
  }

  get canRedo(): boolean {
    return this.future.length > 0;
  }
}
