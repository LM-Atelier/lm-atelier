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

/** Fill the connected region under a point with `value`.
 *
 * Neighbours are the four pixels sharing an edge, and `inRegion` decides
 * which pixels belong, always judged against pixels not yet visited. The walk
 * fills whole runs of a row and remembers only where a run starts on the rows
 * above and below, so even a region covering the largest supported picture
 * never recurses and never queues one entry per pixel. Returns whether any
 * coverage actually changed.
 */
function fillConnected(
  mask: MaskRaster,
  x: number,
  y: number,
  inRegion: (index: number) => boolean,
  value: number,
): boolean {
  const { width, height, data } = mask;
  const column = Math.floor(x);
  const row = Math.floor(y);
  if (!(column >= 0 && row >= 0 && column < width && row < height)) return false;
  const seed = row * width + column;
  if (!inRegion(seed)) return false;
  const visited = new Uint8Array(width * height);
  const pending = [seed];
  let changed = false;
  while (pending.length > 0) {
    const start = pending.pop()!;
    if (visited[start]) continue;
    const rowStart = start - (start % width);
    let left = start;
    while (left > rowStart && !visited[left - 1] && inRegion(left - 1)) left -= 1;
    let right = start;
    while (right < rowStart + width - 1 && !visited[right + 1] && inRegion(right + 1)) right += 1;
    for (let index = left; index <= right; index += 1) {
      visited[index] = 1;
      if (data[index] !== value) {
        data[index] = value;
        changed = true;
      }
    }
    for (const offset of [-width, width]) {
      if (rowStart + offset < 0 || rowStart + offset >= data.length) continue;
      let open = false;
      for (let index = left + offset; index <= right + offset; index += 1) {
        const joins = !visited[index] && inRegion(index);
        if (joins && !open) pending.push(index);
        open = joins;
      }
    }
  }
  return changed;
}

/** The paint bucket: fill the evenly covered region under a point.
 *
 * The region is every connected pixel with exactly the coverage of the one
 * clicked, so clicking inside an outline fills the inside and stops at the
 * outline, and clicking a selected area with `value` 0 removes that area.
 */
export function fillRegion(mask: MaskRaster, x: number, y: number, value = 255): boolean {
  const column = Math.floor(x);
  const row = Math.floor(y);
  if (!(column >= 0 && row >= 0 && column < mask.width && row < mask.height)) return false;
  const seedCoverage = mask.data[row * mask.width + column];
  if (seedCoverage === value) return false;
  return fillConnected(mask, x, y, (index) => mask.data[index] === seedCoverage, value);
}

/** The magic wand: select the connected pixels close in color to the one under a point.
 *
 * `pixels` is the picture as RGBA bytes at the mask's own size; anything else
 * is refused rather than read out of step. Closeness is the largest difference
 * in any color channel or in alpha, so a tolerance of 0 takes only the exact color.
 */
export function selectSimilarColor(
  mask: MaskRaster,
  pixels: Uint8ClampedArray,
  x: number,
  y: number,
  tolerance: number,
  value = 255,
): boolean {
  if (pixels.length !== mask.width * mask.height * 4) {
    throw new Error("the picture's pixels do not match the selection's size");
  }
  const column = Math.floor(x);
  const row = Math.floor(y);
  if (!(column >= 0 && row >= 0 && column < mask.width && row < mask.height)) return false;
  const seed = (row * mask.width + column) * 4;
  const reach = Math.max(0, Math.min(255, Math.round(tolerance)));
  const close = (index: number) => {
    const at = index * 4;
    return Math.abs(pixels[at] - pixels[seed]) <= reach
      && Math.abs(pixels[at + 1] - pixels[seed + 1]) <= reach
      && Math.abs(pixels[at + 2] - pixels[seed + 2]) <= reach
      && Math.abs(pixels[at + 3] - pixels[seed + 3]) <= reach;
  };
  return fillConnected(mask, x, y, close, value);
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

/** The default undo budget: generous for ordinary images, and a hard stop
 * long before a large canvas could exhaust memory. A 32 MP mask is ~32 MB,
 * so a fixed depth of full clones is not a safe bound - bytes are. */
export const DEFAULT_MASK_HISTORY_BYTES = 96 * 1024 * 1024;

/** Run-length encoding of one mask; masks are mostly-uniform by nature, so
 * a stored snapshot is typically orders of magnitude smaller than the
 * raster. Pathological content still encodes correctly, just larger. */
type MaskSnapshot = {
  readonly width: number;
  readonly height: number;
  /** Alternating [value, runLength] pairs over the row-major raster. */
  readonly runs: Uint32Array;
};

export function encodeMask(mask: MaskRaster): MaskSnapshot {
  const runs: number[] = [];
  let value = mask.data[0] ?? 0;
  let length = 0;
  for (const sample of mask.data) {
    if (sample === value) {
      length += 1;
      continue;
    }
    runs.push(value, length);
    value = sample;
    length = 1;
  }
  if (length > 0) runs.push(value, length);
  return { width: mask.width, height: mask.height, runs: Uint32Array.from(runs) };
}

export function decodeMask(snapshot: MaskSnapshot): MaskRaster {
  const mask = createMask(snapshot.width, snapshot.height);
  let offset = 0;
  for (let index = 0; index + 1 < snapshot.runs.length; index += 2) {
    const value = snapshot.runs[index];
    const length = snapshot.runs[index + 1];
    if (value !== 0) mask.data.fill(value, offset, offset + length);
    offset += length;
  }
  return mask;
}

function snapshotBytes(snapshot: MaskSnapshot): number {
  return snapshot.runs.byteLength;
}

/** A byte-budgeted undo ring. Callers snapshot BEFORE the first mutation of
 * a gesture - snapshotting after would store the already-painted raster and
 * make the first undo a no-op. Oldest entries evict when the budget is
 * exceeded, so memory is bounded by bytes rather than by a clone count. */
export class MaskHistory {
  private past: MaskSnapshot[] = [];
  private future: MaskSnapshot[] = [];
  private pastBytes = 0;
  private futureBytes = 0;

  constructor(
    private readonly depth = 64,
    private readonly byteBudget = DEFAULT_MASK_HISTORY_BYTES,
  ) {}

  push(mask: MaskRaster): void {
    this.past.push(encodeMask(mask));
    this.pastBytes += snapshotBytes(this.past[this.past.length - 1]);
    this.future = [];
    this.futureBytes = 0;
    this.evict();
  }

  undo(current: MaskRaster): MaskRaster | null {
    const previous = this.past.pop();
    if (!previous) return null;
    this.pastBytes -= snapshotBytes(previous);
    const snapshot = encodeMask(current);
    this.future.push(snapshot);
    this.futureBytes += snapshotBytes(snapshot);
    this.evict();
    return decodeMask(previous);
  }

  redo(current: MaskRaster): MaskRaster | null {
    const next = this.future.pop();
    if (!next) return null;
    this.futureBytes -= snapshotBytes(next);
    const snapshot = encodeMask(current);
    this.past.push(snapshot);
    this.pastBytes += snapshotBytes(snapshot);
    this.evict();
    return decodeMask(next);
  }

  /** Bytes currently held; the studio surfaces this when a budget bites. */
  get usedBytes(): number {
    return this.pastBytes + this.futureBytes;
  }

  get canUndo(): boolean {
    return this.past.length > 0;
  }

  get canRedo(): boolean {
    return this.future.length > 0;
  }

  private evict(): void {
    while (this.past.length > this.depth) {
      this.pastBytes -= snapshotBytes(this.past.shift()!);
    }
    // The oldest undo step goes first; the redo stack is dropped entirely
    // before any further undo history is sacrificed.
    while (this.usedBytes > this.byteBudget && this.future.length > 0) {
      this.futureBytes -= snapshotBytes(this.future.shift()!);
    }
    while (this.usedBytes > this.byteBudget && this.past.length > 1) {
      this.pastBytes -= snapshotBytes(this.past.shift()!);
    }
  }
}

/** Encode a mask as a PNG whose alpha channel is the coverage.
 *
 * The canvas is the only encoder available in the browser, so this is the
 * one place mask bytes meet a DOM API; everything upstream stays pure.
 * Returns null when no canvas context exists, so callers refuse rather
 * than sending an empty selection.
 */
export async function encodeMaskPng(mask: MaskRaster): Promise<Blob | null> {
  const canvas = document.createElement("canvas");
  canvas.width = mask.width;
  canvas.height = mask.height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.putImageData(
    new ImageData(toAlphaImageData(mask, [255, 255, 255]), mask.width, mask.height),
    0,
    0,
  );
  return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), "image/png"));
}
