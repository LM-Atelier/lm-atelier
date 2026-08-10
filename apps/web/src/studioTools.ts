/** Pointer tools: pure gesture state machines over image-space points.
 *
 * A tool receives already-unprojected image coordinates and mutates only
 * the mask raster (through studioMasks) plus its own in-progress gesture
 * state; drawing the live preview is the canvas component's job, reading
 * `preview()`. Keeping tools DOM-free means every gesture is a plain unit
 * test: down, moves, up, assert the raster.
 */

import {
  fillPolygon,
  fillRect,
  strokeSegment,
  type MaskRaster,
} from "./studioMasks";

export type ImagePoint = { x: number; y: number };

export type ToolPreview =
  | { kind: "none" }
  | { kind: "brush-cursor"; center: ImagePoint; radius: number }
  | { kind: "rect"; from: ImagePoint; to: ImagePoint }
  | { kind: "lasso"; points: ImagePoint[] };

export interface PointerTool {
  /** Whether the gesture writes to the mask as it travels.
   *
   * A brush does, so the tint has to be repainted mid-stroke or the selection
   * is invisible until the pointer lifts. A rectangle or a lasso writes
   * nothing until it closes, so repainting mid-gesture would only redraw what
   * is already on screen. */
  readonly appliesWhileMoving: boolean;
  down(point: ImagePoint, viewportScale?: number): void;
  move(point: ImagePoint, viewportScale?: number): void;
  /** Returns true when the gesture changed the mask (history push point). */
  up(point: ImagePoint, viewportScale?: number): boolean;
  /** Abandon the gesture without committing what it has not applied yet.
   *
   * A rectangle or a lasso applies nothing until it closes, so abandoning
   * leaves the mask untouched. A brush applies as it travels, so what is
   * already painted stays and Undo is the way back - the snapshot taken at
   * gesture start is exactly what makes that work. */
  cancel(): void;
  preview(viewportScale?: number): ToolPreview;
}

/** Brush and eraser: identical gesture, inverse stamp value. */
export class BrushTool implements PointerTool {
  readonly appliesWhileMoving = true;
  private last: ImagePoint | null = null;
  private hover: ImagePoint | null = null;
  private touched = false;

  constructor(
    private readonly mask: MaskRaster,
    public radius: number,
    private readonly value: number = 255,
  ) {}

  down(point: ImagePoint, viewportScale = 1): void {
    strokeSegment(this.mask, point, point, this.imageRadius(viewportScale), this.value);
    this.last = point;
    this.touched = true;
  }

  move(point: ImagePoint, viewportScale = 1): void {
    this.hover = point;
    if (!this.last) return;
    strokeSegment(this.mask, this.last, point, this.imageRadius(viewportScale), this.value);
    this.last = point;
  }

  up(point: ImagePoint, viewportScale = 1): boolean {
    if (this.last) {
      strokeSegment(this.mask, this.last, point, this.imageRadius(viewportScale), this.value);
    }
    const changed = this.touched;
    this.last = null;
    this.touched = false;
    return changed;
  }

  cancel(): void {
    this.last = null;
    this.touched = false;
  }

  preview(viewportScale = 1): ToolPreview {
    const center = this.last ?? this.hover;
    return center
      ? { kind: "brush-cursor", center, radius: this.imageRadius(viewportScale) }
      : { kind: "none" };
  }

  /** Convert the user-facing CSS-pixel radius into image pixels. */
  private imageRadius(viewportScale: number): number {
    const scale = Number.isFinite(viewportScale) && viewportScale > 0 ? viewportScale : 1;
    return this.radius / scale;
  }
}

export class RectTool implements PointerTool {
  readonly appliesWhileMoving = false;
  private origin: ImagePoint | null = null;
  private current: ImagePoint | null = null;

  constructor(private readonly mask: MaskRaster) {}

  down(point: ImagePoint): void {
    this.origin = point;
    this.current = point;
  }

  move(point: ImagePoint): void {
    if (this.origin) this.current = point;
  }

  up(point: ImagePoint): boolean {
    if (!this.origin) return false;
    const from = this.origin;
    this.origin = null;
    this.current = null;
    if (Math.abs(point.x - from.x) < 1 || Math.abs(point.y - from.y) < 1) return false;
    fillRect(this.mask, from.x, from.y, point.x, point.y);
    return true;
  }

  cancel(): void {
    this.origin = null;
    this.current = null;
  }

  preview(): ToolPreview {
    return this.origin && this.current
      ? { kind: "rect", from: this.origin, to: this.current }
      : { kind: "none" };
  }
}

export class LassoTool implements PointerTool {
  readonly appliesWhileMoving = false;
  private points: ImagePoint[] = [];

  constructor(
    private readonly mask: MaskRaster,
    /** Minimum image-space distance between recorded vertices. */
    private readonly minSpacing = 2,
  ) {}

  down(point: ImagePoint): void {
    this.points = [point];
  }

  move(point: ImagePoint): void {
    if (this.points.length === 0) return;
    const last = this.points[this.points.length - 1];
    if (Math.hypot(point.x - last.x, point.y - last.y) >= this.minSpacing) {
      this.points.push(point);
    }
  }

  up(point: ImagePoint): boolean {
    if (this.points.length === 0) return false;
    this.points.push(point);
    const closed = this.points;
    this.points = [];
    if (closed.length < 3) return false;
    fillPolygon(this.mask, closed);
    return true;
  }

  cancel(): void {
    this.points = [];
  }

  preview(): ToolPreview {
    return this.points.length > 0
      ? { kind: "lasso", points: [...this.points] }
      : { kind: "none" };
  }
}
