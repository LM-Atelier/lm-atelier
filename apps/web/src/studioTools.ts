/** Pointer tools: pure gesture state machines over image-space points.
 *
 * A tool receives already-unprojected image coordinates and mutates only
 * the mask raster (through studioMasks) plus its own in-progress gesture
 * state; drawing the live preview is the canvas component's job, reading
 * `preview()`. Keeping tools DOM-free means every gesture is a plain unit
 * test: down, moves, up, assert the raster.
 */

import { constrainedCorner, type CropRatio } from "./studioCropShape";
import type { Corner } from "./studioPerspective";
import {
  fillPolygon,
  fillRect,
  fillRegion,
  selectSimilarColor,
  strokeSegment,
  type MaskRaster,
  type MaskRegion,
} from "./studioMasks";
import type { StudioPerspective } from "./types";

export type ImagePoint = { x: number; y: number };

export type ToolPreview =
  | { kind: "none" }
  | { kind: "brush-cursor"; center: ImagePoint; radius: number }
  | { kind: "rect"; from: ImagePoint; to: ImagePoint }
  | { kind: "lasso"; points: ImagePoint[] }
  /** A perspective correction's four corners, the one in hand or nearest, and its thirds. */
  | { kind: "corners"; corners: StudioPerspective; active: Corner | null; lines: Array<[ImagePoint, ImagePoint]> };

export interface PointerTool {
  /** Whether the gesture writes to the mask as it travels.
   *
   * A brush does, so the tint has to be repainted mid-stroke or the selection
   * is invisible until the pointer lifts. A rectangle or a lasso writes
   * nothing until it closes, so repainting mid-gesture would only redraw what
   * is already on screen. */
  readonly appliesWhileMoving: boolean;
  /** "move" for a tool that drags something about rather than marking a place. */
  readonly cursor?: "move";
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
  /** The part of the mask changed since this was last asked, then forgotten.
   *
   * Only a tool that writes while it travels keeps one, so the canvas can
   * repaint that part of the tint on each move rather than all of it.
   */
  takeChanged?(): MaskRegion | null;
}

/** Brush and eraser: identical gesture, inverse stamp value. */
export class BrushTool implements PointerTool {
  readonly appliesWhileMoving = true;
  private last: ImagePoint | null = null;
  private hover: ImagePoint | null = null;
  private touched = false;
  private changed: MaskRegion | null = null;

  constructor(
    private readonly mask: MaskRaster,
    public radius: number,
    private readonly value: number = 255,
  ) {}

  down(point: ImagePoint, viewportScale = 1): void {
    this.stroke(point, point, viewportScale);
    this.last = point;
    this.touched = true;
  }

  move(point: ImagePoint, viewportScale = 1): void {
    this.hover = point;
    if (!this.last) return;
    this.stroke(this.last, point, viewportScale);
    this.last = point;
  }

  up(point: ImagePoint, viewportScale = 1): boolean {
    if (this.last) {
      this.stroke(this.last, point, viewportScale);
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

  takeChanged(): MaskRegion | null {
    const changed = this.changed;
    this.changed = null;
    return changed;
  }

  /** Paint one segment, and add the box it can have touched to what has changed. */
  private stroke(from: ImagePoint, to: ImagePoint, viewportScale: number): void {
    const radius = this.imageRadius(viewportScale);
    strokeSegment(this.mask, from, to, radius, this.value);
    // A pixel past the radius each way, so rounding in the stamp stays inside.
    const left = Math.max(0, Math.floor(Math.min(from.x, to.x) - radius) - 1);
    const top = Math.max(0, Math.floor(Math.min(from.y, to.y) - radius) - 1);
    const right = Math.min(this.mask.width, Math.ceil(Math.max(from.x, to.x) + radius) + 2);
    const bottom = Math.min(this.mask.height, Math.ceil(Math.max(from.y, to.y) + radius) + 2);
    if (right <= left || bottom <= top) return;
    const before = this.changed;
    const union = before
      ? {
          left: Math.min(before.left, left),
          top: Math.min(before.top, top),
          right: Math.max(before.left + before.width, right),
          bottom: Math.max(before.top + before.height, bottom),
        }
      : { left, top, right, bottom };
    this.changed = { left: union.left, top: union.top, width: union.right - union.left, height: union.bottom - union.top };
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

  /** `replace` makes each new box the whole selection, as a crop box is,
   * and `ratio` holds the box to one shape while it is drawn. */
  constructor(
    private readonly mask: MaskRaster,
    private readonly replace = false,
    private readonly ratio: CropRatio | null = null,
  ) {}

  down(point: ImagePoint): void {
    // A held shape counts whole pixels from a whole pixel.
    this.origin = this.ratio ? { x: Math.round(point.x), y: Math.round(point.y) } : point;
    this.current = this.origin;
  }

  move(point: ImagePoint): void {
    if (this.origin) this.current = this.corner(this.origin, point);
  }

  up(pointer: ImagePoint): boolean {
    if (!this.origin) return false;
    const from = this.origin;
    const point = this.corner(from, pointer);
    this.origin = null;
    this.current = null;
    if (Math.abs(point.x - from.x) < 1 || Math.abs(point.y - from.y) < 1) return false;
    if (this.replace) this.mask.data.fill(0);
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

  private corner(from: ImagePoint, point: ImagePoint): ImagePoint {
    return this.ratio ? constrainedCorner(from, point, this.ratio, this.mask) : point;
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

/** A one-click selection: the paint bucket and the magic wand.
 *
 * It selects where the pointer went down, and commits when it lifts, like the
 * rectangle and the lasso. A slip of the pointer between press and release
 * does not move the click, and a keyboard press starts at the caret and
 * finishes on the next press, exactly as the other selections do.
 */
export class ClickSelectTool implements PointerTool {
  readonly appliesWhileMoving = false;
  private pressed: ImagePoint | null = null;

  constructor(private readonly select: (point: ImagePoint) => boolean) {}

  down(point: ImagePoint): void {
    this.pressed = point;
  }

  move(): void {}

  up(): boolean {
    const at = this.pressed;
    this.pressed = null;
    return at ? this.select(at) : false;
  }

  cancel(): void {
    this.pressed = null;
  }

  preview(): ToolPreview {
    return { kind: "none" };
  }
}

/** Fill the evenly covered region under the click; `value` 0 removes it. */
export function paintBucketTool(mask: MaskRaster, value = 255): PointerTool {
  return new ClickSelectTool((point) => fillRegion(mask, point.x, point.y, value));
}

/** Select the connected pixels close in color to the clicked one; `value` 0 removes them. */
export function magicWandTool(
  mask: MaskRaster,
  pixels: Uint8ClampedArray,
  tolerance: number,
  value = 255,
): PointerTool {
  return new ClickSelectTool((point) =>
    selectSimilarColor(mask, pixels, point.x, point.y, tolerance, value));
}
