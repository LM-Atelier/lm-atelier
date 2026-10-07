/** Correcting a picture's perspective: four corners placed on the picture.
 *
 * The corners go on the corners of something that should be square, such as
 * a page or the front of a building, and the server makes what lies inside
 * them the whole picture, upright. The size and the check on the corners here
 * are the server's own (studio_local_edits.py), so the panel can say what an
 * apply will make and hold it back from corners the server would refuse.
 */

import type { ImagePoint, PointerTool, ToolPreview } from "./studioTools";
import type { StudioPerspective, StudioPoint } from "./types";

/** The corners in order, from the top left going clockwise. */
export const CORNERS = ["top_left", "top_right", "bottom_right", "bottom_left"] as const;
export type Corner = (typeof CORNERS)[number];

/** The picture's own corners: where the four start, and a correction that changes nothing. */
export function pictureCorners(width: number, height: number): StudioPerspective {
  return {
    top_left: { x: 0, y: 0 },
    top_right: { x: width, y: 0 },
    bottom_right: { x: width, y: height },
    bottom_left: { x: 0, y: height },
  };
}

/** Whether the corners are still the picture's own, so there is nothing to correct. */
export function isUnchanged(corners: StudioPerspective, width: number, height: number): boolean {
  const own = pictureCorners(width, height);
  return CORNERS.every((name) => corners[name].x === own[name].x && corners[name].y === own[name].y);
}

function length(from: StudioPoint, to: StudioPoint): number {
  // The square root of an exact whole number, as the server takes it.
  const across = to.x - from.x;
  const down = to.y - from.y;
  return Math.sqrt(across * across + down * down);
}

/** The corrected picture's size: the longer of each pair of opposite sides, rounded. */
export function correctedSize(corners: StudioPerspective): { width: number; height: number } {
  const { top_left: a, top_right: b, bottom_right: c, bottom_left: d } = corners;
  return {
    width: Math.round(Math.max(length(a, b), length(d, c))),
    height: Math.round(Math.max(length(a, d), length(b, c))),
  };
}

/** Whether the corners go around a four-sided shape in order, clockwise with no side crossing another.
 *
 * Every turn from one side to the next must go the same way, as the server
 * checks, which holds for nothing else.
 */
export function isFrame(corners: StudioPerspective): boolean {
  const points = CORNERS.map((name) => corners[name]);
  return points.every((a, index) => {
    const b = points[(index + 1) % 4];
    const c = points[(index + 2) % 4];
    return (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) > 0;
  });
}

/** Where a point of the corrected picture lies on the source, given as fractions of its width and height.
 *
 * The map from a unit square onto the four corners, after Heckbert, which is
 * the map the server resamples through, so a line drawn with it lands where
 * the correction will put a straight line.
 */
export function projected(corners: StudioPerspective, across: number, down: number): ImagePoint {
  const { top_left: p0, top_right: p1, bottom_right: p2, bottom_left: p3 } = corners;
  const sumX = p0.x - p1.x + p2.x - p3.x;
  const sumY = p0.y - p1.y + p2.y - p3.y;
  const determinant = (p1.x - p2.x) * (p3.y - p2.y) - (p3.x - p2.x) * (p1.y - p2.y);
  const g = (sumX * (p3.y - p2.y) - (p3.x - p2.x) * sumY) / determinant;
  const h = ((p1.x - p2.x) * sumY - sumX * (p1.y - p2.y)) / determinant;
  const scale = g * across + h * down + 1;
  return {
    x: ((p1.x - p0.x + g * p1.x) * across + (p3.x - p0.x + h * p3.x) * down + p0.x) / scale,
    y: ((p1.y - p0.y + g * p1.y) * across + (p3.y - p0.y + h * p3.y) * down + p0.y) / scale,
  };
}

/** The lines dividing the corrected picture in thirds, where they fall on the source. */
export function thirds(corners: StudioPerspective): Array<[ImagePoint, ImagePoint]> {
  return [1 / 3, 2 / 3].flatMap((part): Array<[ImagePoint, ImagePoint]> => [
    [projected(corners, part, 0), projected(corners, part, 1)],
    [projected(corners, 0, part), projected(corners, 1, part)],
  ]);
}

function nearest(corners: StudioPerspective, point: ImagePoint): Corner {
  const distance = (name: Corner) => Math.hypot(corners[name].x - point.x, corners[name].y - point.y);
  return CORNERS.reduce((best, name) => (distance(name) < distance(best) ? name : best));
}

/** Dragging the four corners on the canvas.
 *
 * A press takes the corner nearest to it, and a drag moves that corner as far
 * as the pointer moves, so a corner never jumps to where the press landed; the
 * canvas's keyboard caret moves it the same way. The selection is never
 * touched, and the corners are reported once, when the drag ends.
 */
export class PerspectiveTool implements PointerTool {
  readonly appliesWhileMoving = false;
  private held: { corner: Corner; from: ImagePoint; start: StudioPoint } | null = null;
  private hover: ImagePoint | null = null;

  constructor(
    private corners: StudioPerspective,
    private readonly size: { width: number; height: number },
    private readonly onChange: (corners: StudioPerspective) => void,
  ) {}

  down(point: ImagePoint): void {
    const corner = nearest(this.corners, point);
    this.held = { corner, from: point, start: this.corners[corner] };
  }

  move(point: ImagePoint): void {
    this.hover = point;
    if (this.held) this.corners = { ...this.corners, [this.held.corner]: this.placed(this.held, point) };
  }

  up(point: ImagePoint): boolean {
    if (!this.held) return false;
    this.corners = { ...this.corners, [this.held.corner]: this.placed(this.held, point) };
    this.held = null;
    this.onChange(this.corners);
    // The selection is as it was, so there is no stroke to finish.
    return false;
  }

  cancel(): void {
    if (this.held) this.corners = { ...this.corners, [this.held.corner]: this.held.start };
    this.held = null;
  }

  preview(): ToolPreview {
    const active = this.held?.corner ?? (this.hover ? nearest(this.corners, this.hover) : null);
    return {
      kind: "corners",
      corners: this.corners,
      active,
      lines: isFrame(this.corners) ? thirds(this.corners) : [],
    };
  }

  /** Where a held corner goes: as far as the pointer moved, in whole pixels on the picture. */
  private placed(held: { from: ImagePoint; start: StudioPoint }, point: ImagePoint): StudioPoint {
    return {
      x: Math.min(this.size.width, Math.max(0, Math.round(held.start.x + point.x - held.from.x))),
      y: Math.min(this.size.height, Math.max(0, Math.round(held.start.y + point.y - held.from.y))),
    };
  }
}
