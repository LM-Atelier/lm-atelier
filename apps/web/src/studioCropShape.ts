import type { StudioCropBox } from "./types";

/** The shapes a crop box can be held to while it is drawn.
 *
 * Its own file because none of this needs a rendered tree: which shape a box
 * has, where a dragged corner lands, and what box of a new shape fits inside
 * an old one are all plain arithmetic in the picture's own pixels.
 */

export type CropShape = "free" | "picture" | "1:1" | "4:3" | "3:4" | "3:2" | "2:3" | "16:9" | "9:16";

export const CROP_SHAPES: ReadonlyArray<{ shape: CropShape; label: string }> = [
  { shape: "free", label: "Any shape" },
  { shape: "picture", label: "The picture's own shape" },
  { shape: "1:1", label: "Square, 1:1" },
  { shape: "4:3", label: "Landscape, 4:3" },
  { shape: "3:4", label: "Portrait, 3:4" },
  { shape: "3:2", label: "Landscape, 3:2" },
  { shape: "2:3", label: "Portrait, 2:3" },
  { shape: "16:9", label: "Wide, 16:9" },
  { shape: "9:16", label: "Tall, 9:16" },
];

/** A box's shape as whole numbers across and down.
 *
 * `exact` holds a box to whole multiples of those numbers, so a 16:9 box is
 * exactly 16:9. A picture's own shape can reduce to large numbers, 1023 by 767
 * say, whose multiples would leave only the whole picture, so past a small
 * ratio the box is held to it within a pixel instead.
 */
export interface CropRatio {
  across: number;
  down: number;
  exact: boolean;
}

const LARGEST_EXACT_TERM = 64;

function divisor(a: number, b: number): number {
  return b === 0 ? a : divisor(b, a % b);
}

/** The ratio a shape holds a box to in a picture of this size, or null for any shape. */
export function cropRatio(shape: CropShape, picture: { width: number; height: number }): CropRatio | null {
  if (shape === "free") return null;
  const [across, down] = shape === "picture"
    ? [picture.width, picture.height]
    : shape.split(":").map(Number);
  if (!(across > 0 && down > 0)) return null;
  const common = divisor(across, down);
  const reduced = { across: across / common, down: down / common };
  return { ...reduced, exact: Math.max(reduced.across, reduced.down) <= LARGEST_EXACT_TERM };
}

/** A box's width and height at `scale` times the ratio, as whole pixels. */
function sized(ratio: CropRatio, scale: number): { width: number; height: number } {
  if (ratio.exact) {
    const whole = Math.floor(scale);
    return { width: whole * ratio.across, height: whole * ratio.down };
  }
  return { width: Math.floor(scale * ratio.across), height: Math.floor(scale * ratio.down) };
}

/** Where the far corner of a box drawn from `origin` toward `point` lands.
 *
 * The box grows with whichever way the pointer has moved further, so a drag
 * mostly sideways still makes a box of the full shape, and it stops at the
 * picture's edge rather than running past it. `origin` is a whole pixel.
 */
export function constrainedCorner(
  origin: { x: number; y: number },
  point: { x: number; y: number },
  ratio: CropRatio,
  picture: { width: number; height: number },
): { x: number; y: number } {
  const right = point.x >= origin.x;
  const below = point.y >= origin.y;
  const roomAcross = right ? picture.width - origin.x : origin.x;
  const roomDown = below ? picture.height - origin.y : origin.y;
  const wanted = Math.max(Math.abs(point.x - origin.x) / ratio.across, Math.abs(point.y - origin.y) / ratio.down);
  const scale = Math.max(0, Math.min(wanted, roomAcross / ratio.across, roomDown / ratio.down));
  const { width, height } = sized(ratio, scale);
  return { x: origin.x + (right ? width : -width), y: origin.y + (below ? height : -height) };
}

/** The largest box of the ratio inside `box`, centered in it, or null if none fits. */
export function fittedBox(box: StudioCropBox, ratio: CropRatio): StudioCropBox | null {
  const { width, height } = sized(ratio, Math.min(box.width / ratio.across, box.height / ratio.down));
  if (width < 1 || height < 1) return null;
  return {
    left: box.left + Math.floor((box.width - width) / 2),
    top: box.top + Math.floor((box.height - height) / 2),
    width,
    height,
  };
}
