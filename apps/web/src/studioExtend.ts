/** What an extension makes: the pixels each edge gains and where the new edges sit on screen.
 *
 * The margins travel as fractions of the picture, and the server turns each
 * into whole pixels: a fraction of the width for the left and right edges and
 * of the height for the top and bottom, a half pixel rounding up. These work
 * the same numbers out the same way, so the size the Studio shows is the size
 * the workflow is asked for, and the frame it draws is the canvas it paints.
 */

import type { ScreenRect } from "./studioViewport";
import type { StudioCanvasAnchor } from "./types";

export type ExtendSide = "top" | "right" | "bottom" | "left";

export type ExtendMargins = Readonly<Record<ExtendSide, number>>;

/** The most any edge may gain, as a share of the picture; the server refuses more. */
export const MAX_MARGIN = 2;

/** A shape a picture can be extended to, as its width against its height. */
export type ExtendShape = { label: string; width: number; height: number };

export const EXTEND_SHAPES: readonly ExtendShape[] = [
  { label: "Square", width: 1, height: 1 },
  { label: "4:5", width: 4, height: 5 },
  { label: "3:2", width: 3, height: 2 },
  { label: "16:9", width: 16, height: 9 },
  { label: "9:16", width: 9, height: 16 },
];

/** Whole pixels per edge as the shares the margins travel in, or nothing past the most an edge may gain.
 *
 * Each share is the pixels over the side it is measured against, which the
 * server's rounding turns back into exactly those pixels.
 */
function sharesOf(
  pixels: Record<ExtendSide, number>,
  picture: { width: number; height: number },
): ExtendMargins | null {
  const margins = {
    top: pixels.top / picture.height,
    right: pixels.right / picture.width,
    bottom: pixels.bottom / picture.height,
    left: pixels.left / picture.width,
  };
  return Object.values(margins).every((share) => share <= MAX_MARGIN) ? margins : null;
}

/** The margins that make the picture this shape by adding canvas, shared evenly by the two edges that grow.
 *
 * Nothing is ever cut: a picture wider than the shape grows taller, and a
 * taller one wider, to the nearest whole pixel. All zero when the picture is
 * that shape already, and nothing when the shape needs more than an edge may gain.
 */
export function marginsForShape(
  picture: { width: number; height: number },
  shape: ExtendShape,
): ExtendMargins | null {
  // Compared in whole numbers, so a picture exactly the shape never grows by rounding.
  const narrower = picture.width * shape.height < picture.height * shape.width;
  const width = narrower ? Math.round((picture.height * shape.width) / shape.height) : picture.width;
  const height = narrower ? picture.height : Math.round((picture.width * shape.height) / shape.width);
  return marginsForSize(picture, { width, height }, "center");
}

/** The margins that grow the picture to exactly this size, the new room away from where it is anchored.
 *
 * A picture anchored at the center gains half the room on each side, the odd
 * pixel going to the right or the bottom. Nothing when the size is smaller in
 * either direction, which is a crop's work, or needs more than an edge may gain.
 */
export function marginsForSize(
  picture: { width: number; height: number },
  size: { width: number; height: number },
  anchor: StudioCanvasAnchor,
): ExtendMargins | null {
  const across = size.width - picture.width;
  const down = size.height - picture.height;
  if (across < 0 || down < 0) return null;
  const left = anchor.endsWith("left") ? 0 : anchor.endsWith("right") ? across : Math.floor(across / 2);
  const top = anchor.startsWith("top") ? 0 : anchor.startsWith("bottom") ? down : Math.floor(down / 2);
  return sharesOf({ top, right: across - left, bottom: down - top, left }, picture);
}

/** The whole pixels each edge gains on a picture of this size. */
export function marginPixels(
  margins: ExtendMargins,
  picture: { width: number; height: number },
): Record<ExtendSide, number> {
  const pixels = (fraction: number, span: number) => Math.floor(fraction * span + 0.5);
  return {
    top: pixels(margins.top, picture.height),
    right: pixels(margins.right, picture.width),
    bottom: pixels(margins.bottom, picture.height),
    left: pixels(margins.left, picture.width),
  };
}

/** The picture's size once every edge has gained its margin. */
export function extendedSize(
  margins: ExtendMargins,
  picture: { width: number; height: number },
): { width: number; height: number } {
  const gained = marginPixels(margins, picture);
  return {
    width: picture.width + gained.left + gained.right,
    height: picture.height + gained.top + gained.bottom,
  };
}

/** The extended canvas on screen, around the picture as it is shown at this zoom. */
export function extendedFrame(
  margins: ExtendMargins,
  picture: { width: number; height: number },
  shown: ScreenRect,
): ScreenRect {
  const gained = marginPixels(margins, picture);
  const across = picture.width > 0 ? shown.width / picture.width : 0;
  const down = picture.height > 0 ? shown.height / picture.height : 0;
  return {
    x: shown.x - gained.left * across,
    y: shown.y - gained.top * down,
    width: shown.width + (gained.left + gained.right) * across,
    height: shown.height + (gained.top + gained.bottom) * down,
  };
}
