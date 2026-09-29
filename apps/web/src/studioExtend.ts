/** What an extension makes: the pixels each edge gains and where the new edges sit on screen.
 *
 * The margins travel as fractions of the picture, and the server turns each
 * into whole pixels: a fraction of the width for the left and right edges and
 * of the height for the top and bottom, a half pixel rounding up. These work
 * the same numbers out the same way, so the size the Studio shows is the size
 * the workflow is asked for, and the frame it draws is the canvas it paints.
 */

import type { ScreenRect } from "./studioViewport";

export type ExtendSide = "top" | "right" | "bottom" | "left";

export type ExtendMargins = Readonly<Record<ExtendSide, number>>;

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
