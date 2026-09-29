/** Where an earlier picture sits when it is laid over a later one to compare them.
 *
 * Pure, so the geometry is provable without a canvas. The earlier picture is
 * drawn into the later one's frame, which is what keeps a comparison at the
 * zoom and position the canvas already has.
 */

type Size = { readonly width: number; readonly height: number };

/** How far apart two shapes may be and still count as one.
 *
 * Workflows round a size to a multiple of 8 or 16, so an edit of 1000 by 667
 * comes back as 1000 by 672. That is the same picture to anyone looking at it,
 * and refusing to line the two up over less than a percent would be pedantic.
 */
const SHAPE_TOLERANCE = 0.01;

export function sameShape(a: Size, b: Size): boolean {
  if (a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0) return false;
  // Cross-multiplied, so no ratio is ever divided out of a zero.
  return Math.abs(a.width * b.height - b.width * a.height) <= SHAPE_TOLERANCE * a.width * b.height;
}

/** The rectangle, in the later picture's pixels, that the earlier one fills.
 *
 * The same shape fills the frame exactly, so the two line up edge for edge
 * and a split divides one scene. A different shape - an extended picture, say
 * - is fitted whole and centred instead, because stretching it to the frame
 * would show a picture that never existed.
 */
export function compareFit(before: Size, frame: Size): { x: number; y: number; width: number; height: number } {
  if (sameShape(before, frame)) return { x: 0, y: 0, width: frame.width, height: frame.height };
  // Nothing to draw, rather than a rectangle of NaN.
  if (before.width <= 0 || before.height <= 0) return { x: 0, y: 0, width: 0, height: 0 };
  const scale = Math.min(frame.width / before.width, frame.height / before.height);
  const width = before.width * scale;
  const height = before.height * scale;
  return { x: (frame.width - width) / 2, y: (frame.height - height) / 2, width, height };
}
