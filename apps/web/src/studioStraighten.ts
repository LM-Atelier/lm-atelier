/** Straightening a picture: turning it a few degrees and keeping what it still covers.
 *
 * The server turns the picture about its center and keeps the largest box of
 * the picture's own shape that the turned picture still covers, two pixels in
 * from each side so its resampling never reaches the empty corners. This is
 * that same arithmetic, which the canvas uses to show what will be kept.
 */

/** The steepest a picture can be straightened by, either way, in degrees. */
export const STRAIGHTEN_LIMIT = 45;

/** How much of the picture's width and height the straightened picture keeps. */
export function keptScale(width: number, height: number, degrees: number): number {
  const turn = (Math.abs(degrees) * Math.PI) / 180;
  const scale = Math.min(
    width / (width * Math.cos(turn) + height * Math.sin(turn)),
    height / (width * Math.sin(turn) + height * Math.cos(turn)),
  );
  return scale - 4 / Math.min(width, height);
}

/** The size in pixels a straightening keeps, as the server cuts it. */
export function keptSize(width: number, height: number, degrees: number): { width: number; height: number } {
  const scale = keptScale(width, height, degrees);
  return {
    width: Math.max(1, Math.floor(width * scale)),
    height: Math.max(1, Math.floor(height * scale)),
  };
}

/** The picture turned by `degrees`, clockwise when positive, with the kept box
 * filling a canvas of the picture's own size; null without a canvas. */
export function drawStraightened(picture: CanvasImageSource & { width: number; height: number }, degrees: number): HTMLCanvasElement | null {
  const { width, height } = picture;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  const scale = keptScale(width, height, degrees);
  context.imageSmoothingQuality = "high";
  context.translate(width / 2, height / 2);
  context.scale(1 / scale, 1 / scale);
  context.rotate((degrees * Math.PI) / 180);
  context.drawImage(picture, -width / 2, -height / 2);
  return canvas;
}
