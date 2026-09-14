/** The luminance map the relight tool sends beside the picture.
 *
 * The lighting adapter reads its second picture as where the light is: bright
 * on the side the light comes from, fading to dark across the frame. A map at
 * the picture's own size keeps the two aligned.
 */

export type LightDirection = "left" | "right" | "top";

/** Where the gradient starts (brightest) and ends (darkest), in pixels. */
export function lightMapGradient(
  width: number,
  height: number,
  direction: LightDirection,
): [number, number, number, number] {
  switch (direction) {
    case "left":
      return [0, 0, width, 0];
    case "right":
      return [width, 0, 0, 0];
    case "top":
      return [0, 0, 0, height];
  }
}

/** The map as a PNG, or null when this browser cannot draw one. */
export async function renderLightMap(
  width: number,
  height: number,
  direction: LightDirection,
): Promise<Blob | null> {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  const gradient = context.createLinearGradient(...lightMapGradient(width, height, direction));
  gradient.addColorStop(0, "#ffffff");
  gradient.addColorStop(1, "#000000");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);
  return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), "image/png"));
}
