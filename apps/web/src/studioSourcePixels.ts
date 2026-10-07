/** The picture's own pixels, read once for the magic wand.
 *
 * The canvas is the only way a browser hands over decoded pixels, so this is
 * the one place the wand meets a DOM API; the selection itself stays pure.
 * Returns null when the browser gives nothing back or gives back something of
 * the wrong size, so the wand refuses rather than comparing against the wrong
 * picture.
 */
export function readSourcePixels(image: ImageBitmap): Uint8ClampedArray | null {
  try {
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return null;
    context.drawImage(image, 0, 0);
    const { data } = context.getImageData(0, 0, image.width, image.height);
    return data.length === image.width * image.height * 4 ? data : null;
  } catch {
    return null;
  }
}
