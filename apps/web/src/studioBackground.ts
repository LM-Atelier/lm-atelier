/** Replacing a background: the cutout's own alpha is the selection.
 *
 * The cutout workflow finds the subject and returns it on transparency, so its
 * alpha says exactly where the subject is. As a selection, inverted, it covers
 * everything around the subject, and the redraw is placed back through it: the
 * background changes and the subject keeps its own pixels.
 */

import { readSourcePixels } from "./studioSourcePixels";
import { createMask, type MaskRaster } from "./studioMasks";
import type { ChatDetail } from "./types";

export type CutoutOutcome =
  | { state: "waiting" }
  | { state: "ready"; artifactId: string }
  | { state: "failed" };

/** Where the cutout turn stands, read from the session the studio already polls. */
export function cutoutOutcome(session: ChatDetail | null, messageId: string): CutoutOutcome {
  const message = session?.messages.find((item) => item.id === messageId);
  if (!message || message.status === "pending") return { state: "waiting" };
  const image = message.parts.find(
    (part) => part.type === "image" && Boolean(part.artifact_id) && !part.metadata_json.preview,
  );
  return message.status === "complete" && image?.artifact_id
    ? { state: "ready", artifactId: image.artifact_id }
    : { state: "failed" };
}

/** The subject's coverage from a cutout's RGBA pixels: its alpha, soft edges and all. */
export function subjectMask(pixels: Uint8ClampedArray, width: number, height: number): MaskRaster {
  const mask = createMask(width, height);
  for (let index = 0; index < width * height; index += 1) {
    mask.data[index] = pixels[index * 4 + 3];
  }
  return mask;
}

/** The cutout's subject at the source picture's size, or null when it cannot be read.
 *
 * Decoded straight to the source's size, so the selection lines up with the
 * picture it is placed back over even if the cutout came back another size.
 */
export async function readCutoutMask(
  artifactId: string,
  width: number,
  height: number,
): Promise<MaskRaster | null> {
  const response = await fetch(`/api/artifacts/${encodeURIComponent(artifactId)}/content`);
  if (!response.ok) return null;
  const bitmap = await createImageBitmap(await response.blob(), {
    resizeWidth: width,
    resizeHeight: height,
  });
  try {
    const pixels = readSourcePixels(bitmap);
    return pixels && bitmap.width === width && bitmap.height === height
      ? subjectMask(pixels, width, height)
      : null;
  } finally {
    bitmap.close();
  }
}
