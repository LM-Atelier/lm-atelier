/** Replacing a background or a subject: the cutout's own alpha is the selection.
 *
 * The cutout workflow finds the subject and returns it on transparency, so its
 * alpha says exactly where the subject is. Inverted, it covers everything
 * around the subject, and a redraw placed back through it changes the
 * background while the subject keeps its own pixels. As it is, grown a little,
 * it covers the subject, and a removal placed back through it takes the
 * subject away while everything around it keeps its own.
 */

import { readSourcePixels } from "./studioSourcePixels";
import { createMask, type MaskRaster } from "./studioMasks";
import type { ChatDetail } from "./types";

export type CutoutOutcome =
  | { state: "waiting" }
  | { state: "ready"; artifactId: string }
  | { state: "failed" }
  | { state: "stopped" };

/** Where the cutout turn stands, read from the session the studio already polls. */
export function cutoutOutcome(session: ChatDetail | null, messageId: string): CutoutOutcome {
  const message = session?.messages.find((item) => item.id === messageId);
  if (!message || message.status === "pending") return { state: "waiting" };
  // Stopped on purpose, which is not a failure to report.
  if (message.status === "cancelled") return { state: "stopped" };
  const image = message.parts.find(
    (part) => part.type === "image" && Boolean(part.artifact_id) && !part.metadata_json.preview,
  );
  return message.status === "complete" && image?.artifact_id
    ? { state: "ready", artifactId: image.artifact_id }
    : { state: "failed" };
}

/** How far past its outline a removed subject's selection reaches, in pixels.
 *
 * A selection held to the old outline leaves a ring of what was removed, since
 * a cutout's edge never quite reaches it. A small share of the picture's
 * shorter side covers the ring, with a floor so a small picture still gets
 * some.
 */
export function subjectReach(width: number, height: number): number {
  return Math.max(8, Math.round(Math.min(width, height) * 0.04));
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
