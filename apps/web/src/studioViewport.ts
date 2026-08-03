/** The studio's viewport: one transform between screen and image space.
 *
 * Every pointer tool works in image coordinates, so all zoom/pan math lives
 * here as pure functions over `{scale, tx, ty}` - testable without a DOM,
 * and the canvas layers apply the same transform via CSS so the three
 * surfaces can never disagree.
 */

export type Viewport = {
  readonly scale: number;
  readonly tx: number;
  readonly ty: number;
};

export const MIN_SCALE = 1 / 16;
export const MAX_SCALE = 16;

export function identityViewport(): Viewport {
  return { scale: 1, tx: 0, ty: 0 };
}

/** Screen point -> image point under the viewport. */
export function toImagePoint(
  viewport: Viewport,
  screen: { x: number; y: number },
): { x: number; y: number } {
  return {
    x: (screen.x - viewport.tx) / viewport.scale,
    y: (screen.y - viewport.ty) / viewport.scale,
  };
}

/** Image point -> screen point under the viewport. */
export function toScreenPoint(
  viewport: Viewport,
  image: { x: number; y: number },
): { x: number; y: number } {
  return {
    x: image.x * viewport.scale + viewport.tx,
    y: image.y * viewport.scale + viewport.ty,
  };
}

/** Zoom about a screen anchor so the pixel under the cursor stays put. */
export function zoomAbout(
  viewport: Viewport,
  anchor: { x: number; y: number },
  factor: number,
): Viewport {
  const scale = clampScale(viewport.scale * factor);
  const applied = scale / viewport.scale;
  return {
    scale,
    tx: anchor.x - (anchor.x - viewport.tx) * applied,
    ty: anchor.y - (anchor.y - viewport.ty) * applied,
  };
}

export function panBy(viewport: Viewport, dx: number, dy: number): Viewport {
  return { ...viewport, tx: viewport.tx + dx, ty: viewport.ty + dy };
}

/** Center the image in the container at the largest whole-fit scale. */
export function fitViewport(
  image: { width: number; height: number },
  container: { width: number; height: number },
): Viewport {
  const scale = clampScale(
    Math.min(container.width / image.width, container.height / image.height),
  );
  return {
    scale,
    tx: (container.width - image.width * scale) / 2,
    ty: (container.height - image.height * scale) / 2,
  };
}

function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
}
