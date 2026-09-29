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

type ScreenPoint = { x: number; y: number };

/** Follow two pointers from where they were to where they are now.
 *
 * The picture scales by how far apart they have moved and travels with the
 * point midway between them, so what lay under that point stays under it.
 * Unless they twist, which the picture cannot follow since it does not turn,
 * or the zoom meets its limit, what lay under each pointer stays under it too.
 */
export function pinchBy(
  viewport: Viewport,
  from: readonly [ScreenPoint, ScreenPoint],
  to: readonly [ScreenPoint, ScreenPoint],
): Viewport {
  const before = midpoint(from);
  const after = midpoint(to);
  const apart = Math.hypot(from[1].x - from[0].x, from[1].y - from[0].y);
  const now = Math.hypot(to[1].x - to[0].x, to[1].y - to[0].y);
  // Pointers on one spot give no distance to scale by; they still move the picture.
  const factor = apart > 0 && now > 0 ? now / apart : 1;
  return panBy(zoomAbout(viewport, before, factor), after.x - before.x, after.y - before.y);
}

function midpoint([a, b]: readonly [ScreenPoint, ScreenPoint]): ScreenPoint {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/** Center the image in the container at the largest whole-fit scale. */
export function fitViewport(
  image: { width: number; height: number },
  container: { width: number; height: number },
): Viewport {
  // A container with no measured size (a detached or test DOM) would make
  // the ratio zero or NaN and every unprojected point NaN with it.
  if (!image.width || !image.height || !container.width || !container.height) {
    return identityViewport();
  }
  const scale = clampScale(
    Math.min(container.width / image.width, container.height / image.height),
  );
  return {
    scale,
    tx: (container.width - image.width * scale) / 2,
    ty: (container.height - image.height * scale) / 2,
  };
}

/** A rectangle on screen, in the canvas's own CSS pixels. */
export type ScreenRect = {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
};

/** Where the picture is shown under the viewport: its corner and its size at this zoom. */
export function shownRect(viewport: Viewport, image: { width: number; height: number }): ScreenRect {
  return {
    x: viewport.tx,
    y: viewport.ty,
    width: image.width * viewport.scale,
    height: image.height * viewport.scale,
  };
}

function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
}
