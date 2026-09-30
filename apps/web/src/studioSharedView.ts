import { fitViewport, MAX_SCALE, MIN_SCALE, panBy, zoomAbout, type Viewport } from "./studioViewport";

/** A width and a height, of a picture or of the half of the stage it is shown in. */
export type Size = { readonly width: number; readonly height: number };

/** Where both halves of a side-by-side comparison look.
 *
 * The zoom is over each picture's own fit to its half, and the middle of each
 * half shows the same point of both pictures, as shares of their width and
 * height. Two pictures of one shape therefore show the same pixels at the same
 * size, and two of different shapes still show the same part of the scene.
 */
export type SharedView = { readonly zoom: number; readonly x: number; readonly y: number };

/** Each picture whole in its half. */
export const WHOLE_VIEW: SharedView = { zoom: 1, x: 0.5, y: 0.5 };

/** The viewport one half shows its picture with, for the shared view. */
export function halfViewport(view: SharedView, picture: Size, half: Size): Viewport {
  const fit = fitViewport(picture, half).scale;
  const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, fit * view.zoom));
  return {
    scale,
    tx: half.width / 2 - view.x * picture.width * scale,
    ty: half.height / 2 - view.y * picture.height * scale,
  };
}

/** Back from one half's viewport to the view both halves share. */
function sharedFrom(viewport: Viewport, picture: Size, half: Size): SharedView {
  const fit = fitViewport(picture, half).scale;
  // A picture or a half with no size yet has nothing to share but the whole view.
  if (!picture.width || !picture.height || !fit) return WHOLE_VIEW;
  return {
    zoom: viewport.scale / fit,
    x: (half.width / 2 - viewport.tx) / (viewport.scale * picture.width),
    y: (half.height / 2 - viewport.ty) / (viewport.scale * picture.height),
  };
}

/** The shared view zoomed about a point on one half, so what lies under that point stays there. */
export function zoomSharedAbout(
  view: SharedView,
  picture: Size,
  half: Size,
  anchor: { x: number; y: number },
  factor: number,
): SharedView {
  return sharedFrom(zoomAbout(halfViewport(view, picture, half), anchor, factor), picture, half);
}

/** The shared view moved by a drag across one half. */
export function panSharedBy(view: SharedView, picture: Size, half: Size, dx: number, dy: number): SharedView {
  return sharedFrom(panBy(halfViewport(view, picture, half), dx, dy), picture, half);
}
