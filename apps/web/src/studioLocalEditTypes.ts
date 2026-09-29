/** What the browser sends for an edit the studio makes itself, without a model.
 *
 * Kept apart from types.ts, which re-exports it whole, so the growing set of
 * those edits does not crowd the shared declarations.
 */

/** An edit the studio makes itself, without a model. */
export type StudioLocalEditOperation =
  | "rotate_clockwise" | "rotate_counterclockwise" | "flip_horizontal" | "flip_vertical" | "straighten"
  | "perspective" | "crop" | "resize" | "adjust" | "blur" | "pixelate" | "paint" | "caption" | "canvas";

export interface StudioLocalEditRequest {
  source_artifact_id: string;
  operation: StudioLocalEditOperation;
  /** The part to keep, in the picture's own pixels; given with a crop only. */
  crop?: StudioCropBox | null;
  /** How far to turn the picture; given with a straightening only. */
  straighten?: StudioStraighten | null;
  /** Where the corners of what should be square lie; given with a perspective correction only. */
  perspective?: StudioPerspective | null;
  /** The new size in pixels; given with a resize only. */
  size?: StudioPictureSize | null;
  /** Where the light and color sliders stand; given with an adjustment only. */
  adjustments?: StudioColorAdjustments | null;
  /** The uploaded marked area and how far to blur it; given with a blur only. */
  blur?: StudioSelectionBlur | null;
  /** The uploaded marked area and its block size; given with a pixelation only. */
  pixelate?: StudioSelectionPixelate | null;
  /** The uploaded marked area and the paint laid over it; given with a paint only. */
  paint?: StudioSelectionPaint | null;
  /** The uploaded drawing of the words; given with a caption only. */
  caption?: StudioCaptionOverlay | null;
  /** The new canvas, where the picture sits on it, and the fill; given with a canvas change only. */
  canvas?: StudioCanvasChange | null;
}

/** Where the picture sits on a new canvas: a corner, an edge's middle, or the center. */
export type StudioCanvasAnchor =
  | "top_left" | "top" | "top_right" | "left" | "center" | "right"
  | "bottom_left" | "bottom" | "bottom_right";

/** A new canvas size, where the picture sits on it, and what fills the rest. */
export interface StudioCanvasChange {
  width: number;
  height: number;
  anchor: StudioCanvasAnchor;
  fill: "transparent" | "white" | "black";
}

/** Words the browser drew at the picture's size, uploaded as a transparent picture. */
export interface StudioCaptionOverlay {
  overlay_artifact_id: string;
}

/** The marked area to paint, uploaded as a selection, with its color and opacity. */
export interface StudioSelectionPaint {
  mask_artifact_id: string;
  /** As "#rrggbb". */
  color: string;
  /** From 1 to 100 percent. */
  opacity: number;
}

/** The marked area to blur, uploaded as a selection, and the radius in pixels. */
export interface StudioSelectionBlur {
  mask_artifact_id: string;
  radius: number;
}

/** The marked area to pixelate, uploaded as a selection, and the block size in pixels. */
export interface StudioSelectionPixelate {
  mask_artifact_id: string;
  block: number;
}

/** What an edit needs besides the picture and the operation: a box, a size,
 * sliders, or a marked area, which is uploaded before the edit is asked for. */
export type StudioLocalEditDetails = Pick<
  StudioLocalEditRequest, "crop" | "straighten" | "perspective" | "size" | "adjustments" | "canvas"
> & {
  blur?: { selection: Blob; radius: number };
  pixelate?: { selection: Blob; block: number };
  paint?: { selection: Blob; color: string; opacity: number };
  caption?: { words: Blob };
};

/** Where each light and color slider stands, from -100 to 100, with 0 unchanged. */
export interface StudioColorAdjustments {
  brightness: number;
  contrast: number;
  saturation: number;
  warmth: number;
  tint: number;
}

/** A point on a picture, in its own pixels as it is seen upright. */
export interface StudioPoint {
  x: number;
  y: number;
}

/** Where the corners of something that should be a rectangle lie on the picture.
 *
 * Correcting the perspective makes what lies inside them the whole picture,
 * upright and square-cornered.
 */
export interface StudioPerspective {
  top_left: StudioPoint;
  top_right: StudioPoint;
  bottom_right: StudioPoint;
  bottom_left: StudioPoint;
}

/** How far to turn a picture to straighten it, in degrees, clockwise when positive. */
export interface StudioStraighten {
  degrees: number;
}

/** A picture's size in its own pixels. */
export interface StudioPictureSize {
  width: number;
  height: number;
}

/** A box in a picture's own pixels, as the picture is seen upright. */
export interface StudioCropBox {
  left: number;
  top: number;
  width: number;
  height: number;
}
