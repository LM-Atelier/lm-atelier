/** The studio's ephemeral canvas state, as a pure reducer.
 *
 * Tool choice, brush size, and the mask's undo history are view state, not
 * server state - keeping them in one reducer makes the ordering rules
 * testable without a DOM. The load-bearing rule lives here: a gesture
 * snapshots the mask BEFORE its first mutation, because snapshotting after
 * would store the already-painted raster and make the first Undo a no-op.
 * That snapshot is `snapshotBeforeGesture`, called as the gesture starts,
 * not a reducer action: see its note for why.
 */

import { cropRatio, fittedBox, type CropShape } from "./studioCropShape";
import { PerspectiveTool, pictureCorners } from "./studioPerspective";
import { STRAIGHTEN_LIMIT } from "./studioStraighten";
import {
  createMask,
  feather as featherMask,
  fillRect,
  invert as invertMask,
  MAX_FEATHER_PX,
  maskBounds,
  MaskHistory,
  type MaskRaster,
} from "./studioMasks";
import {
  BrushTool,
  LassoTool,
  magicWandTool,
  paintBucketTool,
  RectTool,
  type PointerTool,
} from "./studioTools";

import type { LightDirection } from "./studioLightMap";
import { NEUTRAL_ADJUSTMENTS } from "./studioAdjustments";
import { DEFAULT_CAPTION, type StudioCaption } from "./studioCaption";
import type { StudioColorAdjustments, StudioPerspective, StudioToolKind } from "./types";

export type { StudioToolKind } from "./types";

/** The tools whose drawing is part of the request.
 *
 * Enhance and Extend are not among them: the whole picture is their subject,
 * and the size or the margins are the whole instruction. Gating on "not the
 * instruct tool" instead meant a selection drawn with the brush, left on the
 * canvas, travelled with an Enhance or an Extend that never asked for one -
 * a mask the reader had stopped thinking about, silently narrowing the work.
 * Text is among them: the box around the words is what keeps the rest of the
 * picture as it was.
 */
const MASK_TOOLS: ReadonlySet<StudioToolKind> = new Set<StudioToolKind>([
  "brush",
  "eraser",
  "rect",
  "lasso",
  "bucket",
  "wand",
  "text",
]);

export function toolUsesMask(kind: StudioToolKind): boolean {
  return MASK_TOOLS.has(kind);
}

/** The tools that work on a marked part of the picture, and so offer the selection's controls.
 *
 * Besides the tools whose drawing goes with the request, Blur and Paint work
 * inside the marking. Every other tool takes the whole picture, so a brush
 * size, Invert and Soften edges beside it would change nothing it makes, and
 * the selection is kept, unused, for the next tool that marks.
 */
const MARKING_TOOLS: ReadonlySet<StudioToolKind> = new Set<StudioToolKind>([...MASK_TOOLS, "blur", "paint"]);

export function toolMarksPicture(kind: StudioToolKind): boolean {
  return MARKING_TOOLS.has(kind);
}

/** The ways of drawing a selection: one tool, drawn six ways. */
export const SELECTION_KINDS = ["brush", "eraser", "rect", "lasso", "bucket", "wand"] as const;

export type SelectionKind = (typeof SELECTION_KINDS)[number];

export function isSelectionKind(kind: StudioToolKind): kind is SelectionKind {
  return (SELECTION_KINDS as readonly StudioToolKind[]).includes(kind);
}

export type StudioToolState = {
  readonly kind: StudioToolKind;
  /** The way of selecting last used, which choosing Select again returns to. */
  readonly selectionKind: SelectionKind;
  readonly brushRadius: number;
  readonly featherPx: number;
  /** Whether the paint bucket and the magic wand add to the selection or take away from it. */
  readonly selectionMode: "add" | "remove";
  /** How far a color may differ from the clicked one and still join the wand's selection. */
  readonly colorTolerance: number;
  /** How much larger Enhance should make the picture. */
  readonly upscaleFactor: number;
  /** How far past each edge to paint, as a fraction of the picture. */
  readonly margins: { top: number; right: number; bottom: number; left: number };
  /** The words as they read in the picture now, when the reader gives them. */
  readonly currentWords: string;
  /** The words that should read there instead. */
  readonly newWords: string;
  /** Where the relight tool's light comes from. */
  readonly lightDirection: LightDirection;
  /** How much of the relit picture is kept, from a quarter to all of it. */
  readonly lightIntensity: number;
  /** The light's colour temperature in kelvin, or null for no warmth grade. */
  readonly lightKelvin: number | null;
  /** Where the light and color sliders stand; shown on the canvas until applied. */
  readonly adjustments: StudioColorAdjustments;
  /** How far a blur spreads, in the picture's own pixels. */
  readonly blurRadius: number;
  /** Whether the blur tool blurs the marked area or breaks it into blocks. */
  readonly blurStyle: "blur" | "pixelate";
  /** The shape a crop box is held to while it is drawn. */
  readonly cropShape: CropShape;
  /** How far to turn the picture to straighten it, in degrees, clockwise when positive. */
  readonly straightenDegrees: number;
  /** Where the perspective correction's corners stand; null while they are the picture's own. */
  readonly perspective: StudioPerspective | null;
  /** The side of each block a pixelation makes, in the picture's own pixels. */
  readonly pixelBlock: number;
  /** The paint's color as "#rrggbb", and how much of it covers the picture, in percent. */
  readonly paintColor: string;
  readonly paintOpacity: number;
  /** Words being written to add; shown on the canvas until they are added. */
  readonly caption: StudioCaption;
  /** The picture a replaced subject is taken from. */
  readonly subjectPicture: File | null;
  readonly mask: MaskRaster | null;
  /** Bumped whenever the raster changes so the canvas repaints its tint. */
  readonly maskVersion: number;
  readonly history: MaskHistory;
};

export type StudioToolAction =
  | { type: "select-tool"; kind: StudioToolKind }
  | { type: "set-brush-radius"; radius: number }
  | { type: "set-feather"; px: number }
  | { type: "set-selection-mode"; mode: "add" | "remove" }
  | { type: "set-color-tolerance"; tolerance: number }
  | { type: "set-upscale-factor"; factor: number }
  | { type: "set-margin"; side: "top" | "right" | "bottom" | "left"; fraction: number }
  | { type: "clear-margins" }
  | { type: "set-current-words"; words: string }
  | { type: "set-new-words"; words: string }
  | { type: "set-light-direction"; direction: LightDirection }
  | { type: "set-light-intensity"; intensity: number }
  | { type: "set-light-kelvin"; kelvin: number | null }
  | { type: "set-adjustment"; key: keyof StudioColorAdjustments; value: number }
  | { type: "reset-adjustments" }
  | { type: "set-blur-radius"; radius: number }
  | { type: "set-blur-style"; style: "blur" | "pixelate" }
  | { type: "set-crop-shape"; shape: CropShape }
  | { type: "set-straighten"; degrees: number }
  | { type: "set-perspective"; corners: StudioPerspective | null }
  | { type: "set-pixel-block"; block: number }
  | { type: "set-paint-color"; color: string }
  | { type: "set-paint-opacity"; opacity: number }
  | { type: "set-caption"; patch: Partial<StudioCaption> }
  | { type: "set-subject-picture"; picture: File | null }
  | { type: "image-changed"; width: number; height: number }
  /** Everything as it stood when the Studio was left, for the same picture. */
  | { type: "restore"; state: StudioToolState }
  | { type: "stroke-end" }
  | { type: "invert" }
  | { type: "feather" }
  | { type: "clear" }
  | { type: "undo" }
  | { type: "redo" };

export function initialToolState(): StudioToolState {
  return {
    kind: "instruct",
    selectionKind: "brush",
    brushRadius: 24,
    featherPx: 4,
    selectionMode: "add",
    colorTolerance: 32,
    upscaleFactor: 2,
    margins: { top: 0, right: 0, bottom: 0, left: 0 },
    currentWords: "",
    newWords: "",
    lightDirection: "left",
    lightIntensity: 0.5,
    lightKelvin: null,
    adjustments: NEUTRAL_ADJUSTMENTS,
    blurRadius: 12,
    blurStyle: "blur",
    cropShape: "free",
    straightenDegrees: 0,
    perspective: null,
    pixelBlock: 12,
    paintColor: "#000000",
    paintOpacity: 100,
    caption: DEFAULT_CAPTION,
    subjectPicture: null,
    mask: null,
    maskVersion: 0,
    history: new MaskHistory(),
  };
}

export function studioToolReducer(
  state: StudioToolState,
  action: StudioToolAction,
): StudioToolState {
  switch (action.type) {
    case "select-tool":
      return {
        ...state,
        kind: action.kind,
        selectionKind: isSelectionKind(action.kind) ? action.kind : state.selectionKind,
      };
    case "set-brush-radius":
      return { ...state, brushRadius: clamp(action.radius, 1, 512) };
    case "set-feather":
      return { ...state, featherPx: clamp(action.px, 0, MAX_FEATHER_PX) };
    case "set-selection-mode":
      return { ...state, selectionMode: action.mode };
    case "set-color-tolerance":
      return { ...state, colorTolerance: clamp(action.tolerance, 0, 255) };
    case "set-upscale-factor":
      return { ...state, upscaleFactor: clamp(action.factor, 1, 8) };
    case "set-margin":
      return {
        ...state,
        margins: { ...state.margins, [action.side]: bounded(action.fraction, 0, 2) },
      };
    case "clear-margins":
      return { ...state, margins: { top: 0, right: 0, bottom: 0, left: 0 } };
    case "set-current-words":
      return { ...state, currentWords: action.words };
    case "set-new-words":
      return { ...state, newWords: action.words };
    case "set-light-direction":
      return { ...state, lightDirection: action.direction };
    case "set-light-intensity":
      return {
        ...state,
        lightIntensity: Number.isFinite(action.intensity)
          ? Math.min(1, Math.max(0.25, action.intensity))
          : state.lightIntensity,
      };
    case "set-light-kelvin":
      return { ...state, lightKelvin: action.kelvin };
    case "set-adjustment":
      return Number.isInteger(action.value) && Math.abs(action.value) <= 100
        ? { ...state, adjustments: { ...state.adjustments, [action.key]: action.value } }
        : state;
    case "reset-adjustments":
      return { ...state, adjustments: NEUTRAL_ADJUSTMENTS };
    case "set-blur-radius":
      return { ...state, blurRadius: clamp(action.radius, 1, 100) };
    case "set-straighten":
      // Tenths of a degree, the finest the slider offers.
      return {
        ...state,
        straightenDegrees:
          Math.round(Math.min(STRAIGHTEN_LIMIT, Math.max(-STRAIGHTEN_LIMIT, action.degrees)) * 10) / 10,
      };
    case "set-perspective":
      return { ...state, perspective: action.corners };
    case "set-crop-shape": {
      // A box already drawn takes the new shape at once, as the largest box
      // of that shape inside it, so the box shown is always the one kept.
      const ratio = state.mask ? cropRatio(action.shape, state.mask) : null;
      const box = state.mask && ratio ? maskBounds(state.mask) : null;
      if (!state.mask || !ratio || !box) return { ...state, cropShape: action.shape };
      state.history.push(state.mask);
      const fitted = fittedBox(box, ratio);
      const mask = createMask(state.mask.width, state.mask.height);
      if (fitted) {
        fillRect(mask, fitted.left, fitted.top, fitted.left + fitted.width, fitted.top + fitted.height);
      }
      return { ...state, cropShape: action.shape, mask, maskVersion: state.maskVersion + 1 };
    }
    case "set-blur-style":
      return { ...state, blurStyle: action.style };
    case "set-pixel-block":
      // One-pixel blocks would change nothing, so the smallest is two.
      return { ...state, pixelBlock: clamp(action.block, 2, 100) };
    case "set-paint-color":
      return /^#[0-9a-f]{6}$/.test(action.color) ? { ...state, paintColor: action.color } : state;
    case "set-paint-opacity":
      return { ...state, paintOpacity: clamp(action.opacity, 1, 100) };
    case "set-caption": {
      const caption = { ...state.caption, ...action.patch };
      caption.sizePercent = clamp(caption.sizePercent, 2, 30);
      return /^#[0-9a-f]{6}$/.test(caption.color) ? { ...state, caption } : state;
    }
    // Kept when the picture changes: the new subject can go into another one.
    case "set-subject-picture":
      return { ...state, subjectPicture: action.picture };
    case "restore":
      return action.state;
    case "image-changed": {
      // A new image invalidates the mask entirely; carrying it over would
      // silently apply a selection drawn on different pixels. The sliders
      // start over too: an applied adjustment is already in the new picture.
      return {
        ...state,
        adjustments: NEUTRAL_ADJUSTMENTS,
        straightenDegrees: 0,
        // Corners placed on the old picture mean nothing on the new one.
        perspective: null,
        // Added words are in the new picture; keeping them would draw them twice.
        caption: { ...state.caption, text: "" },
        mask: createMask(action.width, action.height),
        maskVersion: state.maskVersion + 1,
        history: new MaskHistory(),
      };
    }
    case "stroke-end":
      return { ...state, maskVersion: state.maskVersion + 1 };
    case "invert": {
      if (!state.mask) return state;
      state.history.push(state.mask);
      invertMask(state.mask);
      return { ...state, maskVersion: state.maskVersion + 1 };
    }
    case "feather": {
      if (!state.mask || state.featherPx < 1) return state;
      state.history.push(state.mask);
      featherMask(state.mask, state.featherPx);
      return { ...state, maskVersion: state.maskVersion + 1 };
    }
    case "clear": {
      if (!state.mask) return state;
      state.history.push(state.mask);
      return {
        ...state,
        mask: createMask(state.mask.width, state.mask.height),
        maskVersion: state.maskVersion + 1,
      };
    }
    case "undo": {
      if (!state.mask) return state;
      const previous = state.history.undo(state.mask);
      if (!previous) return state;
      return { ...state, mask: previous, maskVersion: state.maskVersion + 1 };
    }
    case "redo": {
      if (!state.mask) return state;
      const next = state.history.redo(state.mask);
      if (!next) return state;
      return { ...state, mask: next, maskVersion: state.maskVersion + 1 };
    }
  }
}

/** What to call a turn the reader gave no words for.
 *
 * The turn contract requires text, and these two tools deliberately ask for
 * none - so an empty box reached the server and was refused before anything
 * ran. Describing the operation is truthful and reads as a caption in the
 * filmstrip, which is where this ends up.
 */
export function defaultInstruction(state: StudioToolState): string {
  if (state.kind === "enhance") return `Enhance to ${state.upscaleFactor}x`;
  if (state.kind === "text") return replaceWordsInstruction(state);
  // The workflow reads no words; these are what the history shows it did.
  if (state.kind === "isolate") return "Cut the subject out onto a transparent background.";
  // The lighting adapter reads its two pictures as figures, and the direction
  // in words must agree with the map or it follows the words.
  if (state.kind === "relight") {
    return `Relight Figure 1 using the luminance map from Figure 2 (light source from the ${state.lightDirection}).`;
  }
  if (state.kind === "extend") {
    const edges = Object.entries(state.margins)
      .filter(([, value]) => value)
      .map(([edge]) => edge);
    return edges.length ? `Extend past the ${edges.join(", ")}` : "Extend the picture";
  }
  return "";
}

/** The words for a text replacement, or nothing until there are new words.
 *
 * The edit runs on the whole picture, so the model is told which words to
 * change when the reader says, and to keep their look; the selection then
 * keeps everything outside the box as it was.
 */
function replaceWordsInstruction(state: StudioToolState): string {
  const replacement = state.newWords.trim();
  if (!replacement) return "";
  const current = state.currentWords.trim();
  const target = current ? `the text "${current}"` : "the text";
  return `Replace ${target} with "${replacement}". Keep the same font, color, size and position, and leave everything else unchanged.`;
}

/** The pointer tool for the current state, or null for text-only modes.
 *
 * `pixels` is the picture as RGBA bytes, needed only by the magic wand. Without
 * them the wand has nothing to compare, so it gives no tool rather than one
 * that selects by guesswork. `onCorners` hears where a perspective
 * correction's corners are dragged to.
 */
export function toolFor(
  state: StudioToolState,
  pixels: Uint8ClampedArray | null = null,
  onCorners: (corners: StudioPerspective) => void = () => {},
): PointerTool | null {
  if (!state.mask) return null;
  const selected = state.selectionMode === "add" ? 255 : 0;
  switch (state.kind) {
    // A blur or a paint is marked with the brush, into the same selection the
    // other tools draw.
    case "brush":
    case "blur":
    case "paint":
      return new BrushTool(state.mask, state.brushRadius);
    case "eraser":
      return new BrushTool(state.mask, state.brushRadius, 0);
    case "rect":
      return new RectTool(state.mask);
    // Words sit in a line, so a box is the natural way to point at them.
    case "text":
      return new RectTool(state.mask);
    // The light comes from a side of the whole picture, chosen in the panel.
    case "relight":
      return null;
    case "lasso":
      return new LassoTool(state.mask);
    case "bucket":
      return paintBucketTool(state.mask, selected);
    case "wand":
      return pixels ? magicWandTool(state.mask, pixels, state.colorTolerance, selected) : null;
    case "instruct":
      return null;
    // Enhance points at nothing: the whole picture is the subject and the only
    // choice is how much larger. A tool that could be a number never becomes a
    // gesture.
    case "enhance":
      return null;
    // Extend is a drag on the frame, not a gesture on the picture: the whole
    // point is pointing at where the picture is not.
    case "extend":
      return null;
    // Isolate points at nothing either: the subject is whatever the workflow
    // finds, and a gesture would only disagree with it.
    case "isolate":
      return null;
    // Nor does replacing a background: the cutout finds the subject, and the
    // words say what goes around it.
    case "background":
      return null;
    // Nor replacing a subject: the cutout finds it, and a second picture says
    // what takes its place.
    case "subject":
      return null;
    // Turning and flipping act on the whole picture, chosen by a press.
    case "transform":
      return null;
    // The corners are dragged on the picture, and none of it is selected.
    case "perspective": {
      const size = { width: state.mask.width, height: state.mask.height };
      return new PerspectiveTool(state.perspective ?? pictureCorners(size.width, size.height), size, onCorners);
    }
    // A crop is one box: drawing another replaces it rather than adding to it.
    case "crop":
      return new RectTool(state.mask, true, cropRatio(state.cropShape, state.mask));
    // A resize is two numbers for the whole picture, a canvas change two and
    // a place, and an adjustment four.
    case "resize":
    case "canvas":
    case "adjust":
    case "caption":
      return null;
  }
}

/** Snapshot the selection before a gesture changes it: the state Undo returns to.
 *
 * Called at the moment the gesture starts rather than dispatched. React runs
 * a dispatched action when it next renders, which is after the event handler
 * that started the gesture has finished, and a brush stamps its first dab in
 * that same handler. A dispatched snapshot therefore already held that dab,
 * and Undo left the start of every brush stroke behind.
 */
export function snapshotBeforeGesture(state: StudioToolState): void {
  // Moving a perspective correction's corner leaves the selection alone,
  // so it keeps no step for Undo to return to.
  if (state.mask && state.kind !== "perspective") state.history.push(state.mask);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, Math.round(value)));
}

/** Hold a value within bounds without rounding it: a share of something, not a count.
 *
 * An edge's margin is a share of the picture. Rounded as a count is, every
 * margin became none of the picture or all of it.
 */
function bounded(value: number, low: number, high: number): number {
  return Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : low;
}
