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

import {
  createMask,
  feather as featherMask,
  invert as invertMask,
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

import type { StudioToolKind } from "./types";

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

export type StudioToolState = {
  readonly kind: StudioToolKind;
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
  | { type: "image-changed"; width: number; height: number }
  | { type: "stroke-end" }
  | { type: "invert" }
  | { type: "feather" }
  | { type: "clear" }
  | { type: "undo" }
  | { type: "redo" };

export function initialToolState(): StudioToolState {
  return {
    kind: "instruct",
    brushRadius: 24,
    featherPx: 4,
    selectionMode: "add",
    colorTolerance: 32,
    upscaleFactor: 2,
    margins: { top: 0, right: 0, bottom: 0, left: 0 },
    currentWords: "",
    newWords: "",
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
      return { ...state, kind: action.kind };
    case "set-brush-radius":
      return { ...state, brushRadius: clamp(action.radius, 1, 512) };
    case "set-feather":
      return { ...state, featherPx: clamp(action.px, 0, 128) };
    case "set-selection-mode":
      return { ...state, selectionMode: action.mode };
    case "set-color-tolerance":
      return { ...state, colorTolerance: clamp(action.tolerance, 0, 255) };
    case "set-upscale-factor":
      return { ...state, upscaleFactor: clamp(action.factor, 1, 8) };
    case "set-margin":
      return {
        ...state,
        margins: { ...state.margins, [action.side]: clamp(action.fraction, 0, 2) },
      };
    case "clear-margins":
      return { ...state, margins: { top: 0, right: 0, bottom: 0, left: 0 } };
    case "set-current-words":
      return { ...state, currentWords: action.words };
    case "set-new-words":
      return { ...state, newWords: action.words };
    case "image-changed": {
      // A new image invalidates the mask entirely; carrying it over would
      // silently apply a selection drawn on different pixels.
      return {
        ...state,
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
 * that selects by guesswork.
 */
export function toolFor(
  state: StudioToolState,
  pixels: Uint8ClampedArray | null = null,
): PointerTool | null {
  if (!state.mask) return null;
  const selected = state.selectionMode === "add" ? 255 : 0;
  switch (state.kind) {
    case "brush":
      return new BrushTool(state.mask, state.brushRadius);
    case "eraser":
      return new BrushTool(state.mask, state.brushRadius, 0);
    case "rect":
      return new RectTool(state.mask);
    // Words sit in a line, so a box is the natural way to point at them.
    case "text":
      return new RectTool(state.mask);
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
  if (state.mask) state.history.push(state.mask);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, Math.round(value)));
}
