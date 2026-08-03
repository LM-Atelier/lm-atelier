/** The studio's ephemeral canvas state, as a pure reducer.
 *
 * Tool choice, brush size, and the mask's undo history are view state, not
 * server state - keeping them in one reducer makes the ordering rules
 * testable without a DOM. The load-bearing rule lives here: a gesture
 * snapshots the mask BEFORE its first mutation, because snapshotting after
 * would store the already-painted raster and make the first Undo a no-op.
 */

import {
  createMask,
  feather as featherMask,
  invert as invertMask,
  MaskHistory,
  type MaskRaster,
} from "./studioMasks";
import { BrushTool, LassoTool, RectTool, type PointerTool } from "./studioTools";

export type StudioToolKind = "instruct" | "brush" | "eraser" | "rect" | "lasso";

export type StudioToolState = {
  readonly kind: StudioToolKind;
  readonly brushRadius: number;
  readonly featherPx: number;
  readonly mask: MaskRaster | null;
  /** Bumped whenever the raster changes so the canvas repaints its tint. */
  readonly maskVersion: number;
  readonly history: MaskHistory;
};

export type StudioToolAction =
  | { type: "select-tool"; kind: StudioToolKind }
  | { type: "set-brush-radius"; radius: number }
  | { type: "set-feather"; px: number }
  | { type: "image-changed"; width: number; height: number }
  | { type: "gesture-start" }
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
    case "gesture-start": {
      if (!state.mask) return state;
      // Snapshot the pre-gesture raster: this is the state Undo restores.
      state.history.push(state.mask);
      return state;
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

/** The pointer tool for the current state, or null for text-only modes. */
export function toolFor(state: StudioToolState): PointerTool | null {
  if (!state.mask) return null;
  switch (state.kind) {
    case "brush":
      return new BrushTool(state.mask, state.brushRadius);
    case "eraser":
      return new BrushTool(state.mask, state.brushRadius, 0);
    case "rect":
      return new RectTool(state.mask);
    case "lasso":
      return new LassoTool(state.mask);
    case "instruct":
      return null;
  }
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, Math.round(value)));
}
