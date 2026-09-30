import { useSyncExternalStore } from "react";
import type { OutputRatioPresetId } from "./types";

/** Which output shapes the composer offers, and in what order, for pictures and for videos.
 *
 * Somebody who only ever makes wide videos should not have to look past six
 * other shapes to find the one they use. A workflow still decides which shapes
 * it can make: this only leaves some out and orders the rest. It is a choice
 * about this browser's composer, so it is remembered here, as the send key is.
 */

export type OutputShapeMode = "image" | "video";

export interface OutputShapeChoice {
  /** Every shape, in the order the composer shows them. */
  order: OutputRatioPresetId[];
  /** Shapes the composer leaves out. */
  hidden: OutputRatioPresetId[];
  /** The shape new work takes when nothing else sets its size; null leaves it to the workflow. */
  default: OutputRatioPresetId | null;
}

export type OutputShapeChoices = Record<OutputShapeMode, OutputShapeChoice>;

/** Every shape, in the order the server lists them. */
export const OUTPUT_SHAPES: readonly OutputRatioPresetId[] = ["1:1", "3:4", "2:3", "9:16", "4:3", "3:2", "16:9"];

export const OUTPUT_SHAPES_KEY = "local-lm-output-shapes";

const DEFAULT_CHOICE: OutputShapeChoice = { order: [...OUTPUT_SHAPES], hidden: [], default: null };
export const DEFAULT_OUTPUT_SHAPES: OutputShapeChoices = { image: DEFAULT_CHOICE, video: DEFAULT_CHOICE };

function isShape(value: unknown): value is OutputRatioPresetId {
  return typeof value === "string" && (OUTPUT_SHAPES as readonly string[]).includes(value);
}

/** A stored choice, or the default when anything about it is not one.
 *
 * The order must name every shape exactly once, so a choice stored before a
 * shape was added falls back to the default rather than losing that shape.
 */
function parsedChoice(value: unknown): OutputShapeChoice {
  if (typeof value !== "object" || value === null) return DEFAULT_CHOICE;
  const { order, hidden, default: chosen } = value as { order?: unknown; hidden?: unknown; default?: unknown };
  if (!Array.isArray(order) || !Array.isArray(hidden)) return DEFAULT_CHOICE;
  if (!order.every(isShape) || !hidden.every(isShape)) return DEFAULT_CHOICE;
  if (order.length !== OUTPUT_SHAPES.length || new Set(order).size !== OUTPUT_SHAPES.length) return DEFAULT_CHOICE;
  const leftOut = [...new Set(hidden)];
  // A choice saved before there was a default has none, and a default since
  // left out is no default: the composer would never offer that shape.
  return { order, hidden: leftOut, default: isShape(chosen) && !leftOut.includes(chosen) ? chosen : null };
}

let cached: { raw: string | null; choices: OutputShapeChoices } = { raw: null, choices: DEFAULT_OUTPUT_SHAPES };

function storedChoices(): OutputShapeChoices {
  let raw: string | null;
  try {
    raw = localStorage.getItem(OUTPUT_SHAPES_KEY);
  } catch {
    return DEFAULT_OUTPUT_SHAPES;
  }
  // The same object while nothing changed, so a subscriber does not re-render
  // on every read.
  if (raw === cached.raw) return cached.choices;
  let parsed: unknown;
  try {
    parsed = raw === null ? null : JSON.parse(raw);
  } catch {
    parsed = null;
  }
  const record = typeof parsed === "object" && parsed !== null ? (parsed as Record<string, unknown>) : {};
  cached = { raw, choices: { image: parsedChoice(record.image), video: parsedChoice(record.video) } };
  return cached.choices;
}

const listeners = new Set<() => void>();

/** Remember one mode's choice, and tell every composer showing shapes. */
export function setOutputShapeChoice(mode: OutputShapeMode, choice: OutputShapeChoice): void {
  const next = { ...storedChoices(), [mode]: choice };
  try {
    localStorage.setItem(OUTPUT_SHAPES_KEY, JSON.stringify(next));
  } catch {
    // Without storage the choice cannot be remembered, and every composer
    // would go on reading the old one, so nothing is told.
    return;
  }
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === OUTPUT_SHAPES_KEY) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** The current choices, re-rendering whoever reads them when they change. */
export function useOutputShapes(): OutputShapeChoices {
  return useSyncExternalStore(subscribe, storedChoices, () => DEFAULT_OUTPUT_SHAPES);
}

/** The shapes a workflow offers that the composer should show, in the chosen order. */
export function arrangedShapes(
  offered: readonly OutputRatioPresetId[],
  choice: OutputShapeChoice,
): OutputRatioPresetId[] {
  return choice.order.filter((shape) => offered.includes(shape) && !choice.hidden.includes(shape));
}

/** The choice with one shape moved one place earlier (-1) or later (1). */
export function movedShape(choice: OutputShapeChoice, shape: OutputRatioPresetId, by: -1 | 1): OutputShapeChoice {
  const from = choice.order.indexOf(shape);
  const to = from + by;
  if (from < 0 || to < 0 || to >= choice.order.length) return choice;
  const order = [...choice.order];
  [order[from], order[to]] = [order[to], order[from]];
  return { ...choice, order };
}

/** The choice with one shape shown or left out; leaving out the default leaves no default. */
export function toggledShape(choice: OutputShapeChoice, shape: OutputRatioPresetId, shown: boolean): OutputShapeChoice {
  const hidden = choice.hidden.filter((other) => other !== shape);
  if (shown) return { ...choice, hidden };
  return { ...choice, hidden: [...hidden, shape], default: choice.default === shape ? null : choice.default };
}

/** The choice with this default shape, or none; a shape left out cannot be the default. */
export function defaultedShape(choice: OutputShapeChoice, shape: OutputRatioPresetId | null): OutputShapeChoice {
  if (shape !== null && choice.hidden.includes(shape)) return choice;
  return { ...choice, default: shape };
}

/** The default shapes a turn asks for, read as it is sent. */
export function defaultOutputShapes(): Record<OutputShapeMode, OutputRatioPresetId | null> {
  const choices = storedChoices();
  return { image: choices.image.default, video: choices.video.default };
}
