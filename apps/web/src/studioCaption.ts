import type { ImagePoint, PointerTool, ToolPreview } from "./studioTools";
import type { StudioCanvasAnchor } from "./types";

/** Words to lay over the picture, and how they look. */
export interface StudioCaption {
  text: string;
  font: CaptionFont;
  /** The letters' height as a share of the picture's height, in percent. */
  sizePercent: number;
  /** As "#rrggbb". */
  color: string;
  bold: boolean;
  /** A thin edge in the opposite shade, so the words read over a busy picture. */
  outline: boolean;
  /** A soft dark shadow below and to the right, lifting the words off the picture. */
  shadow: boolean;
  anchor: StudioCanvasAnchor;
  /** How far the words have been dragged from where the anchor puts them. */
  shift: CaptionShift;
  /** How far the words are turned about their middle, in whole degrees, clockwise when positive. */
  turn: number;
}

/** A distance across the picture as shares of its width and its height, so it
 * means the same at any size the picture is shown at. */
export type CaptionShift = { x: number; y: number };

/** The farthest the words turn either way. */
export const CAPTION_TURN_LIMIT = 180;

export type CaptionFont = "sans" | "serif" | "mono";

/** The typefaces the app already ships, so the words look the same on every machine. */
export const CAPTION_FONTS: Record<CaptionFont, string> = {
  sans: '"Inter", system-ui, sans-serif',
  serif: '"Source Serif 4", Georgia, serif',
  mono: '"JetBrains Mono", monospace',
};

export const DEFAULT_CAPTION: StudioCaption = {
  text: "",
  font: "sans",
  sizePercent: 8,
  color: "#ffffff",
  bold: true,
  outline: true,
  shadow: false,
  anchor: "bottom",
  shift: { x: 0, y: 0 },
  turn: 0,
};

const ANCHOR_PLACES: Record<StudioCanvasAnchor, [number, number]> = {
  top_left: [0, 0],
  top: [1, 0],
  top_right: [2, 0],
  left: [0, 1],
  center: [1, 1],
  right: [2, 1],
  bottom_left: [0, 2],
  bottom: [1, 2],
  bottom_right: [2, 2],
};

/** Where the words go: the x they are aligned to, how, and the top of each line.
 *
 * They keep a margin of four percent of the picture's shorter side from any
 * edge they are anchored to, and a block of lines is placed as one.
 */
export function captionLayout(
  width: number,
  height: number,
  lineCount: number,
  fontPx: number,
  anchor: StudioCanvasAnchor,
): { x: number; align: CanvasTextAlign; tops: number[] } {
  const margin = Math.round(Math.min(width, height) * 0.04);
  const lineHeight = lineHeightPx(fontPx);
  const block = lineHeight * lineCount;
  const [column, row] = ANCHOR_PLACES[anchor];
  const x = column === 0 ? margin : column === 1 ? Math.round(width / 2) : width - margin;
  const align: CanvasTextAlign = column === 0 ? "left" : column === 1 ? "center" : "right";
  const top = row === 0 ? margin : row === 1 ? Math.round((height - block) / 2) : height - margin - block;
  return { x, align, tops: Array.from({ length: lineCount }, (_, index) => top + index * lineHeight) };
}

/** The letters' size in the picture's own pixels. */
export function captionFontPx(height: number, sizePercent: number): number {
  return Math.max(8, Math.round((height * sizePercent) / 100));
}

/** From the top of one line to the top of the next. */
function lineHeightPx(fontPx: number): number {
  return Math.round(fontPx * 1.2);
}

/** The middle of the block of lines, which the words turn about.
 *
 * `widths` are the lines' measured widths; the block is as wide as the widest
 * and sits on the side of `x` its alignment says.
 */
export function captionMiddle(
  layout: { x: number; align: CanvasTextAlign; tops: number[] },
  widths: number[],
  fontPx: number,
): ImagePoint {
  const width = Math.max(0, ...widths);
  const left = layout.align === "left" ? layout.x : layout.align === "right" ? layout.x - width : layout.x - width / 2;
  return { x: left + width / 2, y: layout.tops[0] + (lineHeightPx(fontPx) * layout.tops.length) / 2 };
}

/** Cast a shadow from what is drawn next, or stop casting one. */
function castShadow(context: CanvasRenderingContext2D, fontPx: number | null): void {
  context.shadowColor = fontPx === null ? "rgba(0, 0, 0, 0)" : "rgba(0, 0, 0, 0.55)";
  context.shadowBlur = fontPx === null ? 0 : Math.max(1, Math.round(fontPx / 8));
  const offset = fontPx === null ? 0 : Math.max(1, Math.round(fontPx / 20));
  context.shadowOffsetX = offset;
  context.shadowOffsetY = offset;
}

function isLight(color: string): boolean {
  const [red, green, blue] = [1, 3, 5].map((start) => Number.parseInt(color.slice(start, start + 2), 16));
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue > 127.5;
}

/** The words drawn on transparency at the picture's size, or null without a canvas.
 *
 * The same drawing is shown on the canvas and uploaded when the words are
 * added, so what was previewed is exactly what is laid down.
 */
export async function drawCaption(width: number, height: number, caption: StudioCaption): Promise<HTMLCanvasElement | null> {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  const fontPx = captionFontPx(height, caption.sizePercent);
  const font = `${caption.bold ? 700 : 400} ${fontPx}px ${CAPTION_FONTS[caption.font]}`;
  try {
    // The shipped typefaces load on first use; drawing before they arrive
    // would draw a fallback the second drawing would not match.
    await document.fonts.load(font, caption.text);
  } catch {
    // A font that cannot load leaves the fallback, which is at least the same
    // for the preview and for what is added.
  }
  const lines = caption.text.split("\n");
  const layout = captionLayout(width, height, lines.length, fontPx, caption.anchor);
  context.font = font;
  context.textAlign = layout.align;
  context.textBaseline = "top";
  context.lineJoin = "round";
  context.fillStyle = caption.color;
  context.strokeStyle = isLight(caption.color) ? "#000000" : "#ffffff";
  context.lineWidth = Math.max(1, Math.round(fontPx / 10));
  if (caption.shift.x !== 0 || caption.shift.y !== 0 || caption.turn !== 0) {
    // Turned about their middle, then moved as far as they were dragged. A
    // shadow is cast in the picture's own directions, so it still falls down
    // and to the right of turned words.
    const middle = captionMiddle(layout, lines.map((line) => context.measureText(line).width), fontPx);
    context.translate(middle.x + caption.shift.x * width, middle.y + caption.shift.y * height);
    context.rotate((caption.turn * Math.PI) / 180);
    context.translate(-middle.x, -middle.y);
  }
  lines.forEach((line, index) => {
    // The shadow falls once, from the words' outer edge: the outline when
    // there is one, and not again from the letters drawn inside it.
    castShadow(context, caption.shadow ? fontPx : null);
    if (caption.outline) {
      context.strokeText(line, layout.x, layout.tops[index]);
      castShadow(context, null);
    }
    context.fillText(line, layout.x, layout.tops[index]);
  });
  return canvas;
}

/** A shift kept within one picture's width and height either way: past that
 * the words are wholly off the picture, wherever they started. */
export function boundedShift(shift: CaptionShift): CaptionShift {
  const bound = (value: number) => (Number.isFinite(value) ? Math.min(1, Math.max(-1, value)) : 0);
  return { x: bound(shift.x), y: bound(shift.y) };
}

/** Where a drag leaves the words: moved from where it took hold of them by
 * `by`, in the picture's own pixels, to a whole pixel. */
export function draggedShift(
  hold: CaptionShift,
  by: ImagePoint,
  size: { width: number; height: number },
): CaptionShift {
  return boundedShift({
    x: Math.round(hold.x * size.width + by.x) / size.width,
    y: Math.round(hold.y * size.height + by.y) / size.height,
  });
}

/** What a drag of the words tells the Studio: that it took hold of them, how
 * far it has come since, in the picture's own pixels, and how it ended. */
export interface CaptionDrag {
  hold(): void;
  drag(by: ImagePoint): void;
  letGo(): void;
  cancel(): void;
}

/** Dragging the words about the picture.
 *
 * A press anywhere takes hold of the words where they are, and a drag moves
 * them as far as the pointer moves, so they never jump to where the press
 * landed; the canvas's keyboard caret moves them the same way. Each step is
 * reported as it happens, so the picture shows the words where they are
 * going, and a cancelled drag puts them back where it found them. The
 * selection is never touched.
 */
export class CaptionMoveTool implements PointerTool {
  readonly appliesWhileMoving = false;
  readonly cursor = "move";
  private from: ImagePoint | null = null;

  constructor(private readonly words: CaptionDrag) {}

  down(point: ImagePoint): void {
    this.from = point;
    this.words.hold();
  }

  move(point: ImagePoint): void {
    if (this.from) this.words.drag({ x: point.x - this.from.x, y: point.y - this.from.y });
  }

  up(point: ImagePoint): boolean {
    if (!this.from) return false;
    this.words.drag({ x: point.x - this.from.x, y: point.y - this.from.y });
    this.words.letGo();
    this.from = null;
    // The selection is as it was, so there is no stroke to finish.
    return false;
  }

  cancel(): void {
    if (this.from) this.words.cancel();
    this.from = null;
  }

  preview(): ToolPreview {
    return { kind: "none" };
  }
}
