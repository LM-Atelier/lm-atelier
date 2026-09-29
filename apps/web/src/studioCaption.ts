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
  anchor: StudioCanvasAnchor;
}

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
  anchor: "bottom",
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
  const lineHeight = Math.round(fontPx * 1.2);
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
  lines.forEach((line, index) => {
    if (caption.outline) context.strokeText(line, layout.x, layout.tops[index]);
    context.fillText(line, layout.x, layout.tops[index]);
  });
  return canvas;
}
