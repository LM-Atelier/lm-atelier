/** The colors offered at a press; any other can be picked beside them. */
export const PAINT_SWATCHES: ReadonlyArray<{ color: string; label: string }> = [
  { color: "#000000", label: "Black" },
  { color: "#ffffff", label: "White" },
  { color: "#e53935", label: "Red" },
  { color: "#fdd835", label: "Yellow" },
  { color: "#43a047", label: "Green" },
  { color: "#1e88e5", label: "Blue" },
];

/** A "#rrggbb" color as the red, green and blue levels the canvas tints with. */
export function paintRgb(color: string): [number, number, number] {
  return [1, 3, 5].map((start) => Number.parseInt(color.slice(start, start + 2), 16)) as [number, number, number];
}
