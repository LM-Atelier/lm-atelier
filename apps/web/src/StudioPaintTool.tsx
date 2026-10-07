import { Paintbrush } from "lucide-react";
import type { MaskRaster } from "./studioMasks";
import { PAINT_SWATCHES } from "./studioPaint";
import { useMarkedArea } from "./useMarkedArea";

/** Painting over a marked part of the picture in one color, without a model.
 *
 * The brush marks the area, as it does for a blur, and any other selection
 * tool can mark it first. While this tool is chosen the marking is shown in the
 * paint's own color and opacity, so what is on the canvas is what the press
 * lays down, apart from the feather that softens its edge. Black at full
 * opacity covers what is under it completely.
 */
export function StudioPaintTool({
  mask,
  maskVersion,
  featherPx,
  color,
  opacity,
  busy,
  onColor,
  onOpacity,
  onPaint,
}: {
  /** What has been marked, at the picture's own size. */
  mask: MaskRaster | null;
  /** Changes when the marking changes, since the raster changes in place. */
  maskVersion: number;
  featherPx: number;
  /** As "#rrggbb". */
  color: string;
  /** From 1 to 100 percent. */
  opacity: number;
  busy: boolean;
  onColor: (color: string) => void;
  onOpacity: (opacity: number) => void;
  /** Receives the marked area as a PNG selection, feathered. */
  onPaint: (selection: Blob) => void;
}) {
  const { marked, preparing, refusal, handOver } = useMarkedArea(mask, maskVersion, featherPx);
  const ready = marked && !busy && !preparing;

  return (
    <div className="studio-tool-options">
      <div className="segmented studio-paint-swatches" role="group" aria-label="Paint color">
        {PAINT_SWATCHES.map((swatch) => (
          <button
            key={swatch.color}
            type="button"
            aria-label={swatch.label}
            aria-pressed={color === swatch.color}
            className={color === swatch.color ? "active" : ""}
            onClick={() => onColor(swatch.color)}
          >
            <span className="studio-paint-swatch" style={{ backgroundColor: swatch.color }} />
          </button>
        ))}
      </div>
      <label className="studio-adjust-slider">
        <span>
          <strong>Another color</strong>
        </span>
        <input type="color" value={color} aria-label="Another color"
          onChange={(event) => onColor(event.target.value.toLowerCase())} />
      </label>
      <label className="studio-adjust-slider">
        <span>
          <strong>Opacity</strong> {opacity}%
        </span>
        <input
          type="range"
          min={1}
          max={100}
          step={1}
          value={opacity}
          aria-label="Opacity"
          onChange={(event) => onOpacity(Number(event.target.value))}
        />
      </label>
      <small role={refusal ? "alert" : undefined}>
        {refusal ??
          (busy || preparing
            ? "Applying…"
            : marked
              ? "The marking shows the paint as it will be laid down."
              : "Brush over the part to paint, or select it with another tool first.")}
      </small>
      <button
        type="button"
        className="secondary compact-button"
        // Not disabled: a focused button that becomes disabled drops focus to
        // the page, and a keyboard user would lose their place.
        aria-disabled={!ready}
        onClick={() => {
          if (ready) handOver(onPaint);
        }}
      >
        <Paintbrush size={14} aria-hidden="true" /> Paint the marked area
      </button>
    </div>
  );
}
