import { Droplet } from "lucide-react";
import type { MaskRaster } from "./studioMasks";
import { useMarkedArea } from "./useMarkedArea";

/** How far a blur may spread, in the picture's own pixels. */
export const MAX_BLUR_RADIUS = 100;

/** Blurring a marked part of the picture, exactly and without a model.
 *
 * The brush marks the area, and so does any selection made with the other
 * tools, since they all draw the same selection. The edge is feathered as the
 * selection's feather says, so the blur fades out rather than stopping at a
 * line. Nothing changes until the press, and the rest of the picture is left
 * exactly as it was.
 */
export function StudioBlurTool({
  mask,
  maskVersion,
  featherPx,
  radius,
  busy,
  onRadius,
  onBlur,
}: {
  /** What has been marked, at the picture's own size. */
  mask: MaskRaster | null;
  /** Changes when the marking changes, since the raster changes in place. */
  maskVersion: number;
  featherPx: number;
  radius: number;
  busy: boolean;
  onRadius: (radius: number) => void;
  /** Receives the marked area as a PNG selection, feathered. */
  onBlur: (selection: Blob) => void;
}) {
  const { marked, preparing, refusal, handOver } = useMarkedArea(mask, maskVersion, featherPx);
  const ready = marked && !busy && !preparing;

  return (
    <div className="studio-tool-options">
      <label className="studio-adjust-slider">
        <span>
          <strong>Blur strength</strong> {radius} px
        </span>
        <input
          type="range"
          min={1}
          max={MAX_BLUR_RADIUS}
          step={1}
          value={radius}
          aria-label="Blur strength"
          onChange={(event) => onRadius(Number(event.target.value))}
        />
      </label>
      <small role={refusal ? "alert" : undefined}>
        {refusal ??
          (busy || preparing
            ? "Applying…"
            : marked
              ? "Blurs what is marked and leaves the rest of the picture exactly as it is."
              : "Brush over the part to blur, or select it with another tool first.")}
      </small>
      <button
        type="button"
        className="secondary compact-button"
        // Not disabled: a focused button that becomes disabled drops focus to
        // the page, and a keyboard user would lose their place.
        aria-disabled={!ready}
        onClick={() => {
          if (ready) handOver(onBlur);
        }}
      >
        <Droplet size={14} aria-hidden="true" /> Blur the marked area
      </button>
    </div>
  );
}
