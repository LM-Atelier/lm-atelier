import { Crop } from "lucide-react";
import { useMemo } from "react";
import { maskBounds, type MaskRaster } from "./studioMasks";
import type { StudioCropBox } from "./types";

/** Keeping one part of the picture, cut exactly and without a model.
 *
 * The box is drawn on the canvas, and a new one replaces the last, so what is
 * shown here is always what the press will keep. Nothing happens until then:
 * cropping discards the rest, which is worth one deliberate press.
 */
export function StudioCropTool({
  mask,
  maskVersion,
  busy,
  onCrop,
}: {
  /** What has been drawn, at the picture's own size. */
  mask: MaskRaster | null;
  /** Changes when a box is drawn, since the raster changes in place. */
  maskVersion: number;
  busy: boolean;
  onCrop: (box: StudioCropBox) => void;
}) {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const box = useMemo(() => (mask ? maskBounds(mask) : null), [mask, maskVersion]);
  const size = mask ? { width: mask.width, height: mask.height } : null;
  const ready = box !== null && !busy;
  return (
    <div className="studio-tool-options">
      <small>
        {busy
          ? "Applying…"
          : box && size
            ? `Keeps ${box.width} × ${box.height} of ${size.width} × ${size.height}.`
            : "Drag a box over the part of the picture to keep."}
      </small>
      <button
        type="button"
        className="secondary compact-button"
        // Not disabled: a focused button that becomes disabled drops focus to
        // the page, and a keyboard user would lose their place.
        aria-disabled={!ready}
        onClick={() => {
          if (ready && box) onCrop(box);
        }}
      >
        <Crop size={14} aria-hidden="true" /> Crop to the box
      </button>
    </div>
  );
}
