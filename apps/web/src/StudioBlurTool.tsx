import { Droplet, Grid3x3 } from "lucide-react";
import type { MaskRaster } from "./studioMasks";
import { useMarkedArea } from "./useMarkedArea";

/** How far a blur may spread, in the picture's own pixels. */
export const MAX_BLUR_RADIUS = 100;
/** The largest block a pixelation makes, in the picture's own pixels. */
export const MAX_PIXEL_BLOCK = 100;

/** Blurring or pixelating a marked part of the picture, exactly and without a model.
 *
 * The brush marks the area, and so does any selection made with the other
 * tools, since they all draw the same selection. A blur softens it; a
 * pixelation breaks it into square blocks, each the mean of the pixels it
 * covers, laid from the picture's top-left corner. The edge is feathered as
 * the selection's feather says, so the change fades out rather than stopping
 * at a line. Nothing changes until the press, and the rest of the picture is
 * left exactly as it was.
 */
export function StudioBlurTool({
  mask,
  maskVersion,
  featherPx,
  style,
  radius,
  block,
  busy,
  onStyle,
  onRadius,
  onBlock,
  onBlur,
  onPixelate,
}: {
  /** What has been marked, at the picture's own size. */
  mask: MaskRaster | null;
  /** Changes when the marking changes, since the raster changes in place. */
  maskVersion: number;
  featherPx: number;
  style: "blur" | "pixelate";
  radius: number;
  block: number;
  busy: boolean;
  onStyle: (style: "blur" | "pixelate") => void;
  onRadius: (radius: number) => void;
  onBlock: (block: number) => void;
  /** Each receives the marked area as a PNG selection, feathered. */
  onBlur: (selection: Blob) => void;
  onPixelate: (selection: Blob) => void;
}) {
  const { marked, preparing, refusal, handOver } = useMarkedArea(mask, maskVersion, featherPx);
  const ready = marked && !busy && !preparing;
  const blurring = style === "blur";

  return (
    <div className="studio-tool-options">
      <div className="segmented" role="group" aria-label="Effect">
        <button type="button" aria-pressed={blurring} className={blurring ? "active" : ""}
          onClick={() => onStyle("blur")}>
          Blur
        </button>
        <button type="button" aria-pressed={!blurring} className={blurring ? "" : "active"}
          onClick={() => onStyle("pixelate")}>
          Pixelate
        </button>
      </div>
      {blurring ? (
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
      ) : (
        <label className="studio-adjust-slider">
          <span>
            <strong>Block size</strong> {block} px
          </span>
          <input
            type="range"
            min={2}
            max={MAX_PIXEL_BLOCK}
            step={1}
            value={block}
            aria-label="Block size"
            onChange={(event) => onBlock(Number(event.target.value))}
          />
        </label>
      )}
      <small role={refusal ? "alert" : undefined}>
        {refusal ??
          (busy || preparing
            ? "Applying…"
            : marked
              ? `${blurring ? "Blurs" : "Pixelates"} what is marked and leaves the rest of the picture exactly as it is.`
              : `Brush over the part to ${blurring ? "blur" : "pixelate"}, or select it with another tool first.`)}
      </small>
      <button
        type="button"
        className="secondary compact-button"
        // Not disabled: a focused button that becomes disabled drops focus to
        // the page, and a keyboard user would lose their place.
        aria-disabled={!ready}
        onClick={() => {
          if (ready) handOver(blurring ? onBlur : onPixelate);
        }}
      >
        {blurring ? (
          <>
            <Droplet size={14} aria-hidden="true" /> Blur the marked area
          </>
        ) : (
          <>
            <Grid3x3 size={14} aria-hidden="true" /> Pixelate the marked area
          </>
        )}
      </button>
    </div>
  );
}
