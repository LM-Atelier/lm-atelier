import { Frame } from "lucide-react";
import { useState } from "react";
import { StudioAnchorPicker } from "./StudioAnchorPicker";
import type { StudioCanvasAnchor, StudioCanvasChange, StudioPictureSize } from "./types";

const FILLS: Array<{ fill: StudioCanvasChange["fill"]; label: string }> = [
  { fill: "transparent", label: "Transparent" },
  { fill: "white", label: "White" },
  { fill: "black", label: "Black" },
];

function whole(text: string): number | null {
  const value = Number(text);
  return text.trim() !== "" && Number.isInteger(value) && value >= 1 ? value : null;
}

/** Changing the canvas around the picture, without resampling it.
 *
 * A larger canvas adds room on the sides away from where the picture is
 * anchored, filled as chosen; a smaller one trims from those sides. The
 * picture's own pixels are never scaled, so this is the tool for making room
 * or squaring a picture up, and Resize is the one for changing its size.
 * Mounted afresh for each picture, so it always starts from that one's size.
 */
export function StudioCanvasSizeTool({
  size,
  busy,
  onChange,
}: {
  /** The picture's size now, in its own pixels. */
  size: StudioPictureSize;
  busy: boolean;
  onChange: (canvas: StudioCanvasChange) => void;
}) {
  const [width, setWidth] = useState(String(size.width));
  const [height, setHeight] = useState(String(size.height));
  const [anchor, setAnchor] = useState<StudioCanvasAnchor>("center");
  const [fill, setFill] = useState<StudioCanvasChange["fill"]>("transparent");
  const wanted = { width: whole(width), height: whole(height) };
  const valid = wanted.width !== null && wanted.height !== null;
  const unchanged = wanted.width === size.width && wanted.height === size.height;
  const trims = valid && ((wanted.width ?? 0) < size.width || (wanted.height ?? 0) < size.height);
  const ready = valid && !unchanged && !busy;

  return (
    <div className="studio-tool-options">
      <div className="studio-resize-size">
        <label>
          <span>Width</span>
          <input type="number" min={1} step={1} inputMode="numeric" value={width}
            onChange={(event) => setWidth(event.target.value)} />
        </label>
        <label>
          <span>Height</span>
          <input type="number" min={1} step={1} inputMode="numeric" value={height}
            onChange={(event) => setHeight(event.target.value)} />
        </label>
      </div>
      <StudioAnchorPicker value={anchor} label="Where the picture sits" onChange={setAnchor} />
      <div className="segmented" role="group" aria-label="What fills the new room">
        {FILLS.map(({ fill: choice, label }) => (
          <button
            key={choice}
            type="button"
            aria-pressed={fill === choice}
            className={fill === choice ? "active" : ""}
            onClick={() => setFill(choice)}
          >
            {label}
          </button>
        ))}
      </div>
      <small>
        {busy
          ? "Applying…"
          : !valid
            ? "Enter a width and a height of at least one pixel."
            : unchanged
              ? `Now ${size.width} × ${size.height} pixels. The picture itself is never resized here.`
              : `The canvas goes from ${size.width} × ${size.height} to ${wanted.width} × ${wanted.height}.` +
                (trims ? " What falls outside it is cut away." : "")}
      </small>
      <button
        type="button"
        className="secondary compact-button"
        // Not disabled: a focused button that becomes disabled drops focus to
        // the page, and a keyboard user would lose their place.
        aria-disabled={!ready}
        onClick={() => {
          if (ready && wanted.width !== null && wanted.height !== null) {
            onChange({ width: wanted.width, height: wanted.height, anchor, fill });
          }
        }}
      >
        <Frame size={14} aria-hidden="true" /> Change the canvas
      </button>
    </div>
  );
}
