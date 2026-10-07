import { Scaling } from "lucide-react";
import { useState } from "react";
import type { StudioPictureSize } from "./types";

function whole(text: string): number | null {
  const value = Number(text);
  return text.trim() !== "" && Number.isInteger(value) && value >= 1 ? value : null;
}

function inProportion(value: number, from: number, to: number): number {
  return Math.max(1, Math.round((value * to) / from));
}

/** Changing the picture's size in pixels, resampled once and without a model.
 *
 * The sides stay in proportion unless that is switched off, so a picture is
 * not stretched by accident. Nothing happens until the press, and the new
 * size arrives as the next step, with the picture before it still there.
 * Mounted afresh for each picture, so it always starts from that one's size.
 */
export function StudioResizeTool({
  size,
  busy,
  onResize,
}: {
  /** The picture's size now, in its own pixels. */
  size: StudioPictureSize;
  busy: boolean;
  onResize: (size: StudioPictureSize) => void;
}) {
  const [width, setWidth] = useState(String(size.width));
  const [height, setHeight] = useState(String(size.height));
  const [keepProportions, setKeepProportions] = useState(true);
  const wanted = { width: whole(width), height: whole(height) };
  const valid = wanted.width !== null && wanted.height !== null;
  const unchanged = wanted.width === size.width && wanted.height === size.height;
  const grows = valid && ((wanted.width ?? 0) > size.width || (wanted.height ?? 0) > size.height);
  const ready = valid && !unchanged && !busy;

  const changeWidth = (text: string) => {
    setWidth(text);
    const value = whole(text);
    if (keepProportions && value !== null) {
      setHeight(String(inProportion(value, size.width, size.height)));
    }
  };
  const changeHeight = (text: string) => {
    setHeight(text);
    const value = whole(text);
    if (keepProportions && value !== null) {
      setWidth(String(inProportion(value, size.height, size.width)));
    }
  };

  return (
    <div className="studio-tool-options">
      <div className="studio-resize-size">
        <label>
          <span>Width</span>
          <input type="number" min={1} step={1} inputMode="numeric" value={width}
            onChange={(event) => changeWidth(event.target.value)} />
        </label>
        <label>
          <span>Height</span>
          <input type="number" min={1} step={1} inputMode="numeric" value={height}
            onChange={(event) => changeHeight(event.target.value)} />
        </label>
      </div>
      <label>
        <input type="checkbox" checked={keepProportions}
          onChange={(event) => setKeepProportions(event.target.checked)} />
        <span>Keep proportions</span>
      </label>
      <small>
        {busy
          ? "Applying…"
          : !valid
            ? "Enter a width and a height of at least one pixel."
            : unchanged
              ? `Now ${size.width} × ${size.height} pixels.`
              : `${size.width} × ${size.height} becomes ${wanted.width} × ${wanted.height}.` +
                (grows ? " Growing only spreads the pixels it has; Enhance adds detail." : "")}
      </small>
      <button
        type="button"
        className="secondary compact-button"
        // Not disabled: a focused button that becomes disabled drops focus to
        // the page, and a keyboard user would lose their place.
        aria-disabled={!ready}
        onClick={() => {
          if (ready && wanted.width !== null && wanted.height !== null) {
            onResize({ width: wanted.width, height: wanted.height });
          }
        }}
      >
        <Scaling size={14} aria-hidden="true" /> Resize
      </button>
    </div>
  );
}
