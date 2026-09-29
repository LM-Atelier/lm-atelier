import { Scan } from "lucide-react";
import { correctedSize, isFrame, isUnchanged } from "./studioPerspective";
import type { StudioPerspective } from "./types";

/** Correcting the perspective: four corners placed on the picture, then one press.
 *
 * The corners are dragged on the canvas itself, onto the corners of something
 * that should be square. The panel says what applying will make, and holds it
 * back until the corners have moved and still go around a four-sided shape.
 */
export function StudioPerspectiveTool({
  corners,
  size,
  busy,
  onReset,
  onApply,
}: {
  corners: StudioPerspective;
  /** The picture's own size, where the corners start. */
  size: { width: number; height: number };
  busy: boolean;
  onReset: () => void;
  onApply: (corners: StudioPerspective) => void;
}) {
  const unchanged = isUnchanged(corners, size.width, size.height);
  const frame = isFrame(corners);
  const ready = !unchanged && frame && !busy;
  const made = correctedSize(corners);
  return (
    <div className="studio-tool-options">
      <small>
        Drag each corner onto a corner of something that should be square, such as a page or the
        front of a building. Applying makes it the whole picture, upright. With the keyboard, the
        arrows move the canvas caret and Enter picks up the nearest corner and puts it down.
      </small>
      <small aria-live="polite">
        {busy
          ? "Applying…"
          : unchanged
            ? "The corners start at the picture's own."
            : frame
              ? `Makes a picture ${made.width} by ${made.height} pixels.`
              : "Keep the corners in their places, with no side crossing another."}
      </small>
      <div className="studio-transform-actions">
        <button type="button" className="secondary compact-button" aria-disabled={unchanged}
          onClick={() => {
            if (!unchanged) onReset();
          }}>
          Reset corners
        </button>
        <button
          type="button"
          className="secondary compact-button"
          // Not disabled: a focused button that becomes disabled drops focus to
          // the page, and a keyboard user would lose their place.
          aria-disabled={!ready}
          onClick={() => {
            if (ready) onApply(corners);
          }}
        >
          <Scan size={14} aria-hidden="true" /> Apply the correction
        </button>
      </div>
    </div>
  );
}
