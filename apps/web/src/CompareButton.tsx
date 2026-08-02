import { useState } from "react";
import { Columns2 } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";

/** Slide between an edit's source and its result.
 *
 * The result is clipped over the source, so the slider literally wipes one
 * image across the other - differences show where they are, not side by side
 * where the eye has to carry them.
 */
export function CompareButton({ before, after }: { before: string; after: string }) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState(50);
  return (
    <>
      <button type="button" onClick={() => { setPosition(50); setOpen(true); }}>Compare</button>
      {open && (
        <AccessibleDialog
          title="Compare with the source"
          eyebrow="Before and after"
          closeLabel="Close comparison"
          onClose={() => setOpen(false)}
          className="compare-dialog"
        >
          <div className="compare-stage">
            <img src={before} alt="The source before the edit" />
            <img
              src={after}
              alt="The edited result"
              style={{ clipPath: `inset(0 0 0 ${position}%)` }}
            />
            <div className="compare-divider" style={{ left: `${position}%` }} aria-hidden="true" />
          </div>
          <label className="compare-slider">
            <span>Reveal</span>
            <input
              type="range"
              min={0}
              max={100}
              value={position}
              aria-label="Comparison position"
              onChange={(event) => setPosition(Number(event.target.value))}
            />
          </label>
          <footer className="compare-legend">
            <span><Columns2 size={14} aria-hidden="true" /> Left of the divider is the source; right is the result.</span>
          </footer>
        </AccessibleDialog>
      )}
    </>
  );
}
