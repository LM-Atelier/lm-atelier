import type { Dispatch } from "react";
import { Brush, Eraser, Lasso, PaintBucket, Square, Wand2 } from "lucide-react";
import {
  isSelectionKind,
  type SelectionKind,
  type StudioToolAction,
  type StudioToolState,
} from "./studioToolState";

/** The six ways of drawing the one selection, offered once Select is chosen. */
const SELECTION_WAYS: Array<{ kind: SelectionKind; label: string; icon: typeof Brush }> = [
  { kind: "brush", label: "Brush a selection", icon: Brush },
  { kind: "eraser", label: "Erase from the selection", icon: Eraser },
  { kind: "rect", label: "Select a rectangle", icon: Square },
  { kind: "lasso", label: "Lasso a selection", icon: Lasso },
  { kind: "bucket", label: "Fill an area of the selection", icon: PaintBucket },
  { kind: "wand", label: "Select similar colors", icon: Wand2 },
];

/** The control that depends on how the selection is drawn.
 *
 * A brush has a size. The paint bucket and the magic wand have none: a click
 * either adds to the selection or takes away from it, and the wand also needs
 * to know how close a color must be to the clicked one. When the picture's
 * colors cannot be read, the wand says so rather than doing nothing.
 */
export function StudioSelectionTool({
  tools,
  dispatch,
  colorsUnreadable,
}: {
  tools: StudioToolState;
  dispatch: Dispatch<StudioToolAction>;
  colorsUnreadable: boolean;
}) {
  // The words tool draws a box, which has no size to set.
  if (tools.kind === "text") return null;
  if (tools.kind !== "bucket" && tools.kind !== "wand") {
    return (
      <label>
        Brush size
        <input
          type="range"
          min={1}
          max={200}
          value={tools.brushRadius}
          onChange={(event) =>
            dispatch({ type: "set-brush-radius", radius: Number(event.target.value) })}
        />
      </label>
    );
  }
  return (
    <>
      <div className="segmented" role="group" aria-label="What a click does">
        <button
          type="button"
          aria-pressed={tools.selectionMode === "add"}
          className={tools.selectionMode === "add" ? "active" : ""}
          onClick={() => dispatch({ type: "set-selection-mode", mode: "add" })}
        >
          Add to selection
        </button>
        <button
          type="button"
          aria-pressed={tools.selectionMode === "remove"}
          className={tools.selectionMode === "remove" ? "active" : ""}
          onClick={() => dispatch({ type: "set-selection-mode", mode: "remove" })}
        >
          Take from selection
        </button>
      </div>
      {tools.kind === "wand" && (
        <label>
          Color tolerance
          <input
            type="range"
            min={0}
            max={128}
            value={tools.colorTolerance}
            onChange={(event) =>
              dispatch({ type: "set-color-tolerance", tolerance: Number(event.target.value) })}
          />
        </label>
      )}
      {tools.kind === "wand" && colorsUnreadable && (
        <p role="alert">
          This picture's colors cannot be read here, so the wand cannot select by color.
        </p>
      )}
    </>
  );
}

/** What the panel offers for a selection: the way it is drawn while Select is
 * in hand, how it is drawn, what to do with it as a whole, and how much of the
 * picture it covers. */
export function StudioSelectionControls({
  tools,
  dispatch,
  colorsUnreadable,
  coverage,
}: {
  tools: StudioToolState;
  dispatch: Dispatch<StudioToolAction>;
  colorsUnreadable: boolean;
  /** The selected fraction of the picture; 0 when nothing is selected. */
  coverage: number;
}) {
  return (
    <div className="studio-selection-controls">
      {isSelectionKind(tools.kind) && (
        <div className="studio-selection-ways" role="group" aria-label="How to select">
          {SELECTION_WAYS.map(({ kind, label, icon: Icon }) => (
            <button
              key={kind}
              type="button"
              className={`icon-button ${tools.kind === kind ? "selected" : ""}`}
              aria-label={label}
              aria-pressed={tools.kind === kind}
              title={label}
              onClick={() => dispatch({ type: "select-tool", kind })}
            >
              <Icon size={16} aria-hidden="true" />
            </button>
          ))}
        </div>
      )}
      <StudioSelectionTool tools={tools} dispatch={dispatch} colorsUnreadable={colorsUnreadable} />
      <div className="row-actions">
        <button
          className="secondary compact-button"
          onClick={() => dispatch({ type: "invert" })}
        >
          Invert
        </button>
        <button
          className="secondary compact-button"
          onClick={() => dispatch({ type: "feather" })}
        >
          Soften edges
        </button>
        <button
          className="secondary compact-button"
          onClick={() => dispatch({ type: "clear" })}
        >
          Clear
        </button>
      </div>
      <small>
        {coverage > 0
          ? `${(coverage * 100).toFixed(1)}% of the image selected`
          : "Nothing selected yet - paint over what you want to change."}
      </small>
    </div>
  );
}
