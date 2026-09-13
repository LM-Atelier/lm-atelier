import type { Dispatch } from "react";
import type { StudioToolAction, StudioToolState } from "./studioToolState";

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
