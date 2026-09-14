import type { Dispatch } from "react";
import type { StudioToolAction, StudioToolState } from "./studioToolState";

/** The panel's tool-specific control: what this tool needs said before it runs.
 *
 * Extend and Enhance are a drag or a number, Text is the words before and
 * after, and every other tool is described in words. Kept apart from the
 * studio view so each tool's control reads in one place.
 */
export function StudioToolOptions({
  tools,
  dispatch,
  instruction,
  onInstructionChange,
}: {
  tools: StudioToolState;
  dispatch: Dispatch<StudioToolAction>;
  instruction: string;
  onInstructionChange: (value: string) => void;
}) {
  if (tools.kind === "extend") {
    return (
      <div className="studio-tool-options">
        <span>
          <strong>Extend by</strong>
        </span>
        <small>
          {Object.values(tools.margins).some(Boolean)
            ? (["top", "right", "bottom", "left"] as const)
                .filter((side) => tools.margins[side] > 0)
                .map((side) => `${side} ${Math.round(tools.margins[side] * 100)}%`)
                .join(", ")
            : "Drag an edge of the picture outward, or use the arrow keys on one."}
        </small>
        <button
          className="secondary compact-button"
          onClick={() => dispatch({ type: "clear-margins" })}
        >
          Reset edges
        </button>
      </div>
    );
  }
  if (tools.kind === "enhance") {
    return (
      <label>
        <span>
          <strong>Enlarge by</strong> {tools.upscaleFactor}x
        </span>
        <input
          type="range"
          min={1}
          max={8}
          step={1}
          value={tools.upscaleFactor}
          onChange={(event) =>
            dispatch({ type: "set-upscale-factor", factor: Number(event.target.value) })
          }
        />
      </label>
    );
  }
  if (tools.kind === "text") {
    return (
      <div className="studio-tool-options">
        <small>
          Draw a box around the words. Only what is inside the box changes.
        </small>
        <label>
          <span>
            <strong>Words there now</strong> (optional)
          </span>
          <input
            type="text"
            value={tools.currentWords}
            onChange={(event) => dispatch({ type: "set-current-words", words: event.target.value })}
          />
        </label>
        <label>
          <span>
            <strong>Replace with</strong>
          </span>
          <input
            type="text"
            value={tools.newWords}
            onChange={(event) => dispatch({ type: "set-new-words", words: event.target.value })}
          />
        </label>
      </div>
    );
  }
  return (
    <label>
      <span>
        <strong>
          {tools.kind === "instruct" ? "Describe the edit" : "Describe the change here"}
        </strong>
      </span>
      <textarea
        rows={4}
        value={instruction}
        placeholder="e.g. make it a watercolor painting"
        onChange={(event) => onInstructionChange(event.target.value)}
      />
    </label>
  );
}
