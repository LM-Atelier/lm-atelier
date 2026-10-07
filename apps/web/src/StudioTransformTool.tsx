import { FlipHorizontal2, FlipVertical2, RotateCcw, RotateCw } from "lucide-react";
import { STRAIGHTEN_LIMIT } from "./studioStraighten";
import type { StudioLocalEditOperation } from "./types";

const OPERATIONS: Array<{
  operation: StudioLocalEditOperation;
  label: string;
  icon: typeof RotateCw;
}> = [
  { operation: "rotate_counterclockwise", label: "Rotate left", icon: RotateCcw },
  { operation: "rotate_clockwise", label: "Rotate right", icon: RotateCw },
  { operation: "flip_horizontal", label: "Flip horizontally", icon: FlipHorizontal2 },
  { operation: "flip_vertical", label: "Flip vertically", icon: FlipVertical2 },
];

function turn(degrees: number): string {
  if (degrees === 0) return "level";
  return `${Math.abs(degrees)}° ${degrees > 0 ? "clockwise" : "counterclockwise"}`;
}

/** Turning, straightening and mirroring the picture, made exactly and without a model.
 *
 * A quarter turn or a flip is the whole edit in one press, so there is nothing
 * to describe and no Apply: the result arrives as the next step, where it can
 * be compared with the picture before it, or left behind by picking an earlier
 * step. Straightening is a few degrees chosen on a slider while the canvas
 * shows the turned picture and what it keeps, so it waits for its own press.
 */
export function StudioTransformTool({
  busy,
  degrees,
  onEdit,
  onDegrees,
  onStraighten,
}: {
  /** Another edit is still arriving; pressing now is refused, not queued. */
  busy: boolean;
  /** How far the straightening turns the picture, clockwise when positive. */
  degrees: number;
  onEdit: (operation: StudioLocalEditOperation) => void;
  onDegrees: (degrees: number) => void;
  onStraighten: () => void;
}) {
  const level = degrees === 0;
  return (
    <div className="studio-tool-options">
      <div className="studio-transform-actions" role="group" aria-label="Rotate or flip">
        {OPERATIONS.map(({ operation, label, icon: Icon }) => (
          <button
            key={operation}
            type="button"
            className="secondary compact-button"
            // Not disabled: a focused button that becomes disabled drops focus
            // to the page, and a keyboard user would lose their place mid-edit.
            aria-disabled={busy}
            onClick={() => {
              if (!busy) onEdit(operation);
            }}
          >
            <Icon size={14} aria-hidden="true" /> {label}
          </button>
        ))}
      </div>
      <label className="studio-adjust-slider">
        <span>
          <strong>Straighten</strong> {turn(degrees)}
        </span>
        <input
          type="range"
          min={-STRAIGHTEN_LIMIT}
          max={STRAIGHTEN_LIMIT}
          step={0.1}
          value={degrees}
          aria-label="Straighten"
          onChange={(event) => onDegrees(Number(event.target.value))}
        />
      </label>
      <small>
        {busy
          ? "Applying…"
          : level
            ? "Turns or mirrors the whole picture exactly. No model runs, so nothing needs installing."
            : "The picture shows the turn and the part it keeps, which is the largest box of its own shape."}
      </small>
      <div className="studio-transform-actions">
        <button type="button" className="secondary compact-button" aria-disabled={level}
          onClick={() => {
            if (!level) onDegrees(0);
          }}>
          Level
        </button>
        <button
          type="button"
          className="secondary compact-button"
          aria-disabled={busy || level}
          onClick={() => {
            if (!busy && !level) onStraighten();
          }}
        >
          <RotateCw size={14} aria-hidden="true" /> Straighten
        </button>
      </div>
    </div>
  );
}
