import { FlipHorizontal2, FlipVertical2, RotateCcw, RotateCw } from "lucide-react";
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

/** Turning and mirroring the picture, made exactly and without a model.
 *
 * Each press is the whole edit, so there is nothing to describe and no Apply:
 * the result arrives as the next step, where it can be compared with the
 * picture before it, or left behind by picking an earlier step.
 */
export function StudioTransformTool({
  busy,
  onEdit,
}: {
  /** Another edit is still arriving; pressing now is refused, not queued. */
  busy: boolean;
  onEdit: (operation: StudioLocalEditOperation) => void;
}) {
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
      <small>
        {busy
          ? "Applying…"
          : "Turns or mirrors the whole picture exactly. No model runs, so nothing needs installing."}
      </small>
    </div>
  );
}
