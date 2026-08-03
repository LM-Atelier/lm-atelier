import {
  Brush,
  Eraser,
  Lasso,
  Redo2,
  Square,
  Type,
  Undo2,
} from "lucide-react";
import type { StudioToolKind } from "./studioToolState";

const TOOLS: Array<{ kind: StudioToolKind; label: string; icon: typeof Brush }> = [
  { kind: "instruct", label: "Instruct the whole image", icon: Type },
  { kind: "brush", label: "Brush a selection", icon: Brush },
  { kind: "eraser", label: "Erase from the selection", icon: Eraser },
  { kind: "rect", label: "Select a rectangle", icon: Square },
  { kind: "lasso", label: "Lasso a selection", icon: Lasso },
];

/** The studio's left rail: pick how you point at the image.
 *
 * Direct manipulation leads - every tool here is a gesture, and the
 * instruction box beside the canvas only says what to do with what the
 * gesture selected. Undo and redo sit with the tools because they act on
 * the selection, not on the edit history.
 */
export function StudioToolRail({
  active,
  onSelect,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  disabled = false,
}: {
  active: StudioToolKind;
  onSelect: (kind: StudioToolKind) => void;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  disabled?: boolean;
}) {
  return (
    <nav className="studio-tool-rail" aria-label="Editing tools">
      {TOOLS.map(({ kind, label, icon: Icon }) => (
        <button
          key={kind}
          type="button"
          className={`icon-button ${active === kind ? "selected" : ""}`}
          aria-label={label}
          aria-pressed={active === kind}
          title={label}
          disabled={disabled}
          onClick={() => onSelect(kind)}
        >
          <Icon size={18} aria-hidden="true" />
        </button>
      ))}
      <span className="studio-rail-divider" aria-hidden="true" />
      <button
        type="button"
        className="icon-button"
        aria-label="Undo the selection change"
        title="Undo"
        disabled={disabled || !canUndo}
        onClick={onUndo}
      >
        <Undo2 size={18} aria-hidden="true" />
      </button>
      <button
        type="button"
        className="icon-button"
        aria-label="Redo the selection change"
        title="Redo"
        disabled={disabled || !canRedo}
        onClick={onRedo}
      >
        <Redo2 size={18} aria-hidden="true" />
      </button>
    </nav>
  );
}
