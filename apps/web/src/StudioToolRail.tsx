import {
  Brush,
  Eraser,
  Lasso,
  Redo2,
  Sparkles,
  Square,
  Type,
  Undo2,
} from "lucide-react";
import type { StudioToolKind } from "./studioToolState";
import type { StudioToolCapability } from "./types";

const TOOLS: Array<{ kind: StudioToolKind; label: string; icon: typeof Brush }> = [
  { kind: "instruct", label: "Instruct the whole image", icon: Type },
  { kind: "brush", label: "Brush a selection", icon: Brush },
  { kind: "eraser", label: "Erase from the selection", icon: Eraser },
  { kind: "rect", label: "Select a rectangle", icon: Square },
  { kind: "lasso", label: "Lasso a selection", icon: Lasso },
  { kind: "enhance", label: "Enlarge and restore detail", icon: Sparkles },
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
  capabilities = [],
}: {
  active: StudioToolKind;
  onSelect: (kind: StudioToolKind) => void;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  disabled?: boolean;
  /** What each tool can do here. Empty until the report arrives, which reads
   * as "nothing known against it" rather than as a rail full of warnings. */
  capabilities?: StudioToolCapability[];
}) {
  const blocked = new Map(
    capabilities.filter((tool) => !tool.available).map((tool) => [tool.kind, tool.reason]),
  );
  return (
    <nav className="studio-tool-rail" aria-label="Editing tools">
      {TOOLS.map(({ kind, label, icon: Icon }) => {
        // Enabled but guided, never greyed out: a disabled button explains
        // nothing, and the thing that would fix it is a install away.
        const unavailable = blocked.get(kind);
        return (
          <button
            key={kind}
            type="button"
            className={`icon-button ${active === kind ? "selected" : ""} ${unavailable ? "unavailable" : ""}`}
            aria-label={unavailable ? `${label} - ${unavailable}` : label}
            aria-pressed={active === kind}
            aria-describedby={unavailable ? "studio-tool-guidance" : undefined}
            title={unavailable ? `${label}
${unavailable}` : label}
            disabled={disabled}
            onClick={() => onSelect(kind)}
          >
            <Icon size={18} aria-hidden="true" />
          </button>
        );
      })}
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
