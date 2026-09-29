import {
  Brush,
  Captions,
  Crop,
  Droplet,
  Frame,
  LetterText,
  Maximize2,
  Mountain,
  Paintbrush,
  PersonStanding,
  Redo2,
  RotateCw,
  Scaling,
  Scan,
  Scissors,
  SlidersHorizontal,
  Sparkles,
  SquareDashedMousePointer,
  SunMedium,
  Type,
  Undo2,
} from "lucide-react";
import { isSelectionKind, type SelectionKind, type StudioToolKind } from "./studioToolState";
import type { StudioToolCapability } from "./types";

/** A tool on the rail, or Select, which stands for every way of drawing a selection. */
type RailTool = { kind: StudioToolKind | "select"; label: string; icon: typeof Brush };

/** The tools in five runs, a divider between each: exact edits made without a
 * model; selecting a part of the picture; edits said in words; the subject and
 * the scene around it; and finishing.
 *
 * Select is one tool. The brush, the eraser, a rectangle, a lasso, a fill and
 * similar colors are six ways of drawing one selection, and the panel offers
 * them once Select is chosen; as six buttons here they read as six more things
 * the Studio could do to a picture.
 */
const TOOL_GROUPS: RailTool[][] = [
  [
    { kind: "crop", label: "Crop the picture", icon: Crop },
    { kind: "transform", label: "Rotate, straighten or flip", icon: RotateCw },
    { kind: "perspective", label: "Correct the perspective", icon: Scan },
    { kind: "resize", label: "Resize the picture", icon: Scaling },
    { kind: "canvas", label: "Change the canvas size", icon: Frame },
    { kind: "adjust", label: "Adjust light and color", icon: SlidersHorizontal },
    { kind: "blur", label: "Blur or pixelate part of the picture", icon: Droplet },
    { kind: "paint", label: "Paint over part of the picture", icon: Paintbrush },
    { kind: "caption", label: "Add text to the picture", icon: Captions },
  ],
  [{ kind: "select", label: "Select part of the picture", icon: SquareDashedMousePointer }],
  [
    { kind: "instruct", label: "Instruct the whole image", icon: Type },
    { kind: "text", label: "Replace words in the picture", icon: LetterText },
  ],
  [
    { kind: "isolate", label: "Cut the subject out", icon: Scissors },
    { kind: "background", label: "Replace the background", icon: Mountain },
    { kind: "subject", label: "Replace the subject", icon: PersonStanding },
    { kind: "relight", label: "Relight from a direction", icon: SunMedium },
    { kind: "extend", label: "Extend past the edge", icon: Maximize2 },
  ],
  [{ kind: "enhance", label: "Enlarge and restore detail", icon: Sparkles }],
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
  selectionKind = "brush",
  onSelect,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  disabled = false,
  capabilities = [],
}: {
  active: StudioToolKind;
  /** The way of selecting that choosing Select returns to: the one last used. */
  selectionKind?: SelectionKind;
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
      {TOOL_GROUPS.map((group, index) => [
        index > 0 && <span key={`divider-${index}`} className="studio-rail-divider" aria-hidden="true" />,
        ...group.map(({ kind: railKind, label, icon: Icon }) => {
          // Select stands for whichever way of selecting is in hand, or was last.
          const kind = railKind === "select" ? (isSelectionKind(active) ? active : selectionKind) : railKind;
          const chosen = railKind === "select" ? isSelectionKind(active) : active === kind;
          // Enabled but guided, never greyed out: a disabled button explains
          // nothing, and the thing that would fix it is a install away.
          const unavailable = blocked.get(kind);
          return (
            <button
              key={railKind}
              type="button"
              className={`icon-button ${chosen ? "selected" : ""} ${unavailable ? "unavailable" : ""}`}
              aria-label={unavailable ? `${label} - ${unavailable}` : label}
              aria-pressed={chosen}
              // Only the active tool's reason is on screen, so every other
              // unavailable button pointed at an element that was not there -
              // and when it was there, it explained a different tool.
              aria-describedby={
                unavailable && chosen ? "studio-tool-guidance" : undefined
              }
              title={unavailable ? `${label}
${unavailable}` : label}
              disabled={disabled}
              onClick={() => onSelect(kind)}
            >
              <Icon size={18} aria-hidden="true" />
            </button>
          );
        }),
      ])}
      <span className="studio-rail-divider" aria-hidden="true" />
      <button
        type="button"
        className="icon-button"
        aria-label="Undo the selection change"
        title="Undo"
        aria-disabled={disabled || !canUndo}
        onClick={() => {
          if (!disabled && canUndo) onUndo();
        }}
      >
        <Undo2 size={18} aria-hidden="true" />
      </button>
      <button
        type="button"
        className="icon-button"
        aria-label="Redo the selection change"
        title="Redo"
        aria-disabled={disabled || !canRedo}
        onClick={() => {
          if (!disabled && canRedo) onRedo();
        }}
      >
        <Redo2 size={18} aria-hidden="true" />
      </button>
    </nav>
  );
}
