import {
  ArrowDown,
  ArrowDownLeft,
  ArrowDownRight,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ArrowUpLeft,
  ArrowUpRight,
  Circle,
} from "lucide-react";
import type { StudioCanvasAnchor } from "./types";

const ANCHORS: Array<{ anchor: StudioCanvasAnchor; label: string; icon: typeof Circle }> = [
  { anchor: "top_left", label: "Top left", icon: ArrowUpLeft },
  { anchor: "top", label: "Top", icon: ArrowUp },
  { anchor: "top_right", label: "Top right", icon: ArrowUpRight },
  { anchor: "left", label: "Left", icon: ArrowLeft },
  { anchor: "center", label: "Center", icon: Circle },
  { anchor: "right", label: "Right", icon: ArrowRight },
  { anchor: "bottom_left", label: "Bottom left", icon: ArrowDownLeft },
  { anchor: "bottom", label: "Bottom", icon: ArrowDown },
  { anchor: "bottom_right", label: "Bottom right", icon: ArrowDownRight },
];

/** Nine places on the picture, as a three by three group of presses.
 *
 * A corner, the middle of an edge, or the center: where a canvas change keeps
 * the picture, and where added words sit.
 */
export function StudioAnchorPicker({
  value,
  label,
  onChange,
}: {
  value: StudioCanvasAnchor;
  /** What the choice is for, read out as the group's name. */
  label: string;
  onChange: (anchor: StudioCanvasAnchor) => void;
}) {
  return (
    <div className="segmented studio-canvas-anchor" role="group" aria-label={label}>
      {ANCHORS.map(({ anchor, label: name, icon: Icon }) => (
        <button
          key={anchor}
          type="button"
          aria-label={name}
          aria-pressed={value === anchor}
          className={value === anchor ? "active" : ""}
          onClick={() => onChange(anchor)}
        >
          <Icon size={14} aria-hidden="true" />
        </button>
      ))}
    </div>
  );
}
