import { SlidersHorizontal } from "lucide-react";
import { ADJUSTMENT_LIMIT, isNeutral } from "./studioAdjustments";
import type { StudioColorAdjustments } from "./types";

const SLIDERS: Array<{ key: keyof StudioColorAdjustments; label: string }> = [
  { key: "brightness", label: "Brightness" },
  { key: "contrast", label: "Contrast" },
  { key: "saturation", label: "Saturation" },
  { key: "warmth", label: "Warmth" },
  { key: "tint", label: "Tint" },
  { key: "sharpness", label: "Sharpness" },
];

function signed(value: number): string {
  return value > 0 ? `+${value}` : String(value);
}

/** Light and color: six sliders shown on the picture as they move.
 *
 * The canvas draws the adjusted picture itself, by the same arithmetic the
 * server uses, so what is on screen is what Apply makes. Nothing is saved
 * until then, and Reset puts every slider back without leaving a step.
 */
export function StudioAdjustTool({
  adjustments,
  busy,
  onChange,
  onReset,
  onApply,
}: {
  adjustments: StudioColorAdjustments;
  busy: boolean;
  onChange: (key: keyof StudioColorAdjustments, value: number) => void;
  onReset: () => void;
  onApply: () => void;
}) {
  const unchanged = isNeutral(adjustments);
  const ready = !unchanged && !busy;
  return (
    <div className="studio-tool-options">
      {SLIDERS.map(({ key, label }) => (
        <label key={key} className="studio-adjust-slider">
          <span>
            <strong>{label}</strong> {signed(adjustments[key])}
          </span>
          <input
            type="range"
            min={-ADJUSTMENT_LIMIT}
            max={ADJUSTMENT_LIMIT}
            step={1}
            value={adjustments[key]}
            aria-label={label}
            onChange={(event) => onChange(key, Number(event.target.value))}
          />
        </label>
      ))}
      <small>
        {busy
          ? "Applying…"
          : unchanged
            ? "Move a slider to see the change on the picture."
            : "The picture shows the change. Nothing is kept until you apply it."}
      </small>
      <div className="studio-transform-actions">
        <button type="button" className="secondary compact-button" aria-disabled={unchanged}
          onClick={() => {
            if (!unchanged) onReset();
          }}>
          Reset
        </button>
        <button
          type="button"
          className="secondary compact-button"
          // Not disabled: a focused button that becomes disabled drops focus to
          // the page, and a keyboard user would lose their place.
          aria-disabled={!ready}
          onClick={() => {
            if (ready) onApply();
          }}
        >
          <SlidersHorizontal size={14} aria-hidden="true" /> Apply adjustments
        </button>
      </div>
    </div>
  );
}
