import { SlidersHorizontal } from "lucide-react";
import { ADJUSTMENT_LIMIT, isNeutral } from "./studioAdjustments";
import { currentLook, lookAdjustments, LOOKS } from "./studioLooks";
import type { StudioColorAdjustments } from "./types";

const SLIDERS: Array<{ key: keyof StudioColorAdjustments; label: string; minimum?: number }> = [
  { key: "brightness", label: "Brightness" },
  { key: "contrast", label: "Contrast" },
  { key: "highlights", label: "Highlights" },
  { key: "shadows", label: "Shadows" },
  { key: "whites", label: "Whites" },
  { key: "blacks", label: "Blacks" },
  { key: "saturation", label: "Saturation" },
  { key: "vibrance", label: "Vibrance" },
  { key: "warmth", label: "Warmth" },
  { key: "tint", label: "Tint" },
  { key: "sharpness", label: "Sharpness" },
  { key: "vignette", label: "Vignette" },
  // Grain is added or not; there is no taking it away.
  { key: "grain", label: "Grain", minimum: 0 },
];

function signed(value: number): string {
  return value > 0 ? `+${value}` : String(value);
}

/** Light and color: thirteen sliders shown on the picture as they move.
 *
 * The canvas draws the adjusted picture itself, by the same arithmetic the
 * server uses, so what is on screen is what Apply makes. Nothing is saved
 * until then, and Reset puts every slider back without leaving a step. Auto
 * sets warmth, tint, brightness and contrast from the picture itself, where
 * its colors can be read, and leaves them to be changed like any others. A
 * look sets every slider to a named starting point, and shows as chosen while
 * the sliders still stand exactly there.
 */
export function StudioAdjustTool({
  adjustments,
  busy,
  onChange,
  onReset,
  onApply,
  onAuto,
  onLook,
}: {
  adjustments: StudioColorAdjustments;
  busy: boolean;
  onChange: (key: keyof StudioColorAdjustments, value: number) => void;
  onReset: () => void;
  onApply: () => void;
  /** Sets the sliders from the picture; absent where its colors cannot be read. */
  onAuto?: () => void;
  /** Sets every slider to a look. */
  onLook?: (adjustments: StudioColorAdjustments) => void;
}) {
  const unchanged = isNeutral(adjustments);
  const ready = !unchanged && !busy;
  const chosen = currentLook(adjustments);
  return (
    <div className="studio-tool-options">
      {onLook && (
        <div className="segmented compact" role="group" aria-label="Looks" style={{ flexWrap: "wrap" }}>
          {LOOKS.map((look) => (
            <button key={look.name} type="button" aria-pressed={chosen === look.name}
              className={chosen === look.name ? "active" : ""} aria-disabled={busy}
              onClick={() => {
                if (!busy) onLook(lookAdjustments(look));
              }}>
              {look.name}
            </button>
          ))}
        </div>
      )}
      {SLIDERS.map(({ key, label, minimum = -ADJUSTMENT_LIMIT }) => (
        <label key={key} className="studio-adjust-slider">
          <span>
            <strong>{label}</strong> {signed(adjustments[key])}
          </span>
          <input
            type="range"
            min={minimum}
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
        {onAuto && (
          <button type="button" className="secondary compact-button" aria-disabled={busy}
            onClick={() => {
              if (!busy) onAuto();
            }}>
            Auto
          </button>
        )}
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
