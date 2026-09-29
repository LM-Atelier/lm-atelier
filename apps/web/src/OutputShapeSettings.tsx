import { ArrowDown, ArrowUp } from "lucide-react";
import { RATIO_LABELS } from "./outputRatio";
import {
  DEFAULT_OUTPUT_SHAPES,
  defaultedShape,
  movedShape,
  OUTPUT_SHAPES,
  setOutputShapeChoice,
  toggledShape,
  useOutputShapes,
  type OutputShapeMode,
} from "./outputShapePreferences";

const MODES: Array<{ mode: OutputShapeMode; label: string }> = [
  { mode: "image", label: "Pictures" },
  { mode: "video", label: "Videos" },
];

/** Which output shapes the composer offers, in what order, and which new work takes, as a Models & generation setting. */
export function OutputShapeSettings() {
  const choices = useOutputShapes();
  return (
    <section>
      <div className="detail-title">
        <div>
          <h2>Output shapes</h2>
          <p>
            Which shapes the composer offers for pictures and videos, in what order, and which shape new ones take
            when no chat, preset or profile sets a size. A workflow still offers only the shapes it can make, and
            keeps its own size when it cannot make the default exactly. Saved in this browser.
          </p>
        </div>
      </div>
      {MODES.map(({ mode, label }) => {
        const choice = choices[mode];
        const unchanged = JSON.stringify(choice) === JSON.stringify(DEFAULT_OUTPUT_SHAPES[mode]);
        return (
          <div key={mode} role="group" aria-label={`${label} shapes`}>
            <div className="setting-row">
              <span><strong>{label}</strong></span>
              <button type="button" className="secondary compact-button" aria-disabled={unchanged}
                onClick={() => {
                  if (!unchanged) setOutputShapeChoice(mode, DEFAULT_OUTPUT_SHAPES[mode]);
                }}>
                Reset {label.toLowerCase()}
              </button>
            </div>
            <div className="setting-row">
              <span>Default shape</span>
              <select aria-label={`Default shape for ${label.toLowerCase()}`} value={choice.default ?? ""}
                onChange={(event) => setOutputShapeChoice(
                  mode,
                  defaultedShape(choice, OUTPUT_SHAPES.find((shape) => shape === event.target.value) ?? null),
                )}>
                <option value="">The workflow decides</option>
                {choice.order.filter((shape) => !choice.hidden.includes(shape)).map((shape) => (
                  <option key={shape} value={shape}>{`${shape} ${RATIO_LABELS[shape]}`}</option>
                ))}
              </select>
            </div>
            {choice.order.map((shape, index) => {
              const name = `${shape} ${RATIO_LABELS[shape]}`;
              return (
                <div key={shape} className="setting-row">
                  <label>
                    <input type="checkbox" checked={!choice.hidden.includes(shape)}
                      onChange={(event) => setOutputShapeChoice(mode, toggledShape(choice, shape, event.target.checked))} />
                    <span>{name}</span>
                  </label>
                  <span className="row-actions">
                    <button type="button" className="icon-button" aria-label={`Move ${name} earlier for ${label.toLowerCase()}`}
                      aria-disabled={index === 0}
                      onClick={() => {
                        if (index > 0) setOutputShapeChoice(mode, movedShape(choice, shape, -1));
                      }}>
                      <ArrowUp size={14} aria-hidden="true" />
                    </button>
                    <button type="button" className="icon-button" aria-label={`Move ${name} later for ${label.toLowerCase()}`}
                      aria-disabled={index === choice.order.length - 1}
                      onClick={() => {
                        if (index < choice.order.length - 1) setOutputShapeChoice(mode, movedShape(choice, shape, 1));
                      }}>
                      <ArrowDown size={14} aria-hidden="true" />
                    </button>
                  </span>
                </div>
              );
            })}
          </div>
        );
      })}
    </section>
  );
}
