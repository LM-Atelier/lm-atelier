import { useState } from "react";
import {
  chosenFactor,
  describeFixedEnlargement,
  enlargementChoices,
  enlargementFactor,
  enlargementIdentity,
  enlargementScale,
  type EnlargementScale,
  type StudioUpscaleChoice,
} from "./studioEnlargement";
import type { StudioEnlargement } from "./useStudioEnlargement";

/** The scale Enhance runs its workflow at, as that workflow allows.
 *
 * A workflow that applies a chosen factor offers exactly the factors it takes:
 * its own list, or its own number with the bounds and step it declares, a
 * slider only where both bounds are declared. It starts at the workflow's own
 * default until the person chooses. The factor is named as the workflow's
 * scale, not the picture's size, because another step of the workflow may
 * enlarge it again. One that always enlarges by the same amount offers no
 * choice, and the panel says how much when the workflow's graph shows it; so
 * does one whose bounds hold no factor it would take. Nothing here offers a
 * factor the workflow would not take.
 */
export function StudioEnhanceTool({
  enlargement,
  choice,
  onChoose,
}: {
  enlargement: StudioEnlargement;
  choice: StudioUpscaleChoice | null;
  onChoose: (choice: StudioUpscaleChoice) => void;
}) {
  const { preview, checking, error } = enlargement;
  if (error) return <small role="alert">{error}</small>;
  if (!preview) return <small role="status">{checking ? "Checking what the workflow can do…" : ""}</small>;
  const field = preview.factor;
  if (!field || !field.available) return <small role="status">{describeFixedEnlargement(preview)}</small>;
  const factor = enlargementFactor(field, chosenFactor(preview, choice));
  const choose = (value: number) => onChoose({ preview: enlargementIdentity(preview), factor: value });
  const choices = enlargementChoices(field);
  if (choices) {
    return (
      <div className="segmented" role="group" aria-label="Workflow scale">
        {choices.map((option) => (
          <button key={option} type="button" aria-pressed={option === factor}
            className={option === factor ? "active" : ""} onClick={() => choose(option)}>
            {`${option}x`}
          </button>
        ))}
      </div>
    );
  }
  const scale = enlargementScale(field);
  if (!scale) return <small role="status">{describeFixedEnlargement(preview)}</small>;
  return <EnlargementNumber key={enlargementIdentity(preview)} scale={scale} factor={factor} onChoose={choose} />;
}

/** A factor as a number: a slider where both bounds are declared, a number box otherwise.
 *
 * The box keeps what is being typed until it is a number, so "2." on the way
 * to "2.5" is not snapped back to 2 under the typist's hands.
 */
function EnlargementNumber({
  scale,
  factor,
  onChoose,
}: {
  scale: EnlargementScale;
  factor: number | null;
  onChoose: (value: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const bounded = scale.minimum !== null && scale.maximum !== null;
  return (
    <label>
      <span>
        <strong>Workflow scale</strong> {factor !== null ? `${factor}x` : "the workflow's own"}
      </span>
      <input
        type={bounded ? "range" : "number"}
        aria-label="Workflow scale"
        {...(scale.minimum !== null ? { min: scale.minimum } : {})}
        {...(scale.maximum !== null ? { max: scale.maximum } : {})}
        step={scale.step}
        value={bounded ? factor ?? scale.minimum ?? "" : draft ?? String(factor ?? "")}
        onChange={(event) => {
          const text = event.target.value;
          if (!bounded) setDraft(text);
          const value = Number(text);
          if (text.trim() !== "" && Number.isFinite(value)) onChoose(value);
        }}
        onBlur={() => setDraft(null)}
      />
    </label>
  );
}
