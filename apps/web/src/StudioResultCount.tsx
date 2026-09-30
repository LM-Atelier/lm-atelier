import { useId } from "react";
import { studioOffersResults } from "./studioApplyPlan";

const COUNTS = [1, 2, 3, 4] as const;

/** How many results the next Apply asks for.
 *
 * Each result comes back as its own step, made from the same picture as the
 * others, so they can be compared and the best one taken further. The caller
 * shows it only for tools that run a model.
 */
export function StudioResultCount({
  kind,
  value,
  onChange,
}: {
  kind: string;
  value: number;
  onChange: (value: number) => void;
}) {
  const label = useId();
  if (!studioOffersResults(kind)) return null;
  return (
    <div className="studio-tool-options">
      <span id={label} className="muted">Results</span>
      <div className="segmented" role="group" aria-labelledby={label}>
        {COUNTS.map((count) => (
          <button
            key={count}
            type="button"
            aria-pressed={value === count}
            className={value === count ? "active" : ""}
            onClick={() => onChange(count)}
          >
            {count}
          </button>
        ))}
      </div>
    </div>
  );
}
