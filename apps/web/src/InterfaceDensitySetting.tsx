import { useId } from "react";
import { DENSITIES, setInterfaceDensity, useInterfaceDensity } from "./densityPreference";

const LABELS = { compact: "Compact", standard: "Standard", comfortable: "Comfortable" };

export function InterfaceDensitySetting() {
  const id = useId();
  const density = useInterfaceDensity();
  return (
    <div className="setting-row appearance-row density-row">
      <span>
        <strong id={`${id}-density`}>Interface density</strong>
        <small id={`${id}-density-help`}>
          Compact fits more controls on screen. Comfortable gives them more space. Text size stays the same.
        </small>
      </span>
      <div className="segmented" role="group" aria-labelledby={`${id}-density`} aria-describedby={`${id}-density-help`}>
        {DENSITIES.map(value => (
          <button type="button" key={value} className={density === value ? "active" : ""}
            aria-pressed={density === value} onClick={() => setInterfaceDensity(value)}>
            {LABELS[value]}
          </button>
        ))}
      </div>
    </div>
  );
}
