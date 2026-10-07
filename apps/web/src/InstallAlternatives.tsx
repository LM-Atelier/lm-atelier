import { formatBytes } from "./format";
import type { CatalogHardwareAlternative, HardwareFitStatus } from "./types";

const labels: Record<HardwareFitStatus, string> = {
  recommended: "Recommended fit",
  likely: "Likely to fit",
  tight: "Memory may be tight",
  unsupported: "Hardware not supported",
  unknown: "Hardware fit unknown",
};

export function InstallAlternatives({ alternatives, pending, onSelect }: {
  alternatives: CatalogHardwareAlternative[];
  pending: boolean;
  onSelect: (files: string[]) => void;
}) {
  if (!alternatives.length) return null;
  return (
    <section aria-label="Other model variants">
      <h3>Other model variants</h3>
      <p>Ranked by total memory capacity. These are calculated estimates, not tested results. Review an option to check its download and compatibility.</p>
      <ul>{alternatives.map((option) => (
        <li key={option.selected_files.join("\n")}>
          <strong>{option.selected_files[0]}</strong>
          <p>{labels[option.hardware_fit.status]} · {option.download_size_complete ? formatBytes(option.download_bytes) : "Download size incomplete"} · {option.selected_files.length} file{option.selected_files.length === 1 ? "" : "s"}</p>
          <button className="secondary compact-button" aria-disabled={pending}
            aria-label={`Review ${option.selected_files[0]}`} onClick={() => {
              if (!pending) onSelect(option.selected_files);
            }}>Review option</button>
        </li>
      ))}</ul>
    </section>
  );
}
