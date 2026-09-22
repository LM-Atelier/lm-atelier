import { AccessibleDialog } from "./AccessibleDialog";
import { InstallHardwareFit } from "./InstallHardwareFit";
import { InstallAlternatives } from "./InstallAlternatives";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import type { CatalogPreflight, SystemInfo } from "./types";

interface Props {
  name: string;
  preflight: CatalogPreflight;
  system?: SystemInfo;
  pending: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  onSelectAlternative?: (files: string[]) => void;
  selectingAlternative?: boolean;
  alternativeError?: string;
}

/** The largest accelerator this machine reports, if it reports one. */
function acceleratorBytes(system?: SystemInfo): number | null {
  const sizes = (system?.devices ?? [])
    .filter((device) => device.kind !== "cpu")
    .map((device) => device.total_memory_bytes)
    .filter((value): value is number => typeof value === "number");
  return sizes.length ? Math.max(...sizes) : null;
}

/**
 * Confirms a model transfer before it starts.
 *
 * These downloads run to tens of gigabytes, and the checks that predict whether
 * the result will actually run were already computed and then discarded, so the
 * install began the moment the button was pressed and the user found out
 * afterwards.
 */
export function InstallConfirmDialog({
  name,
  preflight,
  system,
  pending,
  onConfirm,
  onCancel,
  onSelectAlternative,
  selectingAlternative = false,
  alternativeError,
}: Props) {
  const busy = pending || selectingAlternative;
  const accelerator = acceleratorBytes(system);
  const needed = preflight.estimated_vram_bytes;
  const exceedsAccelerator = needed != null && accelerator != null && needed > accelerator;
  const fit = preflight.hardware_fit;
  const warnings = preflight.checks.filter((check) => check.status === "warn"
    && !(fit && ["memory", "memory-availability"].includes(check.id)));
  let downloadSize = formatBytes(preflight.download_bytes);
  let downloadAction = `Download ${downloadSize}`;
  if (preflight.download_size_complete === false) {
    if (preflight.download_bytes > 0) {
      downloadAction = `Download at least ${downloadSize}`;
      downloadSize = `At least ${downloadSize}`;
    } else {
      downloadSize = "Size unknown";
      downloadAction = "Download";
    }
  }

  return (
    <AccessibleDialog
      title={`Install ${name}?`}
      eyebrow="Confirm download"
      closeLabel="Cancel install"
      onClose={onCancel}
      className="install-confirm"
    >
      <dl className="install-facts">
        <div>
          <dt>Download</dt>
          <dd>{downloadSize}</dd>
        </div>
        <div>
          <dt>Free space</dt>
          <dd>{formatBytes(preflight.available_disk_bytes)}</dd>
        </div>
        {!fit && preflight.estimated_ram_bytes != null && (
          <div>
            <dt>Memory to load</dt>
            <dd>{formatBytes(preflight.estimated_ram_bytes)}</dd>
          </div>
        )}
        {!fit && needed != null && (
          <div>
            <dt>Accelerator memory</dt>
            <dd>
              {formatBytes(needed)}
              {accelerator != null && <small> of {formatBytes(accelerator)}</small>}
            </dd>
          </div>
        )}
      </dl>
      {fit && <InstallHardwareFit fit={fit} />}
      {onSelectAlternative && (preflight.hardware_alternatives?.length ?? 0) > 0 && <>
        <p>Selected files: {preflight.selected_files.join(", ")}</p>
        <InstallAlternatives alternatives={preflight.hardware_alternatives ?? []} pending={busy} onSelect={onSelectAlternative} />
      </>}
      {selectingAlternative && <p role="status">Checking this variant…</p>}
      {alternativeError && <ErrorCallout message={alternativeError} />}
      {!fit && exceedsAccelerator && (
        <p className="install-warning" role="status">
          This model needs more accelerator memory than this machine reports. It
          may run slowly on the processor instead, or fail to load.
        </p>
      )}
      {warnings.map((check) => (
        <p className="install-warning" key={check.id} role="status">
          {check.detail}
        </p>
      ))}
      <footer>
        <button className="secondary" aria-disabled={busy} onClick={() => {
          if (!busy) onCancel();
        }}>
          Cancel
        </button>
        <button className="primary" aria-disabled={busy} onClick={() => {
          if (!busy) onConfirm();
        }}>
          {selectingAlternative ? "Checking…" : pending ? "Starting…" : downloadAction}
        </button>
      </footer>
    </AccessibleDialog>
  );
}
