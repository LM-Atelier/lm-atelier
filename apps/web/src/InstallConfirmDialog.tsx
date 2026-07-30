import { AccessibleDialog } from "./AccessibleDialog";
import { formatBytes } from "./format";
import type { CatalogPreflight, SystemInfo } from "./types";

interface Props {
  name: string;
  preflight: CatalogPreflight;
  system?: SystemInfo;
  pending: boolean;
  onConfirm: () => void;
  onCancel: () => void;
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
}: Props) {
  const accelerator = acceleratorBytes(system);
  const needed = preflight.estimated_vram_bytes;
  const exceedsAccelerator = needed != null && accelerator != null && needed > accelerator;
  const warnings = preflight.checks.filter((check) => check.status === "warn");

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
          <dd>{formatBytes(preflight.download_bytes)}</dd>
        </div>
        <div>
          <dt>Free space</dt>
          <dd>{formatBytes(preflight.available_disk_bytes)}</dd>
        </div>
        {preflight.estimated_ram_bytes != null && (
          <div>
            <dt>Memory to load</dt>
            <dd>{formatBytes(preflight.estimated_ram_bytes)}</dd>
          </div>
        )}
        {needed != null && (
          <div>
            <dt>Accelerator memory</dt>
            <dd>
              {formatBytes(needed)}
              {accelerator != null && <small> of {formatBytes(accelerator)}</small>}
            </dd>
          </div>
        )}
      </dl>
      {exceedsAccelerator && (
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
        <button className="secondary" onClick={onCancel} disabled={pending}>
          Cancel
        </button>
        <button className="primary" onClick={onConfirm} disabled={pending}>
          {pending ? "Starting…" : `Download ${formatBytes(preflight.download_bytes)}`}
        </button>
      </footer>
    </AccessibleDialog>
  );
}
