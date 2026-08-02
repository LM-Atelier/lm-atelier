import { formatBytes, formatEta, formatTransferRate } from "./format";
import { progressSampleIsFresh } from "./jobProgress";
import type { RuntimeStatus } from "./types";

export function RuntimeSetupCard({
  runtime,
  installPending,
  onInstall,
}: {
  runtime: RuntimeStatus;
  installPending: boolean;
  onInstall: (engine: RuntimeStatus["engine"]) => void;
}) {
  const structured = runtime.progress_json;
  const exactProgress = structured?.indeterminate
    ? null
    : structured?.overall_progress ?? structured?.stage_progress ?? runtime.progress;
  const stateLabel = runtime.state === "ready"
    ? "Ready"
    : runtime.state === "installing"
      ? exactProgress === null
        ? structured?.stage || "Working"
        : `${Math.round(exactProgress * 100)}%`
      : runtime.state === "unsupported"
        ? "Manual setup required"
        : runtime.state === "failed"
          ? "Setup failed"
          : "Not installed";
  const detail = runtime.security_status === "blocked"
    ? `${runtime.license} · ${runtime.security_message || runtime.message}`
    : runtime.engine === "comfyui"
      ? `${runtime.license} · downloaded separately`
      : runtime.message;
  const transfer = structured?.rate_bytes_per_second && progressSampleIsFresh(structured)
    ? [
        formatTransferRate(structured.rate_bytes_per_second),
        typeof structured.eta_seconds === "number"
          ? formatEta(structured.eta_seconds)
          : null,
      ].filter(Boolean).join(" · ")
    : null;
  return (
    <article className="runtime-setup">
      <div>
        <strong>{runtime.engine}</strong>
        <span>{runtime.release} · {stateLabel}</span>
      </div>
      {runtime.state === "installing" && (
        <progress
          value={exactProgress ?? undefined}
          max={1}
          aria-label={`${runtime.engine} setup progress`}
        />
      )}
      {runtime.state !== "installing" && runtime.state !== "ready" && runtime.supported && (
        <button
          className="secondary compact-button"
          disabled={installPending}
          onClick={() => onInstall(runtime.engine)}
        >
          {runtime.state === "failed"
            ? "Retry"
            : `Install${runtime.size_bytes ? ` · ${formatBytes(runtime.size_bytes)}` : ""}`}
        </button>
      )}
      <small>{detail}</small>
      {transfer && <small>{transfer}</small>}
    </article>
  );
}
