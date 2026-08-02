import { formatEta, formatStageElapsed, formatTransferRate } from "./format";
import type { Job, ProgressV2 } from "./types";

export function jobProgressFraction(job: Job): number | null {
  const progress = job.progress_json;
  if (progress?.indeterminate) return null;
  const structured = progress?.overall_progress ?? progress?.stage_progress;
  if (job.status === "queued" && (structured === null || structured === undefined)) {
    return null;
  }
  const legacy = Number.isFinite(job.progress) ? job.progress : null;
  const value = structured === null || structured === undefined
    ? legacy
    : legacy === null
      ? structured
      : Math.max(structured, legacy);
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.min(1, Math.max(0, value));
  }
  if (job.status === "queued") return null;
  return Number.isFinite(job.progress) ? Math.min(1, Math.max(0, job.progress)) : null;
}

export function progressSampleIsFresh(progress: ProgressV2): boolean {
  const updatedAt = Date.parse(progress.updated_at);
  return Number.isFinite(updatedAt) && Math.abs(Date.now() - updatedAt) <= 5_000;
}

export function jobProgressText(job: Job): string {
  const progress = job.progress_json;
  const pieces = [progress?.stage || job.phase];
  if (
    ["queued", "running"].includes(job.status)
    && typeof progress?.stage_elapsed_ms === "number"
    && Number.isFinite(progress.stage_elapsed_ms)
  ) {
    const startedAt = progress.stage_started_at ? Date.parse(progress.stage_started_at) : NaN;
    const elapsed = Number.isFinite(startedAt)
      ? Math.max(progress.stage_elapsed_ms, Date.now() - startedAt)
      : progress.stage_elapsed_ms;
    pieces.push(formatStageElapsed(elapsed));
  }
  const fraction = jobProgressFraction(job);
  if (fraction !== null) pieces.push(`${Math.round(fraction * 100)}%`);
  if (job.status === "queued" && typeof progress?.queue_position === "number") {
    pieces.push(progress.queue_position > 0 ? `${progress.queue_position} ahead` : "Next");
  }
  const freshSample = progress ? progressSampleIsFresh(progress) : false;
  if (freshSample && typeof progress?.rate_bytes_per_second === "number") {
    pieces.push(formatTransferRate(progress.rate_bytes_per_second));
  }
  if (freshSample && typeof progress?.eta_seconds === "number") {
    pieces.push(formatEta(progress.eta_seconds));
  }
  return pieces.filter(Boolean).join(" · ");
}
