function seconds(value: number): string {
  return String(Number(value.toFixed(4)));
}

export function videoLengthSummary(
  provenance: Record<string, unknown> | undefined,
): string | null {
  const raw = provenance?.video_length;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const length = raw as Record<string, unknown>;
  const requested = Number(length.requested_seconds ?? Number.NaN);
  const delivered = Number(length.delivered_seconds ?? Number.NaN);
  if (!Number.isFinite(requested) || !Number.isFinite(delivered)) return null;
  return Math.abs(requested - delivered) > 1e-9
    ? `Requested ${seconds(requested)}s · delivered ${seconds(delivered)}s`
    : `Delivered ${seconds(delivered)}s`;
}
