export function StatusDot({ healthy, label }: { healthy: boolean; label?: string }) {
  return (
    <span
      className={`status-dot ${healthy ? "healthy" : "offline"}`}
      role={label ? "img" : undefined}
      aria-label={label ? `${label}: ${healthy ? "ready" : "unavailable"}` : undefined}
      aria-hidden={label ? undefined : true}
    />
  );
}
