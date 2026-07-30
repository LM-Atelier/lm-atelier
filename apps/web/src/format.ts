/** Pure formatting helpers, kept out of App.tsx so it can keep shrinking. */

export function formatTransferRate(bytesPerSecond: number): string {
  const units = ["B/s", "KB/s", "MB/s", "GB/s"];
  let value = bytesPerSecond;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

export function formatEta(seconds: number): string {
  if (seconds < 60) return `about ${Math.max(0, Math.round(seconds))} sec`;
  return `about ${Math.round(seconds / 60)} min`;
}

export function formatStageElapsed(milliseconds: number): string {
  const seconds = Math.max(0, Math.round(milliseconds / 1_000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

export function formatBytes(value?: number | null): string {
  if (value == null) return "Unknown";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size >= 10 || index === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[index]}`;
}

export function formatDate(value?: string | null): string {
  if (!value) return "Update unknown";
  return `Updated ${new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(new Date(value))}`;
}

export function downloadJson(value: unknown, filename: string): void {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

/**
 * A documentation link pinned to the running build.
 *
 * Linking at the default branch shows a reader instructions that may describe
 * software they are not running. A release tag matches what they installed;
 * anything that does not look like a release falls back to the branch, because a
 * broken link helps nobody.
 */
export function docsLink(version: string, path: string): string {
  const reference = /^\d+\.\d+\.\d+$/.test(version) ? `v${version}` : "main";
  return `https://github.com/ajccarlson/lm-atelier/blob/${reference}/${path}`;
}

/** Help destinations, troubleshooting first because that is why people look. */
export function supportLinks(version: string): [string, string][] {
  return [
    ["Troubleshooting", docsLink(version, "docs/TROUBLESHOOTING.md")],
    ["Getting started", docsLink(version, "docs/GETTING-STARTED.md")],
    ["Issues", "https://github.com/ajccarlson/lm-atelier/issues"],
    ["Security", docsLink(version, "SECURITY.md")],
    ["Support", docsLink(version, "SUPPORT.md")],
    ["Privacy", docsLink(version, "docs/PRIVACY.md")],
  ];
}
