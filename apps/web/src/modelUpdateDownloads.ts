export interface ModelUpdateDownload {
  jobId: string;
  previousInstallId: string;
  modelName: string;
}

const storageKey = "lm-atelier.model-update-downloads";

export function readModelUpdateDownloads(): ModelUpdateDownload[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(storageKey) ?? "[]");
    if (!Array.isArray(stored)) return [];
    return stored.filter((item): item is ModelUpdateDownload => (
      typeof item === "object" && item !== null
      && typeof item.jobId === "string" && item.jobId.length > 0
      && typeof item.previousInstallId === "string" && item.previousInstallId.length > 0
      && typeof item.modelName === "string"
    ));
  } catch {
    return [];
  }
}

export function saveModelUpdateDownloads(downloads: ModelUpdateDownload[]): void {
  try {
    localStorage.setItem(storageKey, JSON.stringify(downloads));
  } catch {
    // The current view still holds the offer when browser storage is unavailable.
  }
}
