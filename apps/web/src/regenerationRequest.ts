import { api } from "./api";

const REQUEST_PREFIX = "lm-atelier:regeneration-request:v1:";
const REQUEST_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
      .map(([key, item]) => [key, canonicalValue(item)]));
  }
  return value;
}

/** Keep an unacknowledged regeneration's identity across retries and reloads. */
export async function regenerateWithRetry(
  chatId: string,
  messageId: string,
  settings: Record<string, unknown>,
) {
  // Snapshot the wire value before hashing awaits; controls can change meanwhile.
  const supplied = JSON.parse(JSON.stringify(settings)) as Record<string, unknown>;
  const material = JSON.stringify([chatId, messageId, canonicalValue(supplied)]);
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(material));
  const storageKey = REQUEST_PREFIX + Array.from(new Uint8Array(digest),
    (byte) => byte.toString(16).padStart(2, "0")).join("");
  let requestId: string;
  try {
    const previous = localStorage.getItem(storageKey);
    if (previous !== null && !REQUEST_ID.test(previous)) {
      throw new Error("Invalid saved regeneration request.");
    }
    requestId = previous ?? crypto.randomUUID();
    localStorage.setItem(storageKey, requestId);
  } catch {
    throw new Error("Could not save the regeneration request. Check browser storage and try again.");
  }
  const accepted = await api.regenerateMessage(messageId, supplied, requestId);
  try {
    if (localStorage.getItem(storageKey) === requestId) localStorage.removeItem(storageKey);
  } catch {
    // A retained key safely replays this acceptance until storage is writable.
  }
  return accepted;
}
