import type { ChatActivityReference } from "./types";

export const CHAT_ACTIVITY_SEEN_KEY = "lm-atelier:chat-activity-seen:v1";
const MAX_ENTRIES = 2_000;
const MAX_AGE = 90 * 24 * 60 * 60 * 1_000;
type Entry = [chatId: string, eventId: string, sequence: number, seenAt: number];
type StorageAccess = Pick<Storage, "getItem" | "setItem">;

function identity(value: unknown): value is string {
  return typeof value === "string" && /^[a-zA-Z0-9_-]{1,80}$/.test(value);
}

function decode(raw: string | null, now: number): Entry[] {
  if (!raw || raw.length > 500_000) return [];
  try {
    const value: unknown = JSON.parse(raw);
    if (!Array.isArray(value) || value.length > MAX_ENTRIES) return [];
    return value.filter((entry): entry is Entry => Array.isArray(entry)
      && entry.length === 4 && identity(entry[0]) && identity(entry[1])
      && Number.isSafeInteger(entry[2]) && entry[2] > 0
      && Number.isSafeInteger(entry[3]) && entry[3] <= now
      && entry[3] > now - MAX_AGE);
  } catch {
    return [];
  }
}

/** Keep exact rendered identities; seeing a later event never acknowledges another. */
export function createChatActivitySeenStore(storage: StorageAccess | null, now = Date.now) {
  const listeners = new Set<() => void>();
  const read = () => {
    try { return decode(storage?.getItem(CHAT_ACTIVITY_SEEN_KEY) ?? null, now()); }
    catch { return []; }
  };
  let entries: Entry[] = read();
  let version = 0;
  const changed = () => { version += 1; for (const listener of listeners) listener(); };
  const hasSeen = (chatId: string, activity: ChatActivityReference | null | undefined) =>
    Boolean(activity && entries.some((entry) => entry[0] === chatId
      && entry[1] === activity.id && entry[2] === activity.sequence));
  return {
    hasSeen,
    snapshot: () => version,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
    refresh: (raw: string | null) => {
      entries = decode(raw, now());
      changed();
    },
    markSeen: (chatId: string, activity: ChatActivityReference) => {
      if (!identity(chatId) || !identity(activity.id)
        || !Number.isSafeInteger(activity.sequence) || activity.sequence < 1) return;
      if (hasSeen(chatId, activity)) return;
      const stamp = now();
      const combined = new Map<string, Entry>();
      for (const entry of [...read(), ...entries]) {
        if (entry[3] > stamp - MAX_AGE) combined.set(`${entry[0]}:${entry[1]}`, entry);
      }
      combined.set(`${chatId}:${activity.id}`, [chatId, activity.id, activity.sequence, stamp]);
      entries = [...combined.values()].sort((a, b) => b[3] - a[3]).slice(0, MAX_ENTRIES);
      try { storage?.setItem(CHAT_ACTIVITY_SEEN_KEY, JSON.stringify(entries)); }
      catch { /* Unavailable storage leaves this session's acknowledgment in memory. */ }
      changed();
    },
  };
}

export type ChatActivitySeenStore = ReturnType<typeof createChatActivitySeenStore>;
