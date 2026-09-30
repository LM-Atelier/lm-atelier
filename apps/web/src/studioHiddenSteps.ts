import { useMemo, useSyncExternalStore } from "react";

/** Results put out of the Studio's strip, remembered for each session in this browser.
 *
 * Asking for several results, or trying another again and again, fills the
 * strip with alternatives, most of them looked at and set aside. Hiding one
 * takes it out of the strip and nothing more: the picture stays in the library
 * like every result, and showing the hidden ones brings it back. It is a choice
 * about this browser's view of a session, so it is remembered here, for the
 * most recent sessions only.
 */

export const HIDDEN_STEPS_KEY = "local-lm-studio-hidden";

/** How many sessions keep what they hid; the one used longest ago is forgotten first. */
const KEPT_SESSIONS = 20;

type HiddenInSession = { session: string; hidden: string[] };

function isHiddenInSession(value: unknown): value is HiddenInSession {
  if (typeof value !== "object" || value === null) return false;
  const { session, hidden } = value as { session?: unknown; hidden?: unknown };
  return typeof session === "string" && Array.isArray(hidden) && hidden.every((id) => typeof id === "string");
}

function stored(): HiddenInSession[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(HIDDEN_STEPS_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter(isHiddenInSession) : [];
  } catch {
    // Unreadable or unavailable storage hides nothing: the strip shows every result, as it always did.
    return [];
  }
}

// Read once and kept, so what is hidden in this visit holds even where storage cannot keep it.
let entries: HiddenInSession[] | null = null;
const listeners = new Set<() => void>();

function current(): HiddenInSession[] {
  entries ??= stored();
  return entries;
}

function remember(next: HiddenInSession[]): void {
  entries = next.slice(0, KEPT_SESSIONS);
  try {
    localStorage.setItem(HIDDEN_STEPS_KEY, JSON.stringify(entries));
  } catch {
    // Kept for this visit only.
  }
  for (const listener of listeners) listener();
}

/** Take a result out of this session's strip. */
export function hideStudioStep(session: string, artifactId: string): void {
  const all = current();
  const already = all.find((entry) => entry.session === session)?.hidden ?? [];
  remember([
    { session, hidden: [...new Set([...already, artifactId])] },
    ...all.filter((entry) => entry.session !== session),
  ]);
}

/** Put every result this session hid back in its strip. */
export function showStudioSteps(session: string): void {
  remember(current().filter((entry) => entry.session !== session));
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** The pictures hidden from this session's strip. */
export function useStudioHiddenSteps(session: string | null): ReadonlySet<string> {
  const all = useSyncExternalStore(subscribe, current);
  return useMemo(() => new Set(all.find((entry) => entry.session === session)?.hidden ?? []), [all, session]);
}

/** Forget what this module has read, so a test starts from storage alone. */
export function forgetHiddenStepsForTest(): void {
  entries = null;
}
