import { useSyncExternalStore } from "react";

/** Whether this browser covers an LM Atelier tab once it is hidden.
 *
 * Switching to another tab or window, or minimizing this one, hides the page.
 * With the cover on, the workspace stays covered when it comes back, until
 * someone chooses Show. Only the tab that was hidden is covered; a tab still in
 * view is never covered because another one was hidden. This keeps the screen
 * private when it is shared or glanced at; it encrypts nothing.
 */
export const HIDDEN_TAB_COVER_KEY = "local-lm-cover-hidden-tab";

function storedHiddenTabCover(): boolean {
  try {
    return localStorage.getItem(HIDDEN_TAB_COVER_KEY) === "on";
  } catch {
    return false;
  }
}

const listeners = new Set<() => void>();

/** Remember the choice in this browser and apply it to every open tab. Returns whether it was kept. */
export function setHiddenTabCover(enabled: boolean): boolean {
  try {
    if (enabled) localStorage.setItem(HIDDEN_TAB_COVER_KEY, "on");
    else localStorage.removeItem(HIDDEN_TAB_COVER_KEY);
  } catch {
    // A choice that cannot be remembered would quietly lapse on the next page,
    // so nothing changes and the caller is told.
    return false;
  }
  for (const listener of listeners) listener();
  return true;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === HIDDEN_TAB_COVER_KEY || event.key === null) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** The current choice, re-rendering whoever reads it when it changes, here or in another tab. */
export function useHiddenTabCover(): boolean {
  return useSyncExternalStore(subscribe, storedHiddenTabCover, () => false);
}
