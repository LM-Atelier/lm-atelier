import { useSyncExternalStore } from "react";

/** How pictures and videos appear on this screen until someone chooses to show one.
 *
 * "show" is how they have always appeared. "blur" keeps their place and shape
 * but blurs them past recognition, and "hide" does not load them at all. Either
 * way a single item can be shown, and it is covered again once it leaves the
 * screen. This guards what is on screen against someone looking over a
 * shoulder or a shared screen; it encrypts nothing.
 */
export type SensitiveMediaChoice = "show" | "blur" | "hide";

export const SENSITIVE_MEDIA_CHOICES: readonly SensitiveMediaChoice[] = ["show", "blur", "hide"];

export const SENSITIVE_MEDIA_KEY = "local-lm-sensitive-media";

export function isSensitiveMediaChoice(value: unknown): value is SensitiveMediaChoice {
  return typeof value === "string" && (SENSITIVE_MEDIA_CHOICES as readonly string[]).includes(value);
}

function storedSensitiveMediaChoice(): SensitiveMediaChoice {
  try {
    const stored = localStorage.getItem(SENSITIVE_MEDIA_KEY);
    return isSensitiveMediaChoice(stored) ? stored : "show";
  } catch {
    return "show";
  }
}

const listeners = new Set<() => void>();

/** Remember the choice in this browser and apply it everywhere at once. Returns whether it was kept. */
export function setSensitiveMediaChoice(choice: SensitiveMediaChoice): boolean {
  try {
    localStorage.setItem(SENSITIVE_MEDIA_KEY, choice);
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
    if (event.key === SENSITIVE_MEDIA_KEY) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** The current choice, re-rendering whoever reads it when it changes, here or in another tab. */
export function useSensitiveMediaChoice(): SensitiveMediaChoice {
  return useSyncExternalStore(subscribe, storedSensitiveMediaChoice, () => "show");
}
