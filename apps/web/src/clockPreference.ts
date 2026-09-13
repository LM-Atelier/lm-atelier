import { useSyncExternalStore } from "react";

/** How times of day are written.
 *
 * "system" leaves it to the language the computer is set to, which is what
 * every time in the workspace did before there was a choice. The other two
 * write every time on that clock whatever the language says, for somebody whose
 * habit differs from their locale's.
 *
 * Only the clock changes. Dates, the order of day and month, and the time zone
 * still come from the computer, because those are correctness rather than taste.
 */
export type ClockChoice = "system" | "12" | "24";

export const CLOCK_CHOICES: readonly ClockChoice[] = ["system", "12", "24"];

export const CLOCK_KEY = "local-lm-clock";

export function isClockChoice(value: unknown): value is ClockChoice {
  return typeof value === "string" && (CLOCK_CHOICES as readonly string[]).includes(value);
}

function storedClockChoice(): ClockChoice {
  try {
    const stored = localStorage.getItem(CLOCK_KEY);
    return isClockChoice(stored) ? stored : "system";
  } catch {
    return "system";
  }
}

const listeners = new Set<() => void>();

/** Remember a clock choice and repaint every time on screen that reads it. */
export function setClockChoice(choice: ClockChoice): void {
  try {
    localStorage.setItem(CLOCK_KEY, choice);
  } catch {
    // Without storage the choice cannot be remembered, and every reader would
    // go on reading the old one, so nothing repaints.
    return;
  }
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Another window of the workspace changing the choice repaints this one too,
  // so two open windows never write the same time two ways.
  const onStorage = (event: StorageEvent) => {
    if (event.key === CLOCK_KEY) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** The current clock choice, re-rendering whoever reads it when it changes. */
export function useClockChoice(): ClockChoice {
  return useSyncExternalStore(subscribe, storedClockChoice, () => "system");
}

/** What to add to a time format so it follows the choice.
 *
 * Empty for "system", so the language decides exactly as it did before.
 * `hourCycle` rather than `hour12`: `hour12: false` lets some browsers write the
 * first hour after midnight as 24 instead of 00, which is not a 24-hour clock
 * anyone reads. A caller must not also pass `hour12`, which would override this.
 */
export function clockOptions(choice: ClockChoice): Pick<Intl.DateTimeFormatOptions, "hourCycle"> {
  if (choice === "12") return { hourCycle: "h12" };
  if (choice === "24") return { hourCycle: "h23" };
  return {};
}
