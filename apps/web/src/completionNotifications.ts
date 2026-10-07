import { useSyncExternalStore } from "react";
import { playFinishedSound, soundWhenFinished } from "./completionSound";
import type { AppEvent } from "./types";

/** Telling somebody their work finished while they were looking at something else.
 *
 * Off until asked for, because a notification is an interruption and the
 * browser has to be given permission for it. Only shown while the workspace is
 * hidden: somebody looking at the page already sees the result arrive.
 *
 * The text is deliberately generic. A notification can appear on a lock screen
 * or over another application, so it never carries what the message or the
 * request said.
 */
export const NOTIFY_WHEN_FINISHED_KEY = "local-lm-notify-when-finished";

/** What turning notifications on led to. */
export type NotificationOutcome = "on" | "off" | "denied" | "unsupported";

const FINISHED_TEXT: Record<string, string> = {
  "run.completed": "A result is ready.",
  "run.failed": "A request did not finish.",
};

function storedEnabled(): boolean {
  try {
    return localStorage.getItem(NOTIFY_WHEN_FINISHED_KEY) === "on";
  } catch {
    return false;
  }
}

const listeners = new Set<() => void>();

function remember(on: boolean): void {
  try {
    localStorage.setItem(NOTIFY_WHEN_FINISHED_KEY, on ? "on" : "off");
  } catch {
    return;
  }
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === NOTIFY_WHEN_FINISHED_KEY) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** Whether finished work is announced, re-rendering whoever reads it when that changes. */
export function useNotifyWhenFinished(): boolean {
  return useSyncExternalStore(subscribe, storedEnabled, () => false);
}

/** Turn announcements on or off, asking the browser for permission when turning them on.
 *
 * The choice is only remembered as on once the browser has agreed, so the
 * setting never claims to notify when it cannot.
 */
export async function setNotifyWhenFinished(on: boolean): Promise<NotificationOutcome> {
  if (!on) {
    remember(false);
    return "off";
  }
  if (typeof Notification === "undefined") return "unsupported";
  const permission =
    Notification.permission === "granted" ? "granted" : await Notification.requestPermission();
  if (permission !== "granted") {
    remember(false);
    return "denied";
  }
  remember(true);
  return "on";
}

/** How many announced runs are remembered, so a replayed event is not announced twice. */
const REMEMBERED_RUNS = 200;

const announced: string[] = [];

/** Announce a finished run, if somebody asked for that and is not looking at the page.
 *
 * Called for every run event the live connection delivers, including
 * `run.cancelled`, which is deliberately never announced: cancelling was
 * somebody's own action. A run is announced once, keyed by the run the event
 * names, however many times the event arrives - a reconnect replays events.
 * The chime, when chosen, follows the same rules, and plays before the
 * notification so a browser that refuses the one still makes the other.
 *
 * It never throws. It runs inside the handler that keeps the workspace's data
 * current, and a notification that fails - an unsupported browser, a blocked
 * permission, storage that cannot be read - must not cost that update.
 */
export function notifyRunFinished(event: AppEvent): void {
  try {
    const body = FINISHED_TEXT[event.type];
    if (!body || event.entity_id === null || !document.hidden) return;
    const notify =
      storedEnabled() && typeof Notification !== "undefined" && Notification.permission === "granted";
    const chime = soundWhenFinished();
    if (!notify && !chime) return;
    if (announced.includes(event.entity_id)) return;
    announced.push(event.entity_id);
    if (announced.length > REMEMBERED_RUNS) announced.shift();
    if (chime) playFinishedSound(event.type === "run.failed" ? "failed" : "completed");
    if (notify) new Notification("LM Atelier", { body, tag: `run:${event.entity_id}` });
  } catch {
    // Some browsers only allow notifications from a service worker. Missing one
    // announcement is better than breaking the live update that carried it.
  }
}
