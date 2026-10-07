import { useSyncExternalStore } from "react";
import type { Visibility } from "./settings";

/** How much detail a settings editor shows when it opens.
 *
 * Every editor used to open at Basic, so somebody who always wants the
 * sampler and the seed switched level every time. This is only where an editor
 * starts: switching inside an open editor stays with that editor, because a
 * quick look at the expert settings is not a change of habit.
 */
export const SETTING_DETAIL_KEY = "local-lm-setting-detail";

export const SETTING_DETAILS: readonly Visibility[] = ["basic", "advanced", "expert"];

function isSettingDetail(value: unknown): value is Visibility {
  return typeof value === "string" && (SETTING_DETAILS as readonly string[]).includes(value);
}

/** The level a newly opened editor starts at. Read it when the editor mounts. */
export function storedSettingDetail(): Visibility {
  try {
    const stored = localStorage.getItem(SETTING_DETAIL_KEY);
    return isSettingDetail(stored) ? stored : "basic";
  } catch {
    return "basic";
  }
}

const listeners = new Set<() => void>();

export function setSettingDetail(level: Visibility): void {
  try {
    localStorage.setItem(SETTING_DETAIL_KEY, level);
  } catch {
    return;
  }
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === SETTING_DETAIL_KEY) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** The chosen starting level, re-rendering whoever reads it when it changes. */
export function useSettingDetail(): Visibility {
  return useSyncExternalStore(subscribe, storedSettingDetail, () => "basic");
}
