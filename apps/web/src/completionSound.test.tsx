/** A chime for finished work: only when chosen, only while nobody is looking, and never at the cost of an update. */

import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { notifyRunFinished, setNotifyWhenFinished } from "./completionNotifications";
import {
  CHIME_TONES,
  SOUND_WHEN_FINISHED_KEY,
  playFinishedSound,
  setSoundWhenFinished,
  useSoundWhenFinished,
} from "./completionSound";
import { browserWithAudio } from "./completionSound.test-support";
import type { AppEvent } from "./types";

function pageHidden(hidden: boolean) {
  Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden });
}

function event(type: string, entity: string): AppEvent {
  return { sequence: 9, type, entity_id: entity, payload: { job_id: "job-1" }, created_at: "2026-09-01T00:00:00Z" };
}

function Reader() {
  return <output>{useSoundWhenFinished() ? "on" : "off"}</output>;
}

beforeEach(() => {
  localStorage.clear();
  pageHidden(true);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  pageHidden(false);
});

it("is silent until chosen, and choosing it plays the chime once", () => {
  const audio = browserWithAudio();
  render(<Reader />);
  expect(screen.getByRole("status").textContent).toBe("off");

  let accepted = false;
  act(() => {
    accepted = setSoundWhenFinished(true);
  });

  expect(accepted).toBe(true);
  expect(localStorage.getItem(SOUND_WHEN_FINISHED_KEY)).toBe("on");
  expect(screen.getByRole("status").textContent).toBe("on");
  expect(audio.tones).toEqual([...CHIME_TONES.completed]);
});

it("stays off in a browser that cannot play audio", () => {
  vi.stubGlobal("AudioContext", undefined);

  expect(setSoundWhenFinished(true)).toBe(false);
  expect(localStorage.getItem(SOUND_WHEN_FINISHED_KEY)).toBe("off");
});

it("asks a browser that has not started its audio yet to start it", () => {
  const audio = browserWithAudio("suspended");

  playFinishedSound("completed");

  expect(audio.resumed).toBe(1);
});

it("chimes, without notifications, for a finished run rising and a failed one falling", () => {
  const audio = browserWithAudio();
  setSoundWhenFinished(true);
  audio.tones = [];

  notifyRunFinished(event("run.completed", "run-rises"));
  notifyRunFinished(event("run.failed", "run-falls"));
  // A reconnect replays events; the same run chimes once.
  notifyRunFinished(event("run.completed", "run-rises"));

  expect(audio.tones).toEqual([...CHIME_TONES.completed, ...CHIME_TONES.failed]);
  expect(CHIME_TONES.completed[1]).toBeGreaterThan(CHIME_TONES.completed[0]);
  expect(CHIME_TONES.failed[1]).toBeLessThan(CHIME_TONES.failed[0]);
});

it("stays silent while the page is in view, for cancelled runs, and when not chosen", () => {
  const audio = browserWithAudio();
  setSoundWhenFinished(true);
  audio.tones = [];

  pageHidden(false);
  notifyRunFinished(event("run.completed", "run-in-view"));
  pageHidden(true);
  notifyRunFinished(event("run.cancelled", "run-cancelled"));
  setSoundWhenFinished(false);
  notifyRunFinished(event("run.completed", "run-not-chosen"));

  expect(audio.tones).toEqual([]);
});

it("still chimes when the notification cannot be shown, and still notifies when the chime cannot play", async () => {
  const shown: string[] = [];
  let refuseNotification = true;
  class Browser {
    static permission: NotificationPermission = "granted";
    static requestPermission = vi.fn(async () => "granted" as NotificationPermission);
    constructor(_title: string, options?: NotificationOptions) {
      if (refuseNotification) throw new TypeError("Illegal constructor");
      shown.push(options?.tag ?? "");
    }
  }
  vi.stubGlobal("Notification", Browser);
  const audio = browserWithAudio();
  await setNotifyWhenFinished(true);
  setSoundWhenFinished(true);
  audio.tones = [];

  expect(() => notifyRunFinished(event("run.completed", "run-no-notification"))).not.toThrow();
  expect(audio.tones).toEqual([...CHIME_TONES.completed]);

  refuseNotification = false;
  audio.refuse = true;
  expect(() => notifyRunFinished(event("run.completed", "run-no-chime"))).not.toThrow();
  expect(shown).toEqual(["run:run-no-chime"]);
});
