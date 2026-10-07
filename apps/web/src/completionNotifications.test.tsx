/** Announcing finished work: only when asked, only when nobody is looking, and never what was said. */

import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  NOTIFY_WHEN_FINISHED_KEY,
  notifyRunFinished,
  setNotifyWhenFinished,
  useNotifyWhenFinished,
} from "./completionNotifications";
import type { AppEvent } from "./types";

const shown: { title: string; options?: NotificationOptions }[] = [];

/** A browser whose notification permission the test decides. */
function browserPermission(initial: NotificationPermission, answer: NotificationPermission = initial) {
  class FakeNotification {
    static permission: NotificationPermission = initial;
    static requestPermission = vi.fn(async () => {
      FakeNotification.permission = answer;
      return answer;
    });
    constructor(title: string, options?: NotificationOptions) {
      shown.push({ title, options });
    }
  }
  vi.stubGlobal("Notification", FakeNotification);
  return FakeNotification;
}

function pageHidden(hidden: boolean) {
  Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden });
}

function event(type: string, entity: string | null = "run-1"): AppEvent {
  return { sequence: 7, type, entity_id: entity, payload: { job_id: "job-1" }, created_at: "2026-09-01T00:00:00Z" };
}

function Reader() {
  return <output>{useNotifyWhenFinished() ? "on" : "off"}</output>;
}

beforeEach(() => {
  localStorage.clear();
  shown.length = 0;
  pageHidden(true);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  pageHidden(false);
});

it("announces nothing until somebody turns it on", () => {
  browserPermission("granted");

  notifyRunFinished(event("run.completed"));

  expect(shown).toEqual([]);
});

it("asks the browser once, and remembers on only when the browser agrees", async () => {
  const granted = browserPermission("default", "granted");
  render(<Reader />);

  let outcome: string | undefined;
  await act(async () => {
    outcome = await setNotifyWhenFinished(true);
  });

  expect(outcome).toBe("on");
  expect(granted.requestPermission).toHaveBeenCalledOnce();
  expect(localStorage.getItem(NOTIFY_WHEN_FINISHED_KEY)).toBe("on");
  expect(screen.getByRole("status").textContent).toBe("on");
});

it("stays off and says so when the browser refuses, or has no notifications at all", async () => {
  browserPermission("default", "denied");
  expect(await setNotifyWhenFinished(true)).toBe("denied");
  expect(localStorage.getItem(NOTIFY_WHEN_FINISHED_KEY)).toBe("off");

  vi.unstubAllGlobals();
  vi.stubGlobal("Notification", undefined);
  expect(await setNotifyWhenFinished(true)).toBe("unsupported");
  expect(localStorage.getItem(NOTIFY_WHEN_FINISHED_KEY)).not.toBe("on");
});

it("announces a finished or failed run with generic words while the page is hidden", async () => {
  browserPermission("granted");
  await setNotifyWhenFinished(true);

  notifyRunFinished(event("run.completed", "run-a"));
  notifyRunFinished(event("run.failed", "run-b"));
  // A reconnect replays events; the same run is announced once.
  notifyRunFinished(event("run.completed", "run-a"));

  expect(shown).toEqual([
    { title: "LM Atelier", options: { body: "A result is ready.", tag: "run:run-a" } },
    { title: "LM Atelier", options: { body: "A request did not finish.", tag: "run:run-b" } },
  ]);
});

it("stays quiet while the page is in view, for cancelled runs, and for other events", async () => {
  browserPermission("granted");
  await setNotifyWhenFinished(true);

  pageHidden(false);
  notifyRunFinished(event("run.completed", "run-visible"));
  pageHidden(true);
  notifyRunFinished(event("run.cancelled", "run-cancelled"));
  notifyRunFinished(event("text.delta", "run-text"));
  notifyRunFinished(event("run.completed", null));

  expect(shown).toEqual([]);
});

it("stays quiet if permission was withdrawn after it was turned on", async () => {
  const browser = browserPermission("granted");
  await setNotifyWhenFinished(true);
  browser.permission = "denied";

  notifyRunFinished(event("run.completed", "run-withdrawn"));

  expect(shown).toEqual([]);
});

it("never throws, even when the browser refuses to construct a notification", async () => {
  class Throwing {
    static permission: NotificationPermission = "granted";
    static requestPermission = vi.fn(async () => "granted" as NotificationPermission);
    constructor() {
      throw new TypeError("Illegal constructor");
    }
  }
  vi.stubGlobal("Notification", Throwing);
  await setNotifyWhenFinished(true);

  expect(() => notifyRunFinished(event("run.completed", "run-throwing"))).not.toThrow();
});
