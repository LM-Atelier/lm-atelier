/** Finished work announced from the live connection, without costing the updates that connection carries. */

import { act, cleanup, renderHook } from "@testing-library/react";
import { QueryClient } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { setNotifyWhenFinished } from "./completionNotifications";
import type { AppEvent } from "./types";

const handlers: Array<(event: AppEvent) => void> = [];

vi.mock("./api", () => ({
  connectEvents: vi.fn(async (onEvent: (event: AppEvent) => void) => {
    handlers.push(onEvent);
    return () => undefined;
  }),
}));

import { useLiveEvents } from "./useLiveEvents";

const shown: NotificationOptions[] = [];

function runEvent(type: string, runId: string): AppEvent {
  return { sequence: 3, type, entity_id: runId, payload: { job_id: "job-1" }, created_at: "2026-09-01T00:00:00Z" };
}

async function connected() {
  handlers.length = 0;
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const invalidate = vi.spyOn(client, "invalidateQueries");
  renderHook(() => useLiveEvents(client, () => undefined));
  await act(async () => { await Promise.resolve(); });
  return { invalidate, deliver: (event: AppEvent) => act(() => handlers[0]!(event)) };
}

function invalidatedRoots(invalidate: { mock: { calls: unknown[][] } }): string[] {
  return invalidate.mock.calls.map((call) => String((call[0] as { queryKey?: unknown[] }).queryKey?.[0]));
}

beforeEach(async () => {
  localStorage.clear();
  shown.length = 0;
  Object.defineProperty(document, "hidden", { configurable: true, get: () => true });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  Object.defineProperty(document, "hidden", { configurable: true, get: () => false });
});

it("announces a run the live connection reports finished", async () => {
  vi.stubGlobal("Notification", class {
    static permission: NotificationPermission = "granted";
    static requestPermission = vi.fn(async () => "granted" as NotificationPermission);
    constructor(_title: string, options?: NotificationOptions) {
      shown.push(options ?? {});
    }
  });
  await setNotifyWhenFinished(true);
  const { deliver } = await connected();

  await deliver(runEvent("run.completed", "live-run-1"));

  expect(shown).toEqual([{ body: "A result is ready.", tag: "run:live-run-1" }]);
});

it("keeps every refresh a finished run triggers when the notification itself fails", async () => {
  vi.stubGlobal("Notification", class {
    static permission: NotificationPermission = "granted";
    static requestPermission = vi.fn(async () => "granted" as NotificationPermission);
    constructor() {
      throw new TypeError("Illegal constructor");
    }
  });
  await setNotifyWhenFinished(true);
  const { deliver, invalidate } = await connected();

  await deliver(runEvent("run.completed", "live-run-2"));

  expect(invalidatedRoots(invalidate)).toEqual(
    expect.arrayContaining(["chat", "chats", "jobs", "artifacts", "artifact-storage"]),
  );
  // And the connection keeps delivering afterwards.
  await deliver(runEvent("run.failed", "live-run-3"));
  expect(invalidatedRoots(invalidate).filter((root) => root === "chats")).toHaveLength(2);
});

it("refreshes for a cancelled run without announcing it, even in a browser with no notifications", async () => {
  vi.stubGlobal("Notification", undefined);
  const { deliver, invalidate } = await connected();

  await deliver(runEvent("run.cancelled", "live-run-4"));

  expect(invalidatedRoots(invalidate)).toEqual(expect.arrayContaining(["chat", "chats"]));
  expect(shown).toEqual([]);
});
