/** Missed events leave an enlargement preview out of date, so it is asked again. */

import { act, renderHook } from "@testing-library/react";
import { QueryClient } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import type { AppEvent } from "./types";

const handlers: Array<(event: AppEvent) => void> = [];

vi.mock("./api", () => ({
  connectEvents: vi.fn(async (onEvent: (event: AppEvent) => void) => {
    handlers.push(onEvent);
    return () => undefined;
  }),
}));

import { useLiveEvents } from "./useLiveEvents";

afterEach(() => {
  vi.useRealTimers();
  handlers.length = 0;
});

it("marks every enlargement preview out of date once events were missed", async () => {
  vi.useFakeTimers();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const key = ["studio-enlargement", "chat-studio", "art-1", null, "null"];
  client.setQueryData(key, { version: 1, status: "ready" });
  renderHook(() => useLiveEvents(client, vi.fn()));
  await act(async () => { await Promise.resolve(); });

  act(() => handlers[0]!({ sequence: 2, type: "events.replay_gap", entity_id: "", payload: {}, created_at: "2026-09-30T00:00:00Z" }));
  await act(async () => { await vi.advanceTimersByTimeAsync(100); });

  expect(client.getQueryState(key)?.isInvalidated).toBe(true);
});
