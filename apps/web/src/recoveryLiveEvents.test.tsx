import { QueryClient } from "@tanstack/react-query";
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { AppEvent } from "./types";
import { useLiveEvents } from "./useLiveEvents";

const handlers: Array<(event: AppEvent) => void> = [];
const reconnects: Array<() => void> = [];
const clients: QueryClient[] = [];
vi.mock("./api", () => ({ connectEvents: vi.fn(async (onEvent: (event: AppEvent) => void,
  _onState: (connected: boolean) => void, onReconnect: () => void) => {
  handlers.push(onEvent); reconnects.push(onReconnect); return () => undefined;
}) }));

beforeEach(() => { handlers.length = 0; reconnects.length = 0; });
afterEach(() => { cleanup(); for (const client of clients.splice(0)) client.clear(); vi.useRealTimers(); });

async function show() {
  const client = new QueryClient();
  clients.push(client);
  for (const key of [["recovery-items", "", ""], ["recovery-items", "blocked", "2026-10-01"], ["recovery-impact", "deleted-garden", "one-command"], ["artifact-library-v1", "image"], ["artifacts"], ["artifact-storage"]])
    client.setQueryData(key, {});
  renderHook(() => useLiveEvents(client, vi.fn()));
  await act(async () => { await Promise.resolve(); });
  return client;
}

it("refreshes every recovery list after another client changes membership without replacing an action preview", async () => {
  const client = await show();
  act(() => handlers[0]({ sequence: 1, type: "recovery.updated", entity_id: "deleted-garden",
    payload: {}, created_at: "2026-10-02T00:00:00Z" }));
  expect(client.getQueryState(["recovery-items", "", ""])?.isInvalidated).toBe(true);
  expect(client.getQueryState(["recovery-items", "blocked", "2026-10-01"])?.isInvalidated).toBe(true);
  expect(client.getQueryState(["recovery-impact", "deleted-garden", "one-command"])?.isInvalidated).toBe(false);
  for (const key of ["artifact-library-v1", "artifacts", "artifact-storage"])
    expect(client.getQueryState(key === "artifact-library-v1" ? [key, "image"] : [key])?.isInvalidated).toBe(true);
});

it.each(["gap", "reconnect"])("refreshes recovery membership after a %s", async (reason) => {
  vi.useFakeTimers();
  const client = await show();
  act(() => {
    if (reason === "gap") handlers[0]({ sequence: 1, type: "events.replay_gap", entity_id: null,
      payload: {}, created_at: "2026-10-02T00:00:00Z" });
    else reconnects[0]();
  });
  await act(async () => { await vi.advanceTimersByTimeAsync(100); });
  expect(client.getQueryState(["recovery-items", "", ""])?.isInvalidated).toBe(true);
  expect(client.getQueryState(["recovery-items", "blocked", "2026-10-01"])?.isInvalidated).toBe(true);
  expect(client.getQueryState(["recovery-impact", "deleted-garden", "one-command"])?.isInvalidated).toBe(false);
  expect(client.getQueryState(["artifact-library-v1", "image"])?.isInvalidated).toBe(true);
});

it("refreshes workflow selectors after another client changes a family without replacing loaded chat history", async () => {
  const client = await show();
  const keys = ["workflow-families", "workflow-family", "workflows", "workflow-revision", "workflow-ready-revisions", "studio-capabilities"];
  for (const key of [...keys, "chat", "workflow-deletion-impact"])
    client.setQueryData([key, "garden"], {});
  act(() => handlers[0]({ sequence: 1, type: "workflow.updated", entity_id: "family-garden",
    payload: {}, created_at: "2026-10-02T00:00:00Z" }));
  for (const key of keys) expect(client.getQueryState([key, "garden"])?.isInvalidated).toBe(true);
  for (const key of ["chat", "workflow-deletion-impact"])
    expect(client.getQueryState([key, "garden"])?.isInvalidated).toBe(false);
});
