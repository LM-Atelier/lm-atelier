import { QueryClient } from "@tanstack/react-query";
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { AppEvent, Job } from "./types";
import { useLiveEvents } from "./useLiveEvents";

const handlers: Array<(event: AppEvent) => void> = [];
const clients: QueryClient[] = [];
vi.mock("./api", () => ({ connectEvents: vi.fn(async (onEvent: (event: AppEvent) => void) => {
  handlers.push(onEvent); return () => undefined;
}) }));

const LIBRARY = [["artifact-library-v1", "video"], ["artifacts"], ["artifact-storage"]];

beforeEach(() => { handlers.length = 0; });
afterEach(() => { cleanup(); for (const client of clients.splice(0)) client.clear(); });

async function show() {
  const client = new QueryClient();
  clients.push(client);
  for (const key of LIBRARY) client.setQueryData(key, {});
  renderHook(() => useLiveEvents(client, vi.fn()));
  await act(async () => { await Promise.resolve(); });
  return client;
}

function finished(kind: string, status: string) {
  const job = { id: "job-clip", kind, status, attempt: 1, updated_at: "2026-10-06T00:00:00Z" } as unknown as Job;
  act(() => handlers[0]({ sequence: 1, type: "job.progress", entity_id: "job-clip",
    payload: { job }, created_at: "2026-10-06T00:00:00Z" }));
}

it("shows a saved frame or a trimmed video in the library wherever the job was started", async () => {
  const client = await show();
  finished("media_utility", "complete");
  for (const key of LIBRARY) expect(client.getQueryState(key)?.isInvalidated).toBe(true);
});

it.each([["media_utility", "running"], ["media_utility", "failed"], ["image", "complete"]])(
  "leaves the library alone for a %s job that is %s",
  async (kind, status) => {
    const client = await show();
    finished(kind, status);
    for (const key of LIBRARY) expect(client.getQueryState(key)?.isInvalidated).toBe(false);
  },
);
