import { act, renderHook } from "@testing-library/react";
import { QueryClient } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AppEvent } from "./types";

const handlers: Array<(event: AppEvent) => void> = [];

vi.mock("./api", () => ({
  connectEvents: vi.fn(async (onEvent: (event: AppEvent) => void) => {
    handlers.push(onEvent);
    return () => undefined;
  }),
}));

import { useLiveEvents } from "./useLiveEvents";

function delta(messageId: string, text: string, attempt?: number): AppEvent {
  return {
    sequence: 1,
    type: "text.delta",
    entity_id: "run-1",
    payload: attempt === undefined
      ? { assistant_message_id: messageId, text, job_id: "job-1" }
      : { assistant_message_id: messageId, text, job_id: "job-1", attempt },
    created_at: "2026-09-02T00:00:00Z",
  };
}

async function liveTextAfter(events: AppEvent[]): Promise<Record<string, string>> {
  handlers.length = 0;
  let liveText: Record<string, string> = {};
  const setLiveText = (update: Record<string, string> | ((current: Record<string, string>) => Record<string, string>)) => {
    liveText = typeof update === "function" ? update(liveText) : update;
  };
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  renderHook(() => useLiveEvents(client, setLiveText));
  await act(async () => { await Promise.resolve(); });
  expect(handlers.length).toBe(1);
  await act(async () => {
    for (const event of events) handlers[0]!(event);
  });
  return liveText;
}

function progress(jobId: string, attempt: number): AppEvent {
  return {
    sequence: 1,
    type: "job.progress",
    entity_id: jobId,
    payload: {
      job: {
        id: jobId,
        kind: "chat",
        status: "running",
        run_id: "run-1",
        progress: 0,
        phase: "running",
        payload_json: {},
        result_json: {},
        error: null,
        attempt,
        cancellable: true,
        created_at: "2026-09-02T00:00:00Z",
        updated_at: "2026-09-02T00:00:00Z",
        started_at: null,
        completed_at: null,
      },
    },
    created_at: "2026-09-02T00:00:00Z",
  };
}

afterEach(() => { vi.clearAllMocks(); });

describe("useLiveEvents attempt fence", () => {
  it("fences a late delta of an earlier attempt once a job snapshot announced a later one", async () => {
    const text = await liveTextAfter([
      delta("m1", "attempt one ", 1),
      progress("job-1", 2),
      delta("m1", "late from attempt one", 1),
    ]);
    expect(text.m1).toBe("attempt one ");
  });

  it("keeps appending under a job snapshot of the same attempt", async () => {
    const text = await liveTextAfter([
      delta("m1", "first ", 1),
      progress("job-1", 1),
      delta("m1", "second", 1),
    ]);
    expect(text.m1).toBe("first second");
  });

  it("drops an older attempt's delta once a newer attempt has spoken", async () => {
    const text = await liveTextAfter([
      delta("m1", "first ", 1),
      delta("m1", "second ", 2),
      delta("m1", "late from attempt one", 1),
      delta("m1", "more", 2),
    ]);
    expect(text.m1).toBe("second more");
  });

  it("starts a newer attempt's message over instead of appending to the old text", async () => {
    const text = await liveTextAfter([
      delta("m1", "old attempt said this", 1),
      delta("m1", "new", 2),
    ]);
    expect(text.m1).toBe("new");
  });

  it("keeps appending deltas of one attempt, and of events that name no attempt", async () => {
    const text = await liveTextAfter([
      delta("m1", "a", 3),
      delta("m1", "b", 3),
      delta("m2", "x"),
      delta("m2", "y"),
    ]);
    expect(text.m1).toBe("ab");
    expect(text.m2).toBe("xy");
  });
});
