import { act, renderHook } from "@testing-library/react";
import { QueryClient } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AppEvent, Job, JobActivity } from "./types";

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

async function liveTextAfter(
  events: AppEvent[],
  cachedJobs?: Array<{ id: string; attempt: number }>,
): Promise<Record<string, string>> {
  handlers.length = 0;
  let liveText: Record<string, string> = {};
  const setLiveText = (update: Record<string, string> | ((current: Record<string, string>) => Record<string, string>)) => {
    liveText = typeof update === "function" ? update(liveText) : update;
  };
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  if (cachedJobs) {
    // What a reconnect actually starts from: the job list is already in the
    // cache, and it names the attempt each job is on.
    client.setQueryData(["jobs"], cachedJobs);
  }
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

  it("fences an older delta after a reconnect, before any progress event arrives", async () => {
    // The fence is rebuilt on every connect. Seeded only by `job.progress`, it
    // was empty for the whole window between reconnecting and the first
    // progress event - and an in-flight retry's predecessor can still be
    // speaking in that window. Its delta was accepted with nothing to measure
    // it against, and overwrote text the newer attempt had already written.
    const liveText = await liveTextAfter(
      [delta("m1", "from the attempt that was replaced", 1)],
      [{ id: "job-1", attempt: 2 }],
    );

    expect(liveText.m1).toBeUndefined();
  });

  it("still accepts the current attempt's delta after a reconnect", async () => {
    // Seeding must fence what is older and nothing else: a fence that dropped
    // the live attempt's own text would be worse than the gap it closes.
    const liveText = await liveTextAfter(
      [delta("m1", "from the attempt that is running", 2)],
      [{ id: "job-1", attempt: 2 }],
    );

    expect(liveText.m1).toBe("from the attempt that is running");
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


describe("useLiveEvents activity projection", () => {
  const clients: QueryClient[] = [];
  afterEach(() => {
    clients.splice(0).forEach((client) => client.clear());
    vi.useRealTimers();
  });
  async function openActivity(activity: JobActivity, limit = 100) {
    handlers.length = 0;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    clients.push(client);
    const key = ["jobs", "activity", limit];
    client.setQueryData(key, activity);
    const setText = vi.fn();
    const hook = renderHook(() => useLiveEvents(client, setText));
    await act(async () => { await Promise.resolve(); });
    return { client, key, setText, hook, invalidate: vi.spyOn(client, "invalidateQueries") };
  }
  function activityJob(attempt = 1): Job {
    return progress("job-1", attempt).payload.job as Job;
  }
  it("updates visible progress in every activity window without changing the total", async () => {
    const job = activityJob();
    const current = { active: [job], active_count: 501, recent_issues: [] };
    const { client, key, invalidate } = await openActivity(current);
    const expanded = ["jobs", "activity", 200];
    client.setQueryData(expanded, current);
    const event = progress(job.id, 1);
    event.payload.job = { ...job, phase: "Downloading model", progress: 0.5 };
    act(() => handlers[0]!(event));
    for (const queryKey of [key, expanded]) {
      expect(client.getQueryData<JobActivity>(queryKey)).toEqual({
        ...current, active: [event.payload.job],
      });
    }
    expect(invalidate).not.toHaveBeenCalled();
  });
  it("coalesces membership changes into an authoritative count refresh", async () => {
    vi.useFakeTimers();
    const job = activityJob();
    const current = { active: [job], active_count: 501, recent_issues: [] };
    const { client, key, invalidate } = await openActivity(current);
    const event = progress(job.id, 1);
    event.payload.job = { ...job, status: "complete", progress: 1 };
    act(() => { handlers[0]!(event); handlers[0]!(event); });
    expect(client.getQueryData(key)).toEqual(current);
    await act(async () => { await vi.advanceTimersByTimeAsync(100); });
    expect(invalidate).toHaveBeenCalledExactlyOnceWith({ queryKey: ["jobs", "activity"] });
  });
  it("refreshes when a new active job arrives in a complete visible list", async () => {
    vi.useFakeTimers();
    const { invalidate } = await openActivity({ active: [], active_count: 0, recent_issues: [] });
    act(() => handlers[0]!(progress("new-job", 1)));
    await act(async () => { await vi.advanceTimersByTimeAsync(100); });
    expect(invalidate).toHaveBeenCalledExactlyOnceWith({ queryKey: ["jobs", "activity"] });
  });
  it("keeps hidden active progress and verification jobs from causing refresh fanout", async () => {
    vi.useFakeTimers();
    const { invalidate } = await openActivity({ active: [activityJob()], active_count: 501, recent_issues: [] });
    const verification = progress("verification", 1);
    verification.payload.job = { ...activityJob(), id: "verification", kind: "edit_verify", status: "complete" };
    act(() => { handlers[0]!(progress("hidden-active", 1)); handlers[0]!(verification); });
    await act(async () => { await vi.advanceTimersByTimeAsync(100); });
    expect(invalidate).not.toHaveBeenCalled();
  });
  it("seeds the attempt fence from activity rows after reconnecting", async () => {
    const { setText } = await openActivity({ active: [activityJob(2)], active_count: 1, recent_issues: [] });
    act(() => handlers[0]!(delta("message", "Earlier attempt", 1)));
    expect(setText).not.toHaveBeenCalled();
  });
  it("does not replace activity with an older attempt or older stored snapshot", async () => {
    const job = { ...activityJob(2), updated_at: "2026-09-02T00:00:02Z" };
    const current = { active: [job], active_count: 1, recent_issues: [] };
    const { client, key } = await openActivity(current);
    act(() => {
      handlers[0]!(progress(job.id, 1));
      handlers[0]!(progress(job.id, 2));
    });
    expect(client.getQueryData(key)).toEqual(current);
  });
});

describe("useLiveEvents accepted queue reconciliation", () => {
  const clients: QueryClient[] = [];
  afterEach(() => {
    clients.splice(0).forEach((client) => client.clear());
    vi.useRealTimers();
  });
  async function openQueue() {
    vi.useFakeTimers(); handlers.length = 0;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    clients.push(client);
    const key = ["jobs", "queue", "all"];
    const data = { pages: [{ items: [], total: 0 }], pageParams: [null] };
    client.setQueryData(key, data);
    const hook = renderHook(() => useLiveEvents(client, vi.fn()));
    await act(async () => { await Promise.resolve(); });
    return { client, key, data, hook, invalidate: vi.spyOn(client, "invalidateQueries") };
  }
  it("coalesces visible progress bursts into one queue read without copying job payloads", async () => {
    const { client, key, data, invalidate } = await openQueue();
    act(() => { for (let i = 0; i < 20; i++) handlers[0]!(progress("job-" + i, 1)); });
    await act(async () => { await vi.advanceTimersByTimeAsync(1_999); });
    expect(invalidate).not.toHaveBeenCalled();
    expect(client.getQueryData(key)).toEqual(data);
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(invalidate).toHaveBeenCalledExactlyOnceWith({ queryKey: ["jobs", "queue"] });
  });
  it("omits verification events and clears a pending queue refresh on unmount", async () => {
    const { hook, invalidate } = await openQueue();
    const hidden = progress("check", 1);
    hidden.payload.job = { ...(hidden.payload.job as Job), kind: "edit_verify" };
    act(() => handlers[0]!(hidden));
    await act(async () => { await vi.advanceTimersByTimeAsync(2_000); });
    expect(invalidate).not.toHaveBeenCalled();
    act(() => handlers[0]!(progress("visible", 1)));
    hook.unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(2_000); });
    expect(invalidate).not.toHaveBeenCalled();
  });
  it("reconciles queue pages after a replay gap", async () => {
    const { client, key } = await openQueue();
    act(() => handlers[0]!({ sequence: 2, type: "events.replay_gap", entity_id: "",
      payload: {}, created_at: "2026-09-02T00:00:00Z" }));
    await act(async () => { await vi.advanceTimersByTimeAsync(100); });
    expect(client.getQueryState(key)?.isInvalidated).toBe(true);
  });
});

describe("useLiveEvents expanded queue step reconciliation", () => {
  afterEach(() => { vi.useRealTimers(); });
  it("refreshes expanded steps on visible job progress even without an item-list cache", async () => {
    vi.useFakeTimers(); handlers.length = 0;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const key = ["jobs", "queue-steps", "plan"];
    client.setQueryData(key, { pages: [{ items: [], total: 0 }], pageParams: [0] });
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const hook = renderHook(() => useLiveEvents(client, vi.fn()));
    await act(async () => { await Promise.resolve(); });
    act(() => { handlers[0]!(progress("one", 1)); handlers[0]!(progress("two", 1)); });
    await act(async () => { await vi.advanceTimersByTimeAsync(2_000); });
    expect(invalidate).toHaveBeenCalledExactlyOnceWith({ queryKey: ["jobs", "queue-steps"] });
    hook.unmount(); client.clear();
  });
});
