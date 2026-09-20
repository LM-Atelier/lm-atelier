import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { createChatActivitySeenStore } from "./chatActivitySeen";
import { useChatPages } from "./useChatPages";
import { useLiveEvents } from "./useLiveEvents";
import type { AppEvent, ChatSummary } from "./types";

let onEvent: (event: AppEvent) => void;
let onReconnect: () => void;
vi.mock("./api", () => ({
  api: { chatSummaries: vi.fn() },
  connectEvents: vi.fn(async (event: typeof onEvent, _connected: (value: boolean) => void, reconnect: typeof onReconnect) => {
    onEvent = event;
    onReconnect = reconnect;
    return () => undefined;
  }),
}));

const stamp = "2026-09-20T00:00:00Z";
const summary: ChatSummary = {
  id: "chat", title: "Color study", project_id: null, archived: false, pinned: false,
  created_at: stamp, updated_at: stamp,
  activity: { active_work_count: 1, unresolved_failed_count: 0, last_failure: null,
    last_output: { id: "first", sequence: 1, message_id: "message", response_revision_id: "revision", occurred_at: stamp } },
};
const completed: ChatSummary = { ...summary, activity: { ...summary.activity, active_work_count: 0,
  last_output: { ...summary.activity.last_output!, id: "later", sequence: 2 } } };

beforeEach(() => { vi.mocked(api.chatSummaries).mockResolvedValue([summary]); });
afterEach(cleanup);

it.each(["replay gap", "reconnect", "polling"])("reconciles authoritative summaries after %s without acknowledging new output", async (trigger) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const seen = createChatActivitySeenStore(null);
  seen.markSeen(summary.id, summary.activity.last_output!);
  const setText = vi.fn();
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const { result, unmount } = renderHook(() => {
    useLiveEvents(client, setText);
    return useChatPages();
  }, { wrapper });
  await waitFor(() => expect(result.current.data?.[0].activity.active_work_count).toBe(1));
  vi.mocked(api.chatSummaries).mockResolvedValue([completed]);
  act(() => {
    if (trigger === "replay gap") onEvent({ type: "events.replay_gap", sequence: 8, entity_id: "", payload: {}, created_at: stamp });
    if (trigger === "reconnect") onReconnect();
  });
  await waitFor(() => expect(result.current.data?.[0].activity).toEqual(completed.activity), { timeout: 4_500 });
  expect(seen.hasSeen(summary.id, summary.activity.last_output)).toBe(true);
  expect(seen.hasSeen(summary.id, result.current.data![0].activity.last_output)).toBe(false);
  unmount();
  client.clear();
});
