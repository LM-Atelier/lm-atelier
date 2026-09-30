import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { applyAcceptedChatPages } from "./acceptedChatPages";
import type { Chat, Message, TurnAccepted } from "./types";
import { useChatMessagePages } from "./useChatMessagePages";

vi.mock("./api", () => ({ api: { chatMessages: vi.fn() } }));
beforeEach(() => vi.mocked(api.chatMessages).mockReset());
afterEach(cleanup);

function message(index: number): Message {
  return { id: `message-${index}`, chat_id: "chat-one", parent_id: index ? `message-${index - 1}` : null,
    role: "assistant", status: "complete", parts: [],
    created_at: new Date(Date.UTC(2026, 8, 1, 0, 0, index)).toISOString(),
    updated_at: "2026-09-01T00:00:00Z" };
}

it.each([1, 2])("retains %i loaded pages after an accepted send and fresh server reads", async (count) => {
  const all = Array.from({ length: 102 }, (_, index) => message(index));
  vi.mocked(api.chatMessages).mockImplementation(async (_id, options) => {
    const end = options?.before ? Number(options.before.split("-")[1])
      : Number(options?.headId?.split("-")[1] ?? 99) + 1;
    const start = Math.max(0, end - (options?.limit ?? 40));
    return { chat_id: "chat-one", messages: all.slice(start, end), has_older: start > 0, has_newer: Boolean(options?.before) };
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData<Chat>(["chat", "chat-one"], { id: "chat-one", active_head_message_id: "message-99" } as Chat);
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const { result, rerender } = renderHook(({ head }) => useChatMessagePages("chat-one", head), {
    wrapper, initialProps: { head: "message-99" },
  });
  await waitFor(() => expect(result.current.data).toHaveLength(40));
  if (count === 2) {
    await act(async () => { await result.current.loadOlder(); });
    await waitFor(() => expect(result.current.data).toHaveLength(80));
  }
  act(() => applyAcceptedChatPages(client, "chat-one", {
    user_message: { ...all[100], role: "user" }, assistant_message: all[101], run: {},
  } as TurnAccepted, true));
  rerender({ head: "message-101" });
  await waitFor(() => expect(result.current.data).toHaveLength(count * 40 + 2));

  await act(async () => { await client.invalidateQueries({ queryKey: ["chat", "chat-one"] }); });

  await waitFor(() => expect(result.current.isFetching).toBe(false));
  expect(result.current.data?.map((item) => item.id)).toEqual(all.slice(100 - count * 40).map((item) => item.id));
  expect(vi.mocked(api.chatMessages).mock.calls.every((call) => (call[1]?.limit ?? 40) <= 40)).toBe(true);
});
