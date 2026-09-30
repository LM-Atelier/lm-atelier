import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import type { ChatMessageWindow, Message } from "./types";
import { useChatMessagePages } from "./useChatMessagePages";

vi.mock("./api", () => ({ api: { chatMessages: vi.fn() } }));
beforeEach(() => { vi.mocked(api.chatMessages).mockReset(); });
afterEach(cleanup);

function message(index: number): Message {
  return {
    id: `message-${index}`, chat_id: "chat-one", parent_id: index ? `message-${index - 1}` : null,
    role: "assistant", status: "complete", parts: [],
    created_at: "2026-09-01T00:00:00Z", updated_at: "2026-09-01T00:00:00Z",
  };
}

function page(indices: number[], older = false): ChatMessageWindow {
  return { chat_id: "chat-one", messages: indices.map(message), has_older: older, has_newer: false };
}

function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>
    {children}
  </QueryClientProvider>;
  return { client, wrapper };
}

it("loads the latest window and prepends older messages using its first anchor", async () => {
  vi.mocked(api.chatMessages).mockResolvedValueOnce(page([4, 5, 6, 7], true))
    .mockResolvedValueOnce(page([0, 1, 2, 3]));
  const { wrapper } = setup();
  const { result } = renderHook(() => useChatMessagePages("chat-one", "message-7"), { wrapper });
  await waitFor(() => expect(result.current.data).toEqual([4, 5, 6, 7].map(message)));

  await act(async () => { await result.current.loadOlder(); });

  await waitFor(() => expect(result.current.data).toEqual([0, 1, 2, 3, 4, 5, 6, 7].map(message)));
  expect(result.current.hasNextPage).toBe(false);
  expect(api.chatMessages).toHaveBeenLastCalledWith("chat-one", {
    headId: "message-7", before: "message-4", limit: 40, signal: expect.any(AbortSignal),
  });
});

it("keeps loaded messages when an older read fails and retries the same anchor", async () => {
  vi.mocked(api.chatMessages).mockResolvedValueOnce(page([2, 3], true))
    .mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(page([0, 1]));
  const { wrapper } = setup();
  const { result } = renderHook(() => useChatMessagePages("chat-one", "message-3"), { wrapper });
  await waitFor(() => expect(result.current.data).toHaveLength(2));

  await act(async () => { await result.current.loadOlder(); });
  await waitFor(() => expect(result.current.isFetchNextPageError).toBe(true));
  expect(result.current.data).toEqual([2, 3].map(message));
  await act(async () => { await result.current.loadOlder(); });

  await waitFor(() => expect(result.current.data).toEqual([0, 1, 2, 3].map(message)));
  expect(vi.mocked(api.chatMessages).mock.calls.map((call) => call[1]?.before))
    .toEqual([undefined, "message-2", "message-2"]);
});

it("cancels a previous branch read and never folds its late answer into the next branch", async () => {
  let finishOld: (value: ChatMessageWindow) => void = () => undefined;
  vi.mocked(api.chatMessages).mockImplementation((_id, options) => options?.headId === "old-head"
    ? new Promise((resolve) => { finishOld = resolve; }) : Promise.resolve(page([8, 9])));
  const { wrapper } = setup();
  const { result, rerender } = renderHook(({ head }) => useChatMessagePages("chat-one", head), {
    wrapper, initialProps: { head: "old-head" },
  });
  await waitFor(() => expect(api.chatMessages).toHaveBeenCalledTimes(1));
  const oldSignal = vi.mocked(api.chatMessages).mock.calls[0][1]?.signal;

  rerender({ head: "new-head" });
  await waitFor(() => expect(result.current.data).toEqual([8, 9].map(message)));
  expect(oldSignal?.aborted).toBe(true);
  await act(async () => { finishOld(page([0, 1])); });

  expect(result.current.data).toEqual([8, 9].map(message));
});

it("keeps different chats separate even when their head selection is empty", async () => {
  let finishNext: (value: ChatMessageWindow) => void = () => undefined;
  vi.mocked(api.chatMessages).mockResolvedValueOnce(page([0, 1]))
    .mockImplementationOnce(() => new Promise((resolve) => { finishNext = resolve; }));
  const { wrapper } = setup();
  const { result, rerender } = renderHook(({ id }) => useChatMessagePages(id, null), {
    wrapper, initialProps: { id: "chat-one" },
  });
  await waitFor(() => expect(result.current.data).toHaveLength(2));

  rerender({ id: "chat-two" });
  expect(result.current.data).toBeUndefined();
  await act(async () => { finishNext({ ...page([10]), chat_id: "chat-two" }); });

  await waitFor(() => expect(result.current.data).toEqual([message(10)]));
});

it("shows overlapping messages once and keeps their latest-window version", async () => {
  const latest = page([2, 3], true);
  latest.messages[0] = { ...message(2), status: "pending" };
  vi.mocked(api.chatMessages).mockResolvedValueOnce(latest).mockResolvedValueOnce(page([0, 1, 2]));
  const { wrapper } = setup();
  const { result } = renderHook(() => useChatMessagePages("chat-one", "message-3"), { wrapper });
  await waitFor(() => expect(result.current.data).toHaveLength(2));

  await act(async () => { await result.current.loadOlder(); });

  await waitFor(() => expect(result.current.data?.map((item) => item.id))
    .toEqual([0, 1, 2, 3].map((id) => `message-${id}`)));
  expect(result.current.data?.find((item) => item.id === "message-2")?.status).toBe("pending");
});

it("does not cancel or duplicate an older read when it is requested twice", async () => {
  let finishOlder: (value: ChatMessageWindow) => void = () => undefined;
  vi.mocked(api.chatMessages).mockResolvedValueOnce(page([2, 3], true))
    .mockImplementationOnce(() => new Promise((resolve) => { finishOlder = resolve; }));
  const { wrapper } = setup();
  const { result } = renderHook(() => useChatMessagePages("chat-one", "message-3"), { wrapper });
  await waitFor(() => expect(result.current.data).toHaveLength(2));
  act(() => { void result.current.loadOlder(); void result.current.loadOlder(); });
  await waitFor(() => expect(api.chatMessages).toHaveBeenCalledTimes(2));
  const signal = vi.mocked(api.chatMessages).mock.calls[1][1]?.signal;
  expect(signal?.aborted).toBe(false);

  await act(async () => { finishOlder(page([0, 1])); });

  await waitFor(() => expect(result.current.data).toHaveLength(4));
  expect(api.chatMessages).toHaveBeenCalledTimes(2);
});

it("does not read messages before a chat is selected", () => {
  const { wrapper } = setup();
  renderHook(() => useChatMessagePages(null, null), { wrapper });
  expect(api.chatMessages).not.toHaveBeenCalled();
});
