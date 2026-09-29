import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { CURRENT_CHAT_KEY } from "./currentChat";
import type { ChatSummary } from "./types";
import { useChatDeletion } from "./useChatDeletion";

vi.mock("./api", () => ({ api: { deleteChat: vi.fn() } }));

afterEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
});

function summary(id: string, archived = false): ChatSummary {
  return {
    id,
    project_id: null,
    title: id,
    archived,
    pinned: false,
    created_at: "2026-09-29T00:00:00Z",
    updated_at: "2026-09-29T00:00:00Z",
    activity: {} as ChatSummary["activity"],
  };
}

const PAGES_KEY = ["chats", "summaries", "", false];

function setup(chats: ChatSummary[], openChatId: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  client.setQueryData(PAGES_KEY, { pages: [{ items: chats, nextOffset: null }], pageParams: [0] });
  const setCurrentChatId = vi.fn();
  const chatDrafts: Record<string, unknown> = { [openChatId]: { title: "draft" } };
  const composerDrafts: Record<string, unknown> = { [openChatId]: { text: "unsent", promptSource: null } };
  const setChatDrafts = vi.fn((update: (current: never) => unknown) => update(chatDrafts as never));
  const setComposerDrafts = vi.fn((update: (current: never) => unknown) => update(composerDrafts as never));
  const { result } = renderHook(() => useChatDeletion({
    client,
    chats,
    currentChatId: openChatId,
    activeChatId: openChatId,
    setCurrentChatId,
    setChatDrafts,
    setComposerDrafts,
  }), {
    wrapper: ({ children }: { children: ReactNode }) => createElement(QueryClientProvider, { client }, children),
  });
  const listed = () => (client.getQueryData(PAGES_KEY) as { pages: { items: ChatSummary[] }[] }).pages[0].items
    .map((chat) => chat.id);
  return { result, setCurrentChatId, setChatDrafts, setComposerDrafts, listed };
}

describe("deleting a chat", () => {
  it("takes it off the list, opens the next chat that is not archived and remembers it", async () => {
    vi.mocked(api.deleteChat).mockResolvedValue(undefined as never);
    const { result, setCurrentChatId, setChatDrafts, setComposerDrafts, listed } = setup(
      [summary("open"), summary("archived", true), summary("next")],
      "open",
    );

    act(() => result.current.mutate({ id: "open", deleteGeneratedMedia: false }));

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(api.deleteChat).toHaveBeenCalledWith("open", false);
    expect(setCurrentChatId).toHaveBeenCalledWith("next");
    expect(localStorage.getItem(CURRENT_CHAT_KEY)).toBe("next");
    expect(listed()).toEqual(["archived", "next"]);
    expect(setChatDrafts.mock.results[0].value).toEqual({});
    expect(setComposerDrafts.mock.results[0].value).toEqual({});
  });

  it("forgets the open chat when no other chat can open", async () => {
    vi.mocked(api.deleteChat).mockResolvedValue(undefined as never);
    localStorage.setItem(CURRENT_CHAT_KEY, "only");
    const { result, setCurrentChatId } = setup([summary("only"), summary("archived", true)], "only");

    act(() => result.current.mutate({ id: "only", deleteGeneratedMedia: true }));

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(setCurrentChatId).toHaveBeenCalledWith(null);
    expect(localStorage.getItem(CURRENT_CHAT_KEY)).toBeNull();
  });

  it("puts the chat and the open chat back when the server refuses", async () => {
    vi.mocked(api.deleteChat).mockRejectedValue(new Error("refused"));
    const { result, setCurrentChatId, setChatDrafts, listed } = setup([summary("open"), summary("next")], "open");

    act(() => result.current.mutate({ id: "open", deleteGeneratedMedia: false }));

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(setCurrentChatId.mock.calls).toEqual([["next"], ["open"]]);
    expect(localStorage.getItem(CURRENT_CHAT_KEY)).toBe("open");
    expect(listed()).toEqual(["open", "next"]);
    expect(setChatDrafts).not.toHaveBeenCalled();
  });
});
