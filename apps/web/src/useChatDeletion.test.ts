import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { createElement, useState, type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { CURRENT_CHAT_KEY } from "./currentChat";
import type { ChatSummary } from "./types";
import { useChatDeletion } from "./useChatDeletion";
import { recoveryCommand as command, recoveryItem } from "./test/recoveryFixtures";

vi.mock("./api", () => ({ api: { trashChat: vi.fn(), recoveryImpact: vi.fn(), restoreRecovery: vi.fn() } }));

afterEach(() => {
  localStorage.clear();
  cleanup();
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
  function useDeletion() {
    const [selected, select] = useState(openChatId);
    return useChatDeletion({ client, chats, currentChatId: selected, activeChatId: selected,
      setCurrentChatId: (id) => { setCurrentChatId(id); select(id ?? ""); } });
  }
  const { result } = renderHook(useDeletion, {
    wrapper: ({ children }: { children: ReactNode }) => createElement(QueryClientProvider, { client }, children),
  });
  const listed = () => (client.getQueryData(PAGES_KEY) as { pages: { items: ChatSummary[] }[] }).pages[0].items
    .map((chat) => chat.id);
  return { result, setCurrentChatId, listed };
}

describe("deleting a chat", () => {
  it("takes it off the list, opens the next chat that is not archived and remembers it", async () => {
    vi.mocked(api.trashChat).mockResolvedValue(recoveryItem("open"));
    const { result, setCurrentChatId, listed } = setup(
      [summary("open"), summary("archived", true), summary("next")],
      "open",
    );

    act(() => result.current.mutate({ id: "open", deleteGeneratedMedia: false, command }));

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(api.trashChat).toHaveBeenCalledWith("open", { ...command, delete_generated_media: false });
    expect(setCurrentChatId).toHaveBeenCalledWith("next");
    expect(localStorage.getItem(CURRENT_CHAT_KEY)).toBe("next");
    expect(listed()).toEqual(["archived", "next"]);
    expect(result.current.deleted?.item.subject_id).toBe("open");
  });

  it("forgets the open chat when no other chat can open", async () => {
    vi.mocked(api.trashChat).mockResolvedValue(recoveryItem("only"));
    localStorage.setItem(CURRENT_CHAT_KEY, "only");
    const { result, setCurrentChatId } = setup([summary("only"), summary("archived", true)], "only");

    act(() => result.current.mutate({ id: "only", deleteGeneratedMedia: true, command }));

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(setCurrentChatId).toHaveBeenCalledWith(null);
    expect(localStorage.getItem(CURRENT_CHAT_KEY)).toBeNull();
  });

  it("puts the chat and the open chat back when the server refuses", async () => {
    vi.mocked(api.trashChat).mockRejectedValue(new Error("refused"));
    const { result, setCurrentChatId, listed } = setup([summary("open"), summary("next")], "open");

    act(() => result.current.mutate({ id: "open", deleteGeneratedMedia: false, command }));

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(setCurrentChatId.mock.calls).toEqual([["next"], ["open"]]);
    expect(localStorage.getItem(CURRENT_CHAT_KEY)).toBe("open");
    expect(listed()).toEqual(["open", "next"]);
    expect(result.current.deleted).toBeNull();
  });
});
