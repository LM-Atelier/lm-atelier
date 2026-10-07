import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { createElement, useState, type ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import { useChatDeletion } from "./useChatDeletion";
import { recoveryCommand as command, recoveryImpact, recoveryItem } from "./test/recoveryFixtures";
import type { ChatSummary } from "./types";

vi.mock("./api", async (original) => {
  const actual = await original<typeof import("./api")>();
  return { ApiError: actual.ApiError, api: { trashChat: vi.fn(), recoveryImpact: vi.fn(), restoreRecovery: vi.fn() } };
});
const clients: QueryClient[] = [];
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.trashChat).mockResolvedValue(recoveryItem("open"));
  vi.mocked(api.recoveryImpact).mockResolvedValue(recoveryImpact("open", ["restore", "purge"]));
  vi.mocked(api.restoreRecovery).mockResolvedValue({ deletion_id: "deleted-open", kind: "chat", subject_id: "open", action: "restore", replayed: false, reclaimed_bytes: 0 });
});
afterEach(() => { cleanup(); for (const client of clients.splice(0)) client.clear(); localStorage.clear(); });

async function setup(trashNow = true) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  clients.push(client);
  const chats = ["open", "next", "third"].map((id) => ({ id, archived: false } as ChatSummary));
  const key = ["chats", "summaries", "", false];
  client.setQueryData(key, { pages: [{ items: chats, nextOffset: null }], pageParams: [0] });
  client.setQueryData(["chat", "open", "messages", "head"], { messages: ["Garden spacing"] });
  const select = vi.fn();
  function useDeletion() {
    const [current, setCurrent] = useState<string | null>("open");
    const deletion = useChatDeletion({ client, chats, currentChatId: current, activeChatId: current,
      setCurrentChatId: (id) => { select(id); setCurrent(id); } });
    return { ...deletion, current, choose: setCurrent };
  }
  const { result } = renderHook(useDeletion, { wrapper: ({ children }: { children: ReactNode }) => createElement(QueryClientProvider, { client }, children) });
  if (trashNow) await act(async () => { await result.current.mutateAsync({ id: "open", deleteGeneratedMedia: false, command }); });
  return { result, client, key, select };
}

it("restores the original open chat and refreshes its lists after a successful Undo", async () => {
  const { result, client, key } = await setup();
  expect(result.current.current).toBe("next");
  expect(client.getQueryData(["chat", "open", "messages", "head"])).toBeUndefined();
  await act(async () => { await result.current.undo.mutateAsync(result.current.deleted!.item); });
  expect(result.current.current).toBe("open");
  expect(result.current.deleted).toBeNull();
  expect(client.getQueryState(key)?.isInvalidated).toBe(true);
  expect(api.restoreRecovery).toHaveBeenCalledWith("deleted-open", expect.objectContaining({
    expected_revision: "b".repeat(64), restore_unfiled: false,
  }));
});

it("preserves a chat the user opens after deletion", async () => {
  const { result, select } = await setup();
  act(() => result.current.choose("third"));
  await act(async () => { await result.current.undo.mutateAsync(result.current.deleted!.item); });
  expect(result.current.current).toBe("third");
  expect(select.mock.calls).toEqual([["next"]]);
});

it("retries an uncertain Undo with its identical operation key and impact", async () => {
  const { result } = await setup();
  vi.mocked(api.restoreRecovery).mockRejectedValueOnce(new Error("Connection interrupted."));
  const item = result.current.deleted!.item;
  await act(async () => { await expect(result.current.undo.mutateAsync(item)).rejects.toThrow("Connection interrupted."); });
  await waitFor(() => expect(result.current.undo.isError).toBe(true));
  await act(async () => { await result.current.undo.mutateAsync(item); });
  expect(vi.mocked(api.restoreRecovery).mock.calls[1]).toEqual(vi.mocked(api.restoreRecovery).mock.calls[0]);
  expect(api.recoveryImpact).toHaveBeenCalledTimes(1);
});

it("keeps a refused Undo available and requires an explicit new preview", async () => {
  const { result } = await setup();
  const item = result.current.deleted!.item;
  vi.mocked(api.restoreRecovery).mockRejectedValueOnce(new ApiError(409, undefined, "The conversation changed.", "recovery-impact-stale"));
  await act(async () => { await expect(result.current.undo.mutateAsync(item)).rejects.toThrow("The conversation changed."); });
  expect(result.current.deleted?.item.deletion_id).toBe(item.deletion_id);
  vi.mocked(api.recoveryImpact).mockResolvedValue({ ...recoveryImpact("open", ["restore"]), revision: "d".repeat(64) });
  act(() => result.current.recheckUndo());
  expect(api.restoreRecovery).toHaveBeenCalledTimes(1);
  await act(async () => { await result.current.undo.mutateAsync(item); });
  expect(vi.mocked(api.restoreRecovery).mock.calls[1]?.[1].expected_revision).toBe("d".repeat(64));
});

it("does not silently restore unfiled when the original project disappears", async () => {
  const { result } = await setup();
  vi.mocked(api.recoveryImpact).mockResolvedValue({ ...recoveryImpact("open", ["restore"]), conflicts: ["original_project_missing"] });
  vi.mocked(api.restoreRecovery).mockRejectedValue(new ApiError(409, undefined, "Choose to restore this chat unfiled.", "recovery-original-project-missing"));
  await act(async () => { await expect(result.current.undo.mutateAsync(result.current.deleted!.item)).rejects.toThrow("Choose to restore this chat unfiled."); });
  expect(result.current.current).toBe("next");
  expect(vi.mocked(api.restoreRecovery).mock.calls[0]?.[1].restore_unfiled).toBe(false);
  expect(result.current.deleted).not.toBeNull();
});

it("preserves a chat opened while pending Trash is refused", async () => {
  let refuse!: (error: Error) => void;
  vi.mocked(api.trashChat).mockReturnValue(new Promise((_resolve, reject) => { refuse = reject; }));
  const { result, client, key, select } = await setup(false);
  act(() => result.current.mutate({ id: "open", deleteGeneratedMedia: false, command }));
  await waitFor(() => expect(result.current.current).toBe("next"));
  act(() => result.current.choose("third"));
  act(() => refuse(new ApiError(409, undefined, "The conversation changed.", "recovery-impact-stale")));
  await waitFor(() => expect(result.current.isError).toBe(true));
  expect(result.current.current).toBe("third");
  expect(select.mock.calls).toEqual([["next"]]);
  expect(result.current.deleted).toBeNull();
  expect(client.getQueryData<{ pages: { items: ChatSummary[] }[] }>(key)?.pages[0].items.map((chat) => chat.id)).toEqual(["open", "next", "third"]);
  expect(client.getQueryData(["chat", "open", "messages", "head"])).toEqual({ messages: ["Garden spacing"] });
});
