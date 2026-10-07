import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { api } from "./api";
import type { ChatDetail } from "./types";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: { openStudioSession: vi.fn(), studioSession: vi.fn(), sendTurn: vi.fn() },
}));

function session(id: string): ChatDetail {
  return {
    id, project_id: null, title: "Studio", pinned: false, archived: true,
    routing_mode: "image", confirm_uncertain_media: false,
    active_chat_profile_id: null, active_image_profile_id: null,
    active_video_profile_id: null, active_head_message_id: null,
    created_at: "2026-09-19T00:00:00Z", updated_at: "2026-09-19T00:00:00Z",
    messages: [],
  };
}

function deferred() {
  let resolve!: (value: ChatDetail) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<ChatDetail>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function openHook() {
  const first = deferred();
  const second = deferred();
  vi.mocked(api.openStudioSession).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
  vi.mocked(api.studioSession).mockImplementation(async (id) => session(id));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const hook = renderHook(
    ({ source }: { source: string | null }) => useStudioSession(source, null),
    {
      initialProps: { source: "art-a" as string | null },
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    },
  );
  return { ...hook, first, second, client };
}

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.resetAllMocks();
});

describe("opening a different Studio picture", () => {
  it.each(["first", "second"] as const)("keeps the latest picture when %s resolves first", async (order) => {
    const hook = openHook();
    await waitFor(() => expect(api.openStudioSession).toHaveBeenCalledWith("art-a", null));
    hook.rerender({ source: "art-b" });
    await waitFor(() => expect(api.openStudioSession).toHaveBeenCalledWith("art-b", null));
    expect(hook.result.current.sessionId).toBeNull();
    expect(hook.result.current.busy).toBe(true);

    await act(async () => hook[order].resolve(session(order === "first" ? "chat-a" : "chat-b")));
    if (order === "first") {
      expect(hook.result.current.sessionId).toBeNull();
      expect(localStorage.getItem("local-lm-studio-session")).toBeNull();
      await act(async () => hook.second.resolve(session("chat-b")));
    } else {
      await waitFor(() => expect(hook.result.current.sessionId).toBe("chat-b"));
      await act(async () => hook.first.resolve(session("chat-a")));
    }
    await waitFor(() => expect(hook.result.current.sessionId).toBe("chat-b"));
    expect(hook.result.current.session?.id).toBe("chat-b");
    expect(JSON.parse(localStorage.getItem("local-lm-studio-session")!)).toEqual({ id: "chat-b", source: "art-b" });
    expect(api.openStudioSession).toHaveBeenCalledTimes(2);

    vi.mocked(api.sendTurn).mockResolvedValue({} as Awaited<ReturnType<typeof api.sendTurn>>);
    act(() => hook.result.current.apply("Warm the lighting", "art-b"));
    await waitFor(() => expect(api.sendTurn).toHaveBeenCalled());
    expect(vi.mocked(api.sendTurn).mock.calls[0][0]).toBe("chat-b");
  });

  it("ignores an earlier opening failure after the latest picture opens", async () => {
    const hook = openHook();
    await waitFor(() => expect(api.openStudioSession).toHaveBeenCalledTimes(1));
    hook.rerender({ source: "art-b" });
    await waitFor(() => expect(api.openStudioSession).toHaveBeenCalledTimes(2));
    await act(async () => hook.second.resolve(session("chat-b")));
    await waitFor(() => expect(hook.result.current.sessionId).toBe("chat-b"));
    await act(async () => hook.first.reject(new Error("The earlier image could not open.")));
    expect(hook.result.current.error).toBeNull();
    expect(hook.result.current.sessionId).toBe("chat-b");
  });

  it.each(["close", "unmount"] as const)("does not remember a late opening after %s", async (action) => {
    const hook = openHook();
    await waitFor(() => expect(api.openStudioSession).toHaveBeenCalledTimes(1));
    if (action === "close") hook.rerender({ source: null });
    else hook.unmount();
    await act(async () => hook.first.resolve(session("chat-a")));
    expect(localStorage.getItem("local-lm-studio-session")).toBeNull();
    expect(hook.client.getQueryData(["studio-session", "chat-a"])).toBeUndefined();
  });
});
