/** Stopping the edit a studio session is running. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", async (original) => ({
  ...(await original<typeof import("./api")>()),
  api: { openStudioSession: vi.fn(), studioSession: vi.fn(), cancelChat: vi.fn() },
}));

const SESSION = { id: "chat-studio", messages: [] };

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.clearAllMocks();
});

async function openStudio() {
  vi.mocked(api.openStudioSession).mockResolvedValue(SESSION as never);
  vi.mocked(api.studioSession).mockResolvedValue(SESSION as never);
  const hook = renderHook(() => useStudioSession("art-1", null), { wrapper });
  await waitFor(() => expect(hook.result.current.sessionId).toBe("chat-studio"));
  return hook;
}

describe("stopping a studio edit", () => {
  it("stops the session's running work and reads the session again", async () => {
    const { result } = await openStudio();
    vi.mocked(api.cancelChat).mockResolvedValue({ id: "job-1", status: "cancelled" } as never);
    const readsBefore = vi.mocked(api.studioSession).mock.calls.length;

    act(() => result.current.stop());

    await waitFor(() => expect(api.cancelChat).toHaveBeenCalledWith("chat-studio"));
    await waitFor(() => expect(vi.mocked(api.studioSession).mock.calls.length).toBeGreaterThan(readsBefore));
    expect(result.current.error).toBeNull();
  });

  it("says nothing when the edit had already ended, and reports any other refusal", async () => {
    const { result } = await openStudio();
    vi.mocked(api.cancelChat).mockRejectedValueOnce(
      new ApiError(409, "chat has no cancellable run", "chat has no cancellable run", "chat-run-absent"),
    );

    act(() => result.current.stop());
    await waitFor(() => expect(result.current.stopping).toBe(false));
    await waitFor(() => expect(api.cancelChat).toHaveBeenCalledTimes(1));
    expect(result.current.error).toBeNull();

    vi.mocked(api.cancelChat).mockRejectedValueOnce(new Error("The connection dropped."));
    act(() => result.current.stop());

    await waitFor(() => expect(result.current.error?.message).toBe("The connection dropped."));
  });
});
