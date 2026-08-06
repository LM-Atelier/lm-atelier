import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useStudioSession } from "./useStudioSession";
import { api } from "./api";

vi.mock("./api", () => ({
  api: { openStudioSession: vi.fn(), studioSession: vi.fn(), sendTurn: vi.fn() },
}));

function chain(chatId: string, artifactId: string) {
  return {
    id: chatId,
    messages: [
      {
        id: `${chatId}-m1`,
        chat_id: chatId,
        parent_id: null,
        role: "assistant",
        status: "complete",
        created_at: "2026-08-06T00:00:00Z",
        updated_at: "2026-08-06T00:00:00Z",
        parts: [
          {
            id: "p1",
            position: 0,
            type: "image",
            text: null,
            artifact_id: artifactId,
            metadata_json: {},
          },
        ],
      },
    ],
  };
}

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.clearAllMocks();
});

describe("a studio session belongs to one picture", () => {
  it("does not show the previous chain under a newly opened picture", async () => {
    // Opening primes the cache with the session itself, so the mock has to be
    // the whole thing rather than just its id.
    vi.mocked(api.openStudioSession).mockResolvedValue(chain("chat-a", "edit-of-a") as never);
    vi.mocked(api.studioSession).mockResolvedValue(chain("chat-a", "edit-of-a") as never);

    const { result, rerender } = renderHook(
      ({ source }: { source: string }) => useStudioSession(source, null),
      { wrapper, initialProps: { source: "art-a" } },
    );
    await waitFor(() => expect(result.current.steps.length).toBe(2));

    // Switching pictures: the next open has not resolved, so the only session
    // on hand belongs to the picture that is no longer on screen.
    vi.mocked(api.openStudioSession).mockReturnValue(new Promise(() => {}) as never);
    rerender({ source: "art-b" });

    const artifacts = result.current.steps.map((step) => step.artifactId);
    expect(artifacts).not.toContain("edit-of-a");
    // And the surface says so, rather than looking idle and ready to apply.
    await waitFor(() => expect(result.current.busy).toBe(true));
  });

  it("refuses an apply while the picture is still opening", async () => {
    vi.mocked(api.openStudioSession).mockReturnValue(new Promise(() => {}) as never);

    const { result } = renderHook(() => useStudioSession("art-b", null), { wrapper });
    result.current.apply("make it warmer", "art-b");

    await waitFor(() => expect(result.current.error).toBeTruthy());
    expect(api.sendTurn).not.toHaveBeenCalled();
  });

  it("ignores a stored session that was opened for another picture", async () => {
    localStorage.setItem(
      "local-lm-studio-session",
      JSON.stringify({ id: "chat-a", source: "art-a" }),
    );
    vi.mocked(api.openStudioSession).mockReturnValue(new Promise(() => {}) as never);

    const { result } = renderHook(() => useStudioSession("art-b", null), { wrapper });

    expect(api.studioSession).not.toHaveBeenCalled();
    expect(result.current.sessionId).toBeNull();
  });
});
