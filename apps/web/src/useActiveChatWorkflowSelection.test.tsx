import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { useActiveChatWorkflowSelection } from "./useActiveChatWorkflowSelection";
import type { RoutingMode, WorkflowFamily } from "./types";

vi.mock("./api", () => ({
  api: {
    workflowFamilies: vi.fn(),
    chatWorkflowSelections: vi.fn(),
    setChatWorkflowSelection: vi.fn(),
  },
}));

function family(id: string, capability: "chat" | "image" | "video"): WorkflowFamily {
  return {
    id,
    name: id,
    description: "",
    use_case: "",
    tags: [],
    enabled: true,
    archived: false,
    compatibility: false,
    variants: [{
      id: `${id}-variant`,
      variant_key: capability === "image" ? "create" : capability,
      name: capability,
      operation: capability === "chat" ? "text" : `text_to_${capability}`,
      current_revision_id: `${id}-revision`,
      current_revision_version: 1,
      engine: capability === "chat" ? "llama.cpp" : "comfyui",
      capabilities: [capability],
      trusted: true,
      readiness: "ready",
      readiness_reason: null,
    }],
    preferences: [{
      selector_capability: capability,
      enabled: true,
      is_default: false,
      sort_order: 0,
    }],
    created_at: "2026-08-08T00:00:00Z",
    updated_at: "2026-08-08T00:00:00Z",
  };
}

function wrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

describe("useActiveChatWorkflowSelection", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("does not guess or query a workflow while routing is Auto", () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { result } = renderHook(
      () => useActiveChatWorkflowSelection("chat-1", "auto"),
      { wrapper: wrapper(client) },
    );

    expect(result.current).toEqual({ kind: "unresolved", capability: null });
    expect(api.workflowFamilies).not.toHaveBeenCalled();
    expect(api.chatWorkflowSelections).not.toHaveBeenCalled();
  });

  it.each([
    ["text", "chat"],
    ["image", "image"],
    ["video", "video"],
  ] as const)("loads exactly the %s mode's %s capability", async (mode, capability) => {
    vi.mocked(api.workflowFamilies).mockResolvedValue([family(`family-${capability}`, capability)]);
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { result } = renderHook(
      () => useActiveChatWorkflowSelection("chat-1", mode),
      { wrapper: wrapper(client) },
    );

    await waitFor(() => expect(result.current.kind).toBe("ready"));
    expect(api.workflowFamilies).toHaveBeenCalledWith(capability);
    expect(api.workflowFamilies).not.toHaveBeenCalledWith("vision");
    expect(result.current.capability).toBe(capability);
  });

  it("persists the exact chat and capability captured when the choice was made", async () => {
    vi.mocked(api.workflowFamilies).mockImplementation(async (capability) => [
      family(`family-${capability}`, capability as "chat" | "image" | "video"),
    ]);
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
    let finish!: () => void;
    vi.mocked(api.setChatWorkflowSelection).mockImplementation(
      () => new Promise((resolve) => {
        finish = () => resolve({
          selector_capability: "image",
          mode: "family",
          workflow_family_id: "family-image",
          workflow_revision_id: null,
          legacy_profile_id: null,
        });
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const initialProps: { chatId: string; mode: RoutingMode } = {
      chatId: "chat-image",
      mode: "image",
    };
    const { result, rerender } = renderHook(
      ({ chatId, mode }) => useActiveChatWorkflowSelection(chatId, mode),
      {
        wrapper: wrapper(client),
        initialProps,
      },
    );
    await waitFor(() => expect(result.current.kind).toBe("ready"));

    act(() => {
      if (result.current.kind === "ready") {
        result.current.choose({ mode: "family", workflow_family_id: "family-image" });
      }
    });
    await waitFor(() => expect(api.setChatWorkflowSelection).toHaveBeenCalledOnce());
    rerender({ chatId: "chat-video", mode: "video" });
    await waitFor(() => expect(result.current.kind).toBe("ready"));
    expect(result.current.kind === "ready" && result.current.saving).toBe(false);
    await act(async () => finish());

    expect(api.setChatWorkflowSelection).toHaveBeenCalledWith(
      "chat-image",
      "image",
      { mode: "family", workflow_family_id: "family-image" },
    );
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ["chat", "chat-image", "workflow-selections"],
    });
  });

  it("keeps the save pending until the authoritative selection is reconciled", async () => {
    vi.mocked(api.workflowFamilies).mockResolvedValue([family("family-image", "image")]);
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
    vi.mocked(api.setChatWorkflowSelection).mockResolvedValue({
      selector_capability: "image",
      mode: "automatic",
      workflow_family_id: null,
      workflow_revision_id: null,
      legacy_profile_id: null,
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    let reconcile!: () => void;
    vi.spyOn(client, "invalidateQueries").mockImplementation(
      () => new Promise((resolve) => {
        reconcile = () => resolve();
      }),
    );
    const { result } = renderHook(
      () => useActiveChatWorkflowSelection("chat-1", "image"),
      { wrapper: wrapper(client) },
    );
    await waitFor(() => expect(result.current.kind).toBe("ready"));

    act(() => {
      if (result.current.kind === "ready") result.current.choose({ mode: "automatic" });
    });
    await waitFor(() => {
      expect(api.setChatWorkflowSelection).toHaveBeenCalledOnce();
      expect(reconcile).toBeTypeOf("function");
    });
    expect(result.current.kind === "ready" && result.current.saving).toBe(true);

    await act(async () => reconcile());
    await waitFor(() => {
      expect(result.current.kind === "ready" && result.current.saving).toBe(false);
    });
  });

  it("does not show a failed choice on a different chat or capability", async () => {
    vi.mocked(api.workflowFamilies).mockImplementation(async (capability) => [
      family(`family-${capability}`, capability as "chat" | "image" | "video"),
    ]);
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
    vi.mocked(api.setChatWorkflowSelection).mockRejectedValue(new Error("image save failed"));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const initialProps: { chatId: string; mode: RoutingMode } = {
      chatId: "chat-image",
      mode: "image",
    };
    const { result, rerender } = renderHook(
      ({ chatId, mode }) => useActiveChatWorkflowSelection(chatId, mode),
      { wrapper: wrapper(client), initialProps },
    );
    await waitFor(() => expect(result.current.kind).toBe("ready"));
    act(() => {
      if (result.current.kind === "ready") result.current.choose({ mode: "automatic" });
    });
    await waitFor(() => expect(
      result.current.kind === "ready" && result.current.saveError?.message,
    ).toBe("image save failed"));

    rerender({ chatId: "chat-video", mode: "video" });
    await waitFor(() => expect(result.current.kind).toBe("ready"));
    expect(result.current.kind === "ready" && result.current.saveError).toBeNull();
    expect(result.current.kind === "ready" && result.current.saving).toBe(false);
  });

  it("reports a read failure instead of presenting Default as confirmed", async () => {
    vi.mocked(api.workflowFamilies).mockRejectedValue(new Error("families unavailable"));
    vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { result } = renderHook(
      () => useActiveChatWorkflowSelection("chat-1", "image"),
      { wrapper: wrapper(client) },
    );

    await waitFor(() => expect(result.current.kind).toBe("read-error"));
    expect(result.current.kind === "read-error" && result.current.error.message)
      .toBe("families unavailable");
  });
});
